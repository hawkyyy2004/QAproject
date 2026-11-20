import csv
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qaproject import knowledge
from qaproject.knowledge_context import explicit_aliases, canonical_name, entity_context
from qaproject.knowledge_repair import enqueue_repair, rejections
from qaproject.pipeline import Worker
from qaproject.store import Store, digest
from qaproject.glm import GLMClient, ProviderError
from qaproject.app import create_app
from test_knowledge import provider, run, TEXT, add_reviewed_fact


def test_context_retains_all_named_facts_but_compacts_unrelated_entities():
    existing=[dict(entity_name='温盐环流',entity_type='Process',entity_description='事实一'),
              dict(entity_name='温盐环流',entity_type='Process',entity_description='事实二'),
              dict(entity_name='珊瑚',entity_type='Organism',entity_description='描述'*1000)]
    related,index=entity_context(existing,[dict(text='温盐环流的变化。')])
    assert related==existing[:2] and index==[dict(entity_name='珊瑚',entity_type='Organism')]
    related,_=entity_context(existing,[dict(text='其他文字')],names=['温盐环流'])
    assert related==existing[:2]
    assert len(json.dumps((related,index))) < len(json.dumps(existing))


def test_aliases_require_explicit_unique_definition_and_type():
    aliases=explicit_aliases([dict(text='Thermohaline Circulation (TC) drives transport.')])
    assert aliases=={'tc':'Thermohaline Circulation'}
    assert canonical_name('TC',aliases,[],'Process')=='Thermohaline Circulation'
    existing=[dict(entity_name='TC',entity_type='Process')]
    assert canonical_name('Thermohaline Circulation',aliases,existing,'Process')=='TC'
    assert canonical_name('TC',aliases,existing,'Theory')=='Thermohaline Circulation'
    assert not explicit_aliases([dict(text='Thermohaline Circulation (TC). Tidal Current (TC).')])
    assert not explicit_aliases([dict(text='TC is nearby. Temperature (cold).')])
    assert canonical_name('Current',aliases,existing,'Process')=='Current'


def test_export_reuses_hash_once_and_next_request_revalidates(tmp_path,provider,monkeypatch):
    from qaproject import knowledge_read
    store=Store(tmp_path/'live');doc,_=store.import_file('a.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    first=knowledge.listing(store)['items'][0]
    for n in range(8):add_reviewed_fact(store,first['id'],f'合成事实{n}')
    calls=[];real=knowledge_read.file_hash
    def hashed(path):
        calls.append(path);return real(path)
    monkeypatch.setattr(knowledge_read,'file_hash',hashed)
    exported=knowledge.export_csv(store)
    assert len(calls)==1
    rows=list(csv.DictReader(io.StringIO(exported.decode())))
    assert len(rows)==1 and rows[0]['entity_description'].count('合成事实')==8
    Path(store.document(doc)['path']).write_text('changed')
    with pytest.raises(ValueError,match='原件'):knowledge.export_csv(store)
    assert len(calls)==2


def test_change_during_export_is_detected(tmp_path,provider,monkeypatch):
    store=Store(tmp_path/'live');doc,_=store.import_file('a.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    real=knowledge.detail
    def changed(*args,**kwargs):
        result=real(*args,**kwargs)
        Path(store.document(doc)['path']).write_text('modified during export')
        return result
    monkeypatch.setattr(knowledge,'detail',changed)
    with pytest.raises(ValueError,match='变更'):knowledge.export_csv(store)


def rejected_store(tmp_path,monkeypatch,provider):
    real=GLMClient.call
    def reject(self,instruction,payload,units=(),**kw):
        result,stats=real(self,instruction,payload,units,**kw)
        if 'entries' in payload:
            for check in result['checks']:check['language_correct']=False
        return result,stats
    monkeypatch.setattr(GLMClient,'call',reject)
    store=Store(tmp_path/'live');doc,_=store.import_file('a.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    monkeypatch.setattr(GLMClient,'call',real)
    return store,doc,rejections(store,doc)[0]


def test_repair_is_explicit_idempotent_and_keeps_rejection_history(tmp_path,provider,monkeypatch):
    store,doc,rejected=rejected_store(tmp_path,monkeypatch,provider)
    assert rejected['repairable'] and len(provider)==2
    with store.connect() as db:before=db.execute('SELECT result FROM knowledge_batches').fetchone()[0]
    task=enqueue_repair(store,doc,rejected['id'],rejected['expected_hash'])
    assert enqueue_repair(store,doc,rejected['id'],rejected['expected_hash'])==task
    run(store)
    assert len(provider)==4 and knowledge.listing(store)['total']==1
    assert rejections(store,doc)[0]['status']=='approved'
    assert enqueue_repair(store,doc,rejected['id'],rejected['expected_hash'])['status']=='approved'
    assert not Worker(store).run_once()
    with store.connect() as db:assert db.execute('SELECT result FROM knowledge_batches').fetchone()[0]==before


def test_repair_network_resume_does_not_regenerate_or_repeat_normal_batches(tmp_path,provider,monkeypatch):
    store,doc,rejected=rejected_store(tmp_path,monkeypatch,provider)
    real=GLMClient.call
    def interrupted(self,instruction,payload,units=(),**kw):
        if 'entries' in payload:raise ProviderError('isolated network failure','timeout',{})
        return real(self,instruction,payload,units,**kw)
    monkeypatch.setattr(GLMClient,'call',interrupted)
    original=enqueue_repair(store,doc,rejected['id'],rejected['expected_hash']);run(store)
    assert rejections(store,doc)[0]['status']=='retryable' and len(provider)==3
    monkeypatch.setattr(GLMClient,'call',real)
    resumed=enqueue_repair(store,doc,rejected['id'],rejected['expected_hash'])
    assert resumed['task_id']==original['task_id']
    run(store)
    assert len(provider)==4 and knowledge.listing(store)['total']==1


def test_repair_rejects_conflict_changed_source_and_unsupported_evidence(tmp_path,provider,monkeypatch):
    store,doc,rejected=rejected_store(tmp_path,monkeypatch,provider)
    with pytest.raises(ValueError,match='变更'):enqueue_repair(store,doc,rejected['id'],'bad')
    Path(store.document(doc)['path']).write_text('changed')
    with pytest.raises(ValueError,match='变更'):enqueue_repair(store,doc,rejected['id'],rejected['expected_hash'])
    assert len(provider)==2


def test_repair_api_session_guard_and_history(tmp_path,provider,monkeypatch):
    app=create_app(tmp_path/'app',workers=False);store=app.state.stores['live']
    doc,_=store.import_file('a.txt',TEXT.encode(),'live',task_stage='knowledge')
    real=GLMClient.call
    def rejected(self,instruction,payload,units=(),**kw):
        result,stats=real(self,instruction,payload,units,**kw)
        for check in result.get('checks',[]):check['type_correct']=False
        return result,stats
    monkeypatch.setattr(GLMClient,'call',rejected);run(store)
    client=TestClient(app);headers={'X-QA-Session':client.get('/api/bootstrap').json()['session']}
    data=client.get(f'/api/knowledge/jobs/{doc}/rejected').json()
    item=data['items'][0];url=f"/api/knowledge/jobs/{doc}/rejected/{item['id']}/retry"
    assert 'snapshot' not in item
    assert client.post(url,json={'expected_hash':item['expected_hash']}).status_code==403
    assert client.post(url,headers=headers,json={'expected_hash':'0'*64}).status_code==409
    assert client.post(url,headers=headers,json={'expected_hash':item['expected_hash']}).status_code==200
    assert len(provider)==2


def test_explicit_aliases_unify_new_batches_without_extra_calls(tmp_path,provider,monkeypatch):
    store=Store(tmp_path/'live');doc,_=store.import_file('alias.txt',b'isolated alias document','live',task_stage='knowledge')
    texts=['Thermohaline Circulation (TC) transports heat.']+['Other scientific evidence.']*5+['TC transports salt.']
    units=[]
    for n,text in enumerate(texts):
        u=dict(id=f'u{n}',text=text,image='',asset_id='',page=0,seg=n+1,locator=f'seg-{n+1}',language='en')
        u['hash']=digest(u);units.append(u)
    monkeypatch.setattr(knowledge,'prepare',lambda *args:(units,0))
    real=GLMClient.call
    def extracted(self,instruction,payload,group=(),**kw):
        if 'entries' in payload:return real(self,instruction,payload,group,**kw)
        provider.append(payload)
        u=group[0]
        return dict(entries=[dict(unit_id=u['id'],entity_name='TC',entity_type='Process',
            entity_description=u['text'],topic='海洋环流',quote_id='p1')]),{}
    monkeypatch.setattr(GLMClient,'call',extracted)
    run(store)
    assert len(provider)==4
    item=knowledge.listing(store)['items'][0]
    assert knowledge.listing(store)['total']==1 and item['fact_count']==2
    assert item['entity_name']=='Thermohaline Circulation'
    assert len(knowledge.detail(store,item['id'])['evidence'])==2


def test_unsupported_rejection_is_not_repaired_and_quality_retries_are_bounded(tmp_path,provider,monkeypatch):
    store,doc,rejected=rejected_store(tmp_path,monkeypatch,provider)
    real=GLMClient.call
    def still_wrong(self,instruction,payload,units=(),**kw):
        result,stats=real(self,instruction,payload,units,**kw)
        for check in result.get('checks',[]):check['type_correct']=False
        return result,stats
    monkeypatch.setattr(GLMClient,'call',still_wrong)
    for _ in range(2):
        enqueue_repair(store,doc,rejected['id'],rejected['expected_hash']);run(store)
    with pytest.raises(ValueError,match='两次'):enqueue_repair(store,doc,rejected['id'],rejected['expected_hash'])
    assert not knowledge.listing(store)['items']
    with store.connect() as db:
        audit=json.loads(db.execute('SELECT result FROM knowledge_batches').fetchone()[0])
        audit['rejected'][0]['review']['supported']=False
        db.execute('UPDATE knowledge_batches SET result=?',(json.dumps(audit),))
    unsupported=rejections(store,doc)[0]
    assert not unsupported['repairable']
    with pytest.raises(ValueError,match='不足'):enqueue_repair(store,doc,unsupported['id'],unsupported['expected_hash'])


def test_repair_uses_shared_document_lock_and_recovers_saved_generation(tmp_path,provider,monkeypatch):
    from qaproject.execution import WorkInterrupted
    store,doc,rejected=rejected_store(tmp_path,monkeypatch,provider)
    real=GLMClient.call
    def stop_at_review(self,instruction,payload,units=(),**kw):
        if 'entries' in payload:raise WorkInterrupted()
        return real(self,instruction,payload,units,**kw)
    monkeypatch.setattr(GLMClient,'call',stop_at_review)
    enqueue_repair(store,doc,rejected['id'],rejected['expected_hash'])
    worker=Worker(store);task=store.claim(worker.name)
    assert store.claim('other-worker',document_id=doc) is None
    worker.process(task)
    assert rejections(store,doc)[0]['status']=='queued'
    monkeypatch.setattr(GLMClient,'call',real)
    run(store)
    assert knowledge.listing(store)['total']==1 and len(provider)==4
