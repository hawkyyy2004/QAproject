"""Failed drafts must be repaired and independently reviewed before delivery."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from test_bilingual import setup_pipeline, run, pair
from qaproject.glm import GLMClient, ProviderError
from qaproject.pipeline import Worker, approve, quality_flags
from qaproject.store import digest, encode


def history(store, kind):
    with store.connect() as db:
        return [json.loads(r[0]) for r in db.execute('SELECT detail FROM events WHERE kind=? ORDER BY rowid',(kind,))]


def repair_answer(result, payload, units):
    if 'repair' in payload:
        return {'pairs':[pair(units[0]['id'],90+payload['repair']['attempt'])],'skipped':[]}
    return result


def fail_original(result, payload, units):
    for check,c in zip(result['checks'],payload['candidates']):
        if '91' not in c['versions'][0]['qa']['input']:
            check['translation_consistent']=False
            check['note']='英文数值与原证据不一致。'
    return result


def test_repair_rebinds_both_languages_and_every_delivered_row_can_commit(setup_pipeline):
    _,store,_,_,calls,behavior,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=1
    behavior.update(generation=repair_answer,review=fail_original)
    doc=run(setup_pipeline)
    assert doc['status']=='waiting_review' and len(doc['candidates'])==2,doc['error']
    assert [k for k,_,_ in calls]==['analysis','generation','review','generation','review']
    revision=history(store,'pair_revision')[0]
    old=revision['previous_versions']
    assert all(not c['review']['translation_consistent'] for c in old)
    assert {c['id'] for c in old}=={c['id'] for c in doc['candidates']}
    for c in doc['candidates']:
        previous=next(v for v in old if v['id']==c['id'])
        assert c['row_hash']!=previous['row_hash']
        assert c['row']['source']==previous['row']['source'] and c['evidence_hash']==previous['evidence_hash']
        assert c['review']['binding']['candidate_hashes']=={v['id']:v['row_hash'] for v in doc['candidates']}
        with pytest.raises(ValueError,match='版本'):approve(store,c['id'],True,previous['row_hash'])
        approve(store,c['id'],True,c['row_hash'])
    assert store.document(doc['id'])['qa_pair_count']==1
    assert 'repairing' in doc['timings'] and 'repair_reviewing' in doc['timings'] and 'preflight' in doc['timings']


def test_all_healthy_generation_precedes_any_repairs(setup_pipeline):
    _,_,_,_,calls,behavior,cfg=setup_pipeline
    behavior['generation']=repair_answer
    def first_fails(result,payload,units):
        for check,c in zip(result['checks'],payload['candidates']):
            if '合成观测 0 中' in c['versions'][0]['qa']['input']:
                check['independent']=False
        return result
    behavior['review']=first_fails
    doc=run(setup_pipeline)
    assert doc['bilingual_count']==5 and all(c['status']=='pending' for c in doc['candidates']),doc['error']
    kinds=[(k,'repair' in p) for k,p,_ in calls]
    first_repair=kinds.index(('generation',True))
    assert kinds[:first_repair]==[('analysis',False),('generation',False),('review',False),('generation',False),('review',False)]


def test_repair_stays_on_one_unit_and_rejects_broader_evidence(setup_pipeline):
    _,store,_,_,calls,behavior,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=1
    behavior['review']=fail_original
    def wrong(result,payload,units):
        if 'repair' in payload:
            q=pair('a-different-unit',91)
            return {'pairs':[q],'skipped':[]}
        return result
    behavior['generation']=wrong
    doc=run(setup_pipeline)
    assert doc['status']=='insufficient' and not doc['candidates'] and doc['bilingual_count']==0
    repairs=[(p,u) for k,p,u in calls if k=='generation' and 'repair' in p]
    assert repairs and all(len(u)==1 for _,u in repairs)
    assert all('不能切换' in r['error'] for r in history(store,'pair_revision'))
    with store.connect() as db:
        assert all(r[0]<=2 for r in db.execute('SELECT repair_attempts FROM qa_pairs'))
    assert not store.documents(summary_only=True)[0]['candidate_count']


def test_crash_after_revision_resumes_review_without_regenerating(setup_pipeline):
    _,store,doc_id,worker,calls,behavior,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=1
    behavior['generation']=repair_answer
    crash=[True]
    def review(result,payload,units):
        if '91' in payload['candidates'][0]['versions'][0]['qa']['input'] and crash[0]:
            raise ProviderError('隔离测试超时','timeout',{})
        return fail_original(result,payload,units)
    behavior['review']=review
    failed=run(setup_pipeline)
    assert failed['status']=='failed' and len(failed['candidates'])==2
    assert all(not c['review'] for c in failed['candidates'])
    ids={c['id']:c['row_hash'] for c in failed['candidates']}
    with store.connect() as db:
        db.execute("UPDATE tasks SET status='queued' WHERE document_id=? AND status='failed'",(doc_id,))
    crash[0]=False
    calls.clear()
    worker.process(store.claim(worker.name,document_id=doc_id))
    doc=store.document(doc_id)
    assert doc['status']=='waiting_review' and all(c['status']=='pending' for c in doc['candidates']),doc['error']
    assert [k for k,_,_ in calls]==['review']
    assert ids=={c['id']:c['row_hash'] for c in doc['candidates']}
    assert len(history(store,'pair_revision'))==1


def test_optimize_existing_uses_same_queue_and_preserves_approved_rows(setup_pipeline):
    app,store,doc_id,worker,calls,behavior,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=2
    doc=run(setup_pipeline)
    for c in doc['candidates'][:2]:approve(store,c['id'],True,c['row_hash'])
    with store.connect() as db:
        for c in doc['candidates'][2:]:
            db.execute("UPDATE candidates SET status='needs_review',review=? WHERE id=?",
                       (encode(dict(c['review'],supported=False,note='旧任务存在未通过项。')),c['id']))
    client=TestClient(app)
    headers={'X-QA-Session':client.get('/api/bootstrap').json()['session']}
    endpoint='/api/documents/'+doc_id+'/optimize'
    assert client.post(endpoint).status_code==403
    assert client.post(endpoint,headers=headers).status_code==200
    assert client.post(endpoint,headers=headers).status_code==400
    behavior['generation']=repair_answer
    calls.clear()
    worker.process(store.claim(worker.name,document_id=doc_id))
    latest=store.document(doc_id)
    assert latest['qa_count']==2 and latest['bilingual_count']==2
    assert [(c['row'],c['review']) for c in latest['candidates'][:2]]==[(c['row'],c['review']) for c in doc['candidates'][:2]]
    assert [k for k,_,_ in calls]==['generation','review']
    assert history(store,'analysis_resumed')


def test_legacy_keyword_veto_reuses_valid_review_without_regeneration(setup_pipeline):
    app,store,doc_id,worker,calls,_,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=1
    doc=run(setup_pipeline)
    with store.connect() as db:
        for c in doc['candidates']:
            old=dict(c['review'],local_flags=['问题可能依赖未定义指代，需要写明对象和范围。'])
            db.execute("UPDATE candidates SET status='needs_review',review=? WHERE id=?",(encode(old),c['id']))
    client=TestClient(app)
    headers={'X-QA-Session':client.get('/api/bootstrap').json()['session']}
    assert client.post('/api/documents/'+doc_id+'/optimize',headers=headers).status_code==200
    calls.clear()
    worker.process(store.claim(worker.name,document_id=doc_id))
    latest=store.document(doc_id)
    assert all(c['status']=='pending' and not c['review'].get('local_flags') for c in latest['candidates'])
    assert not calls and any(e['stage']=='review' for e in history(store,'cache_hit'))
    assert [(c['id'],c['row_hash']) for c in latest['candidates']]==[(c['id'],c['row_hash']) for c in doc['candidates']]


def test_named_subject_is_not_rejected_by_keyword_alone():
    assert not quality_flags({'input':'该研究选用广义回归神经网络（GRNN）反演海浪频谱，其网络结构包括哪几部分？'})
    assert not quality_flags({'input':'In this study, a GRNN retrieves ocean wave spectra. What are the components of its network structure?'})
    assert quality_flags({'input':'What are these results?'})
    assert quality_flags({'input':'这些是什么？'})


def test_preflight_still_blocks_changed_original(setup_pipeline):
    _,store,doc_id,_,_,behavior,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=1
    def change_source(result,*args):
        from pathlib import Path
        Path(store.document(doc_id)['path']).write_text('different evidence')
        return result
    behavior['review']=change_source
    doc=run(setup_pipeline)
    assert doc['status']=='failed' and '原始文档副本已变更' in doc['error'] and doc['qa_count']==0


def test_other_file_generation_takes_priority_over_repairs(setup_pipeline):
    _,store,doc_id,worker,calls,behavior,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=1
    other,_=store.import_file('second.txt','另一个隔离文件，温度记录15摄氏度。'.encode(),'live')
    behavior.update(generation=repair_answer,review=fail_original)
    doc=run(setup_pipeline)
    assert doc['status']=='queued' and history(store,'quality_deferred')
    assert [k for k,_,_ in calls]==['analysis','generation','review']
    next_task=store.claim(worker.name)
    assert next_task['document_id']==other
    behavior.pop('review')
    worker.process(next_task)
    deferred=store.claim(worker.name)
    assert deferred['document_id']==doc_id and deferred['stage']=='quality_queued'
    calls.clear()
    behavior['review']=fail_original
    worker.process(deferred)
    assert [k for k,_,_ in calls]==['generation','review']
    assert store.document(doc_id)['status']=='waiting_review'


def test_stale_worker_cannot_publish_a_revision(setup_pipeline):
    _,store,doc_id,_,_,behavior,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=1
    behavior['review']=fail_original
    def expire(result,payload,units):
        if 'repair' in payload:
            with store.connect() as db:
                db.execute("UPDATE tasks SET worker='replacement-worker' WHERE document_id=? AND status='running'",(doc_id,))
                db.execute("UPDATE documents SET status='generating',progress='replacement worker owns task' WHERE id=?",(doc_id,))
            return repair_answer(result,payload,units)
        return result
    behavior['generation']=expire
    doc=run(setup_pipeline)
    assert doc['progress']=='replacement worker owns task'
    assert not history(store,'pair_revision') and history(store,'stale_worker_result')
    assert all(c['status']=='needs_review' for c in doc['candidates'])


def test_visual_repair_sends_original_image_bytes_again(tmp_path,monkeypatch):
    import pymupdf
    from qaproject.app import create_app
    from qaproject.settings import settings
    from test_prototype import fake_provider
    app=create_app(tmp_path/'visual',workers=False)
    store=app.state.stores['live']
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,40,40),False)
    pix.clear_with(255)
    doc_id,_=store.import_file('synthetic.png',pix.tobytes('png'),'live')
    seen=[]
    def provider(request):
        body=json.loads(request.content)
        content=body['messages'][1]['content']
        payload=json.loads(content[0]['text'])
        images=[p['image_url']['url'] for p in content if p['type']=='image_url']
        seen.append(images)
        body['messages'][1]['content']=encode(payload)
        response=fake_provider(httpx.Request('POST','https://fixture.invalid',content=json.dumps(body)))
        result=json.loads(response.json()['choices'][0]['message']['content'])
        if 'repair' in payload:
            result['pairs'][0]['zh']['input']='在合成海域A的2025年1月观测中，测得温度为多少？'
        if 'candidates' in payload and len(seen)==3:
            result['checks'][0]['translation_consistent']=False
        data=response.json()
        data['choices'][0]['message']['content']=encode(result)
        return httpx.Response(200,json=data)
    real_client=httpx.Client
    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:real_client(transport=httpx.MockTransport(provider)))
    monkeypatch.setattr('qaproject.glm.api_key',lambda:'test-placeholder')
    monkeypatch.setattr('qaproject.pipeline.settings',lambda:dict(settings(),target_per_document=1,min_per_document=1))
    worker=Worker(store)
    worker.process(store.claim(worker.name))
    doc=store.document(doc_id)
    assert doc['status']=='waiting_review' and len(seen)==5,doc['error']
    assert all(images==seen[0] and images[0].startswith('data:image/png;base64,') for images in seen)
    assert len(history(store,'pair_revision'))==1
    for c in doc['candidates']:approve(store,c['id'],True,c['row_hash'])
    assert store.document(doc_id)['qa_count']==2
