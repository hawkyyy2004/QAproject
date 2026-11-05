import csv
import io
import json
import threading

from fastapi.testclient import TestClient
import pytest

from qaproject.app import create_app
from qaproject.execution import ExecutionLimits
from qaproject.knowledge import CHECKS, FIELDS, detail, enqueue, export_csv, listing
from qaproject.pipeline import Worker
from qaproject.store import Store, encode


TEXT='温盐环流是由海水温度和盐度差异引起密度差异而驱动的大尺度海洋环流。'


@pytest.fixture
def provider(monkeypatch):
    calls=[]
    def call(self,instruction,payload,units=(),**kwargs):
        calls.append(payload)
        if 'entries' in payload:
            return {'checks':[dict(id=e['id'],**{k:True for k in CHECKS},note='原文支持定义、分类和适用范围。') for e in payload['entries']]}, {'finish_reason':'stop'}
        u=units[0]
        return {'entries':[dict(unit_id=u['id'],entity_name='温盐环流',entity_type='Process',
                               entity_description=TEXT,topic='海洋环流',quote=u['text'])]}, {'finish_reason':'stop'}
    monkeypatch.setattr('qaproject.knowledge.GLMClient.call',call)
    return calls


def run(store):
    worker=Worker(store)
    assert worker.run_once()


def test_independent_extraction_review_export_and_idempotence(tmp_path,provider):
    store=Store(tmp_path/'live')
    doc_id,_=store.import_file('ocean.txt',TEXT.encode(),'live',task_stage='knowledge')
    run(store)
    assert len(provider)==2
    data=listing(store)
    assert data['total']==1 and data['items'][0]['entity_type']=='Process'
    entry=detail(store,data['items'][0]['id'])
    assert entry['locator']=='seg-01' and entry['source_valid']
    assert entry['row']['page_urls']=='' and entry['row']['full_text']==TEXT
    assert entry['review']['evidence_hash']==entry['evidence_hash']
    assert entry['row']['full_text_length']==len(TEXT)
    rows=list(csv.DictReader(io.StringIO(export_csv(store).decode('utf-8-sig'))))
    assert list(rows[0])==FIELDS and rows[0]['entity_description']==TEXT
    assert store.document(doc_id)['qa_count']==0
    assert store.document(doc_id)['status']=='knowledge_completed'
    assert enqueue(store,doc_id)['status']=='completed'
    store.update_document(doc_id,status='knowledge_failed')
    assert enqueue(store,doc_id)['status']=='completed'
    assert store.document(doc_id)['status']=='knowledge_completed'
    assert not Worker(store).run_once()


def test_existing_qa_file_is_not_modified(tmp_path,provider):
    store=Store(tmp_path/'live');doc_id,_=store.import_file('old.txt',TEXT.encode(),'live')
    with store.connect() as db:
        db.execute("UPDATE tasks SET status='succeeded' WHERE document_id=?",(doc_id,))
    store.update_document(doc_id,status='waiting_review',summary='原有QA梗概',seconds=42)
    before=store.document(doc_id)
    enqueue(store,doc_id);run(store)
    after=store.document(doc_id)
    assert before==after
    assert listing(store)['total']==1


def test_missing_review_cannot_be_exported_and_retry_reuses_extraction(tmp_path,provider,monkeypatch):
    from qaproject.knowledge import GLMClient
    good=GLMClient.call
    def incomplete(self,instruction,payload,units=(),**kw):
        if 'entries' in payload:return {'checks':[]},{'finish_reason':'stop'}
        return good(self,instruction,payload,units,**kw)
    monkeypatch.setattr(GLMClient,'call',incomplete)
    store=Store(tmp_path/'live');doc_id,_=store.import_file('ocean.txt',TEXT.encode(),'live',task_stage='knowledge')
    run(store)
    assert listing(store,status='')['total']==0
    with store.connect() as db:assert db.execute('SELECT status FROM knowledge_jobs').fetchone()[0]=='failed'
    assert store.document(doc_id)['status']=='knowledge_failed'
    monkeypatch.setattr(GLMClient,'call',good)
    enqueue(store,doc_id);run(store)
    assert listing(store)['total']==1
    assert store.document(doc_id)['status']=='knowledge_completed'
    assert sum('units' in c and 'entries' not in c for c in provider)==1


def test_rejected_entry_retained_but_not_exported(tmp_path,provider,monkeypatch):
    from qaproject.knowledge import GLMClient
    good=GLMClient.call
    def reject(self,instruction,payload,units=(),**kw):
        result,stats=good(self,instruction,payload,units,**kw)
        for check in result.get('checks',[]):check['supported']=False
        return result,stats
    monkeypatch.setattr(GLMClient,'call',reject)
    store=Store(tmp_path/'live');store.import_file('ocean.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    assert listing(store,status='')['total']==0
    with store.connect() as db:
        audit=json.loads(db.execute('SELECT result FROM knowledge_batches').fetchone()[0])
    assert len(audit['rejected'])==1 and audit['rejected'][0]['review']['supported'] is False
    assert len(list(csv.DictReader(io.StringIO(export_csv(store).decode('utf-8-sig')))))==0


def test_fake_quote_fails_closed(tmp_path,provider,monkeypatch):
    from qaproject.knowledge import GLMClient
    good=GLMClient.call
    def bad(self,instruction,payload,units=(),**kw):
        result,stats=good(self,instruction,payload,units,**kw)
        for entry in result.get('entries',[]):entry['quote']='原文没有这句话'
        return result,stats
    monkeypatch.setattr(GLMClient,'call',bad)
    store=Store(tmp_path/'live');store.import_file('ocean.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    assert listing(store,status='')['total']==0


def test_source_and_review_tampering_blocks_export(tmp_path,provider):
    store=Store(tmp_path/'live');doc_id,_=store.import_file('ocean.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    entry=listing(store)['items'][0]
    with store.connect() as db:
        db.execute("UPDATE knowledge_entries SET evidence_hash='tampered'")
    with pytest.raises(ValueError,match='绑定'):export_csv(store)


def test_changed_original_blocked(tmp_path,provider):
    from pathlib import Path
    store=Store(tmp_path/'live');doc_id,_=store.import_file('ocean.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    Path(store.document(doc_id)['path']).write_text('原件修改',encoding='utf-8')
    assert not detail(store,listing(store)['items'][0]['id'])['source_valid']
    with pytest.raises(ValueError,match='原件'):export_csv(store)


def test_csv_knowledge_rows_preserve_sources_without_numeric_sampling(tmp_path,provider):
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=FIELDS);w.writeheader()
    for i in range(9):w.writerow(dict(entity_name='温盐环流',entity_type='Process',entity_description=TEXT,
        page_urls='https://example.org/ocean',page_titles='Ocean',full_text=TEXT,full_text_length=len(TEXT)))
    store=Store(tmp_path/'live');doc_id,_=store.import_file('entries.csv',out.getvalue().encode(),'live',task_stage='knowledge');run(store)
    with store.connect() as db:units=json.loads(db.execute('SELECT units FROM knowledge_jobs').fetchone()[0])
    assert len(units)==9 and units[-1]['locator']=='record-0009-seg-01'
    assert all(u['text']==TEXT for u in units)
    assert listing(store)['items'][0]['document_id']==doc_id
    entry=detail(store,listing(store)['items'][0]['id'])
    assert entry['row']['page_urls']=='https://example.org/ocean'


def test_api_default_qa_preserved_and_knowledge_is_separate(tmp_path,provider):
    app=create_app(tmp_path/'app',workers=False);client=TestClient(app)
    headers={'X-QA-Session':client.get('/api/bootstrap').json()['session']}
    assert client.post('/api/imports?processing=knowledge',files={'files':('a.txt',TEXT.encode())}).status_code==403
    client.post('/api/imports',headers=headers,files={'files':('qa.txt',b'Existing QA file')})
    result=client.post('/api/imports?processing=knowledge',headers=headers,files={'files':('ocean.txt',TEXT.encode())})
    assert result.status_code==200
    store=app.state.stores['live']
    with store.connect() as db:
        qa=db.execute("SELECT * FROM tasks WHERE initial_stage=''").fetchone()
        assert qa is not None
        db.execute("UPDATE tasks SET status='succeeded' WHERE id=?",(qa['id'],))
    run(store)
    assert client.get('/api/qas').json()['total']==0
    assert client.get('/api/knowledge?entity_type=Process&q=温盐').json()['total']==1
    assert client.get('/api/knowledge?topic=其他').json()['total']==0
    assert client.get('/api/knowledge/jobs').json()[0]['status']=='completed'
    assert client.get('/api/knowledge/export').content.startswith(b'\xef\xbb\xbf')
    assert client.get('/api/knowledge?page=0').status_code==422
    assert client.get('/api/knowledge?entity_type=invalid').status_code==400
    assert client.get('/knowledge.js').status_code==200


def test_resume_completed_batches_and_shared_file_exclusivity(tmp_path,provider):
    store=Store(tmp_path/'live');doc_id,_=store.import_file('ocean.txt',TEXT.encode(),'live',task_stage='knowledge')
    limits=ExecutionLimits();first=Worker(store,limits=limits);second=Worker(store,limits=limits)
    task=limits.claim(store,first.name)
    enqueue(store,doc_id)
    assert limits.claim(store,second.name) is None
    store.recover_abandoned()
    limits.release(store,task)
    assert first.run_once()
    with store.connect() as db:db.execute("UPDATE knowledge_jobs SET status='failed'")
    before=len(provider);enqueue(store,doc_id);run(store)
    assert len(provider)==before and listing(store)['total']==1
    limits.close()


def test_knowledge_first_can_still_start_original_qa_once(tmp_path,provider):
    store=Store(tmp_path/'live');doc_id,_=store.import_file('ocean.txt',TEXT.encode(),'live',task_stage='knowledge')
    run(store)
    first=store.ensure_qa_task(doc_id);second=store.ensure_qa_task(doc_id)
    assert first==second and first['status']=='queued'
    assert store.document(doc_id)['status']=='queued'
    assert listing(store)['total']==1
    task=store.claim('qa-worker')
    assert task['stage']=='queued' and task['id']==first['task_id']
    assert not task['review_candidate_id'] and not task['revision_id']


def test_public_summary_cannot_expose_legacy_rejected_rows(tmp_path,provider):
    app=create_app(tmp_path/'app',workers=False);store=app.state.stores['live']
    store.import_file('ocean.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    with store.connect() as db:
        db.execute("""INSERT INTO knowledge_entries SELECT 'legacy-rejected',document_id,'legacy-unit',
            evidence_hash,row_json,row_hash,topic,quote,'rejected',review,created FROM knowledge_entries LIMIT 1""")
    client=TestClient(app)
    for query in ['', '?status=', '?status=rejected']:
        data=client.get('/api/knowledge'+query).json()
        assert data['total']==1 and all(r['status']=='approved' for r in data['items'])
    assert client.get('/api/knowledge/legacy-rejected').status_code==400
    assert len(list(csv.DictReader(io.StringIO(client.get('/api/knowledge/export').content.decode('utf-8-sig')))))==1


def test_seven_field_summary_bounds_original_text_but_detail_and_export_are_complete(tmp_path,provider):
    store=Store(tmp_path/'live');text=TEXT*30
    store.import_file('long.txt',text.encode(),'live',task_stage='knowledge');run(store)
    item=listing(store)['items'][0]
    assert item['full_text_preview']==text[:500] and item['full_text_length']==len(text)
    assert item['page_urls']=='' and item['page_titles']=='long.txt'
    assert detail(store,item['id'])['row']['full_text']==text
    rows=list(csv.DictReader(io.StringIO(export_csv(store).decode('utf-8-sig'))))
    assert list(rows[0])==FIELDS and rows[0]['full_text']==text


def test_processing_record_tabs_have_independent_membership_and_paged_summaries(tmp_path):
    app=create_app(tmp_path/'app',workers=False);store=app.state.stores['live']
    qa_id,_=store.import_file('qa.txt',b'qa-only','live')
    ids=[]
    for i in range(24):
        doc_id,_=store.import_file(f'knowledge-{i:02d}.txt',f'knowledge {i}'.encode(),'live',task_stage='knowledge')
        ids.append(doc_id)
    store.ensure_qa_task(ids[0])
    client=TestClient(app)
    docs={d['id']:d for d in client.get('/api/documents?summary_only=true').json()}
    assert docs[qa_id]['has_qa_tasks']==1 and docs[ids[0]]['has_qa_tasks']==1
    assert docs[ids[1]]['has_qa_tasks']==0
    first=client.get('/api/knowledge/jobs?page=1&page_size=10').json()
    second=client.get('/api/knowledge/jobs?page=2&page_size=10').json()
    assert first['total']==24 and first['pages']==3 and len(first['items'])==10
    assert not {j['document_id'] for j in first['items']}.intersection(j['document_id'] for j in second['items'])
    match=client.get('/api/knowledge/jobs?page=1&q=knowledge-23').json()
    assert match['total']==1 and match['items'][0]['document_id']==ids[23]
    record=client.get('/api/knowledge/jobs/'+ids[23]).json()
    assert record['status']=='queued' and 'units' not in record
    assert client.get('/api/knowledge/jobs/'+qa_id).status_code==404
    assert client.get('/api/knowledge/jobs?page=0').status_code==422


def test_quote_ids_preserve_pdf_whitespace_and_reject_fabricated_text():
    from qaproject.knowledge import evidence_quote,passages
    unit={'text':'海水温度和盐度\n\n差异驱动环流。'*100}
    source=passages(unit)
    assert ''.join(p['text'] for p in source)==unit['text']
    assert evidence_quote({'quote_id':source[0]['id']},unit)==source[0]['text']
    assert evidence_quote({'quote':'海水温度和盐度差异驱动环流。'},unit)=='海水温度和盐度\n\n差异驱动环流。'
    with pytest.raises(ValueError):evidence_quote({'quote':'海水温度与盐度差异驱动环流。'},unit)
    with pytest.raises(ValueError):evidence_quote({'quote_id':'invented'},unit)


def test_relocated_uploaded_original_uses_verified_id_without_rewriting_history(tmp_path,provider):
    from pathlib import Path
    from qaproject.source_files import original_path
    store=Store(tmp_path/'live');doc_id,_=store.import_file('moved.txt',TEXT.encode(),'live',task_stage='knowledge')
    actual=Path(store.document(doc_id)['path'])
    historical=str(tmp_path/'old'/'originals'/doc_id/'moved.txt')
    with store.connect() as db:db.execute('UPDATE documents SET path=? WHERE id=?',(historical,doc_id))
    assert original_path(store.document(doc_id),store.root)==actual
    run(store)
    assert store.document(doc_id)['path']==historical
    entry=detail(store,listing(store)['items'][0]['id'])
    assert entry['source_path']==str(actual) and entry['registered_source_path']==historical
    assert entry['source_valid'] and entry['source_relocated']
    assert export_csv(store)
    actual.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='不一致'):original_path(store.document(doc_id),store.root)


def test_relocation_cannot_guess_other_ids_or_replace_external_library(tmp_path):
    from qaproject.source_files import original_path
    store=Store(tmp_path/'live');doc_id,_=store.import_file('original.txt',TEXT.encode(),'live')
    doc=store.document(doc_id)
    doc['path']=str(tmp_path/'old'/'originals'/'wrong-id'/'original.txt')
    with pytest.raises(ValueError):original_path(doc,store.root)
    doc.update(source_kind='library',source_root=str(tmp_path/'external'))
    with pytest.raises(ValueError):original_path(doc,store.root)


def test_one_bad_quote_does_not_abort_valid_entries(tmp_path,provider,monkeypatch):
    from qaproject.knowledge import GLMClient
    original=GLMClient.call
    def mixed(self,instruction,payload,units=(),**kwargs):
        result,stats=original(self,instruction,payload,units,**kwargs)
        if 'entries' not in payload:
            valid=result['entries'][0];valid.pop('quote');valid['quote_id']='p1'
            result['entries'].append(dict(valid,entity_name='无依据候选',quote_id='invented'))
        return result,stats
    monkeypatch.setattr(GLMClient,'call',mixed)
    store=Store(tmp_path/'live');doc_id,_=store.import_file('pdf-text.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    assert listing(store)['total']==1 and store.document(doc_id)['status']=='knowledge_completed'
    entry=detail(store,listing(store)['items'][0]['id'])
    assert entry['quote']==TEXT and entry['review']['supported'] is True
    with store.connect() as db:
        audit=json.loads(db.execute('SELECT result FROM knowledge_batches').fetchone()[0])
        assert len(audit['rejected'])==1 and audit['rejected'][0]['review']['local_validation']
    assert len(provider)==2


def test_empty_image_only_evidence_is_explicit_failure(tmp_path,provider,monkeypatch):
    from qaproject.store import digest
    unit=dict(id='image',text='Image evidence.',page=0,seg=1,image='not-read.png',asset_id='image',locator='image-0001',language='en')
    unit['hash']=digest(unit)
    monkeypatch.setattr('qaproject.knowledge.prepare',lambda *args:([unit],0))
    store=Store(tmp_path/'live');store.import_file('image.png',b'isolated image fixture','live',task_stage='knowledge');run(store)
    assert not provider and listing(store)['total']==0
    with store.connect() as db:
        job=db.execute('SELECT status,error FROM knowledge_jobs').fetchone()
        assert job['status']=='failed' and '仅含图片' in job['error']


def test_combined_upload_enqueues_both_once_and_qa_failure_does_not_block_knowledge(tmp_path,provider):
    app=create_app(tmp_path/'app',workers=False);client=TestClient(app)
    headers={'X-QA-Session':client.get('/api/bootstrap').json()['session']}
    def upload():return client.post('/api/imports?processing=both',headers=headers,files={'files':('ocean.txt',TEXT.encode())})
    result=upload();assert result.status_code==200
    doc_id=result.json()[0]['id'];assert not upload().json()[0]['created']
    store=app.state.stores['live']
    with store.connect() as db:
        rows=db.execute('SELECT * FROM tasks WHERE document_id=? ORDER BY rowid',(doc_id,)).fetchall()
        assert len(rows)==2 and rows[0]['initial_stage']=='' and rows[1]['initial_stage']=='knowledge'
    qa=store.claim('qa');assert qa['stage']=='queued'
    assert store.claim('knowledge') is None
    store.task_update(qa['id'],'qa','failed','failed')
    store.update_document(doc_id,status='failed',error='原QA失败原因')
    run(store)
    assert listing(store)['total']==1
    assert store.document(doc_id)['error']=='原QA失败原因'
    assert store.document(doc_id)['status']=='failed'
    assert client.get('/api/qas').json()['total']==0


@pytest.mark.parametrize('processing,expected',[('qa',['']),('knowledge',['knowledge']),('both',['','knowledge'])])
def test_library_processing_choice_is_consistent(tmp_path,processing,expected):
    root=tmp_path/'library';root.mkdir();(root/'ocean.txt').write_text(TEXT,encoding='utf-8')
    app=create_app(tmp_path/'app',workers=False,library_root=root)
    app.state.library.scan()
    client=TestClient(app);headers={'X-QA-Session':client.get('/api/bootstrap').json()['session']}
    item=client.get('/api/library').json()['items'][0]
    response=client.post('/api/library/imports?processing='+processing,headers=headers,
        json={'files':[{'id':item['id'],'version':item['version']}]})
    assert response.status_code==200 and not response.json()[0].get('error')
    with app.state.stores['live'].connect() as db:
        assert [r[0] for r in db.execute('SELECT initial_stage FROM tasks ORDER BY rowid')]==expected


def test_entity_context_and_exact_repeats_across_batches(tmp_path,provider):
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=FIELDS);writer.writeheader()
    for _ in range(7):
        writer.writerow(dict(entity_name='温盐环流',entity_description=TEXT,full_text=TEXT))
    store=Store(tmp_path/'live')
    store.import_file('repeated.csv',out.getvalue().encode(),'live',task_stage='knowledge')
    run(store)
    extracts=[p for p in provider if 'entries' not in p]
    assert len(extracts)==2
    assert extracts[0]['existing_entities']==[]
    assert extracts[1]['existing_entities']==[dict(entity_name='温盐环流',entity_type='Process',entity_description=TEXT)]
    assert listing(store)['total']==1
    assert len([p for p in provider if 'entries' in p])==2
    assert detail(store,listing(store)['items'][0]['id'])['fact_count']==2


def test_complementary_entity_fact_keeps_own_evidence(tmp_path,provider,monkeypatch):
    from qaproject.knowledge import GLMClient
    good=GLMClient.call
    addition='温盐环流的变化会影响海洋热量输送。'
    def call(self,instruction,payload,units=(),**kw):
        result,stats=good(self,instruction,payload,units,**kw)
        if 'entries' not in payload and payload.get('existing_entities'):
            result['entries'][0]['entity_description']=addition
        return result,stats
    monkeypatch.setattr(GLMClient,'call',call)
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=FIELDS);writer.writeheader()
    for i in range(7):
        writer.writerow(dict(entity_name='温盐环流',entity_description=TEXT,full_text=addition if i==6 else TEXT))
    store=Store(tmp_path/'live');store.import_file('complement.csv',out.getvalue().encode(),'live',task_stage='knowledge');run(store)
    items=listing(store)['items']
    assert len(items)==1 and items[0]['fact_count']==2
    entity=detail(store,items[0]['id'])
    assert entity['row']['entity_description']==TEXT+'\n\n'+addition
    assert entity['row']['full_text']==TEXT+'\n\n'+addition
    extra=next(e for e in entity['evidence'] if e['description']==addition)
    assert extra['quote']==addition and extra['review']['row_hash']==extra['row_hash']
    assert entity['review']['row_hash']==entity['row_hash']


def add_reviewed_fact(store,entry_id,description,*,name=None,entity_type=None):
    """Independent reviewed fact fixture; never used against application data."""
    from qaproject.store import digest
    with store.connect() as db:
        original=dict(db.execute('SELECT * FROM knowledge_entries WHERE id=?',(entry_id,)).fetchone())
        row=json.loads(original['row_json']);row['entity_description']=description
        if name is not None:row['entity_name']=name
        if entity_type is not None:row['entity_type']=entity_type
        entry_id=digest([original['id'],row])
        review=json.loads(original['review']);review.update(id=entry_id,row_hash=digest(row))
        db.execute('INSERT INTO knowledge_entries VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (entry_id,original['document_id'],original['unit_id'],original['evidence_hash'],encode(row),
             digest(row),original['topic'],original['quote'],'approved',encode(review),original['created']+1))
    return entry_id


def test_entity_accumulates_without_rewriting_facts_and_versions(tmp_path,provider):
    from qaproject.knowledge import jobs
    store=Store(tmp_path/'live');store.import_file('a.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    item=listing(store)['items'][0];before=detail(store,item['id'])
    extra=add_reviewed_fact(store,item['id'],'海水盐度差异参与驱动温盐环流。')
    # Repeated explanation in another fact is shown once, with both audits kept.
    add_reviewed_fact(store,item['id'],TEXT,name=' 温盐环流 ')
    after=detail(store,item['id'])
    assert listing(store)['total']==1 and after['id']==before['id']
    assert after['entity_version']!=before['entity_version'] and after['fact_count']==3
    assert after['row']['entity_description'].count(TEXT)==1
    assert after['row']['full_text']==TEXT  # shared unit is not duplicated
    assert after['row']['full_text_length']==len(TEXT)
    assert after['evidence'][0]['row_hash']==before['row_hash']
    assert detail(store,extra)['id']==before['id']
    matching=listing(store,q='参与驱动')
    assert matching['total']==1 and matching['items'][0]['fact_count']==3
    assert matching['items'][0]['entity_description']==after['row']['entity_description']
    assert jobs(store)['items'][0]['approved_count']==1
    assert jobs(store)['items'][0]['fact_count']==3
    rows=list(csv.DictReader(io.StringIO(export_csv(store,q='参与驱动').decode('utf-8-sig'))))
    assert len(rows)==1 and rows[0]=={k:str(after['row'][k]) for k in FIELDS}


def test_entity_does_not_cross_file_type_or_similar_name(tmp_path,provider):
    store=Store(tmp_path/'live');store.import_file('a.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    first=listing(store)['items'][0]
    add_reviewed_fact(store,first['id'],TEXT,entity_type='Theory')
    add_reviewed_fact(store,first['id'],TEXT,name='温盐环流模型')
    store.import_file('b.txt',(TEXT+'另一个文件。').encode(),'live',task_stage='knowledge');run(store)
    assert listing(store)['total']==4
    assert listing(store,document_id=first['document_id'])['total']==3
    assert listing(store,entity_type='Theory')['total']==1
    assert detail(store,first['id'])['fact_count']==1


def test_corrupted_later_fact_blocks_whole_entity_export(tmp_path,provider):
    store=Store(tmp_path/'live');store.import_file('a.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    first=listing(store)['items'][0]
    extra=add_reviewed_fact(store,first['id'],'海水盐度差异参与驱动温盐环流。')
    with store.connect() as db:db.execute("UPDATE knowledge_entries SET review='{}' WHERE id=?",(extra,))
    with pytest.raises(ValueError,match='绑定'):detail(store,first['id'])
    with pytest.raises(ValueError,match='绑定'):export_csv(store)


def test_export_freezes_member_ids_not_only_entity_ids(tmp_path,provider,monkeypatch):
    from qaproject import knowledge
    store=Store(tmp_path/'live');store.import_file('a.txt',TEXT.encode(),'live',task_stage='knowledge');run(store)
    first=listing(store)['items'][0];real=knowledge.detail
    def during_export(store,entry_id,member_ids=None):
        assert member_ids==[first['id']]
        add_reviewed_fact(store,entry_id,'海水盐度差异参与驱动温盐环流。')
        return real(store,entry_id,member_ids)
    monkeypatch.setattr(knowledge,'detail',during_export)
    rows=list(csv.DictReader(io.StringIO(export_csv(store).decode('utf-8-sig'))))
    assert rows[0]['entity_description']==TEXT
    assert real(store,first['id'])['fact_count']==2


def test_complementary_facts_in_one_batch_are_not_silently_dropped(tmp_path,provider,monkeypatch):
    from qaproject.knowledge import GLMClient
    good=GLMClient.call
    def call(self,instruction,payload,units=(),**kw):
        result,stats=good(self,instruction,payload,units,**kw)
        if 'entries' not in payload:
            base=result['entries'][0]
            result['entries'].append(dict(base,unit_id=units[1]['id'],quote=units[1]['text'],
                entity_description='海水盐度差异参与驱动温盐环流。'))
        return result,stats
    monkeypatch.setattr(GLMClient,'call',call)
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=FIELDS);writer.writeheader()
    for _ in range(2):writer.writerow(dict(entity_name='温盐环流',entity_description=TEXT,full_text=TEXT))
    store=Store(tmp_path/'live');store.import_file('a.csv',out.getvalue().encode(),'live',task_stage='knowledge');run(store)
    assert listing(store)['total']==1 and listing(store)['items'][0]['fact_count']==2
