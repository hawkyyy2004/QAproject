"""QA edits preserve formal records until paired review and atomic commit succeed."""
import json

import pytest
from fastapi.testclient import TestClient

from qaproject.pipeline import approve_document
from qaproject.store import encode
from test_bilingual import setup_pipeline, run
from test_document_commit import client_for


def prepared(setup):
    app,store,doc_id,worker,*_ = setup
    run(setup)
    approve_document(store,doc_id)
    client = client_for(app)
    listing = client.get('/api/qas').json()
    target = listing['items'][0]
    detail = client.get('/api/qas/'+target['id']).json()
    versions = [dict(id=v['id'],expected_hash=v['row_hash'],
                     **{k:v['row'][k] for k in ('input','thinking','output')}) for v in detail['versions']]
    versions[0]['thinking'] += '（人工复核修改）'
    return client,target,versions


def snapshot(store):
    with store.connect() as db:
        return [tuple(r) for r in db.execute('SELECT * FROM qas ORDER BY id')]


def test_list_committed_only_newest_first_stable_pages(setup_pipeline):
    app,store,doc_id,*_ = setup_pipeline
    doc = run(setup_pipeline)
    client = client_for(app)
    assert client.get('/api/qas').json()['total']==0
    approve_document(store,doc_id)
    with store.connect() as db:
        db.execute('UPDATE qas SET created=100')
        db.execute('UPDATE qas SET created=200 WHERE id=?',(doc['candidates'][0]['id'],))
    all_items = client.get('/api/qas').json()['items']
    assert all_items[0]['id']==doc['candidates'][0]['id']
    pages = [client.get('/api/qas',params={'page':i,'page_size':3}).json() for i in range(1,5)]
    assert [v['id'] for p in pages for v in p['items']]==[v['id'] for v in all_items]
    assert client.get('/api/qas?page=999').json()['page']==1
    assert client.get('/api/qas?page=0').status_code==422
    assert client.get('/api/qas?page_size=10000').status_code==422


def test_review_edits_pair_once_preserving_ids_dates_exports_and_history(setup_pipeline):
    _,store,doc_id,worker,calls,*_ = setup_pipeline
    client,target,versions = prepared(setup_pipeline)
    old = snapshot(store)
    response = client.post('/api/qas/'+target['id']+'/revisions',json={'versions':versions})
    assert response.status_code==200,response.text
    assert snapshot(store)==old
    assert client.post('/api/qas/'+target['id']+'/revisions',json={'versions':versions}).status_code==400
    calls.clear()
    task = store.claim(worker.name,document_id=doc_id)
    worker.process(task)
    assert [c[0] for c in calls]==['review']
    assert len(calls[0][1]['candidates'][0]['versions'])==2
    assert client.get('/api/qas/'+target['id']).json()['revision']['status']=='applied'
    after = snapshot(store)
    assert len(after)==10 and [(r[0],r[2]) for r in old]==[(r[0],r[2]) for r in after]
    detail = client.get('/api/qas/'+target['id']).json()
    assert detail['versions'][0]['row']['thinking']==versions[0]['thinking']
    assert '人工复核修改' in client.get('/api/export').content.decode('utf-8-sig')
    with store.connect() as db:
        rev = db.execute('SELECT * FROM qa_revisions').fetchone()
        history=json.loads(rev['before_json'])
        assert {r['id']:r['row_json'] for r in history}=={r[0]:r[1] for r in old if r[0] in {v['id'] for v in history}}
        assert len(json.loads(rev['review'])['binding']['candidate_hashes'])==2
    worker.process(task)  # stale replay cannot apply twice
    assert snapshot(store)==after


@pytest.mark.parametrize('damage',['translation','evidence','source','concurrent','lease','duplicate'])
def test_failed_edit_never_changes_formal_records(setup_pipeline,damage):
    _,store,doc_id,worker,_,behavior,_ = setup_pipeline
    client,target,versions = prepared(setup_pipeline)
    old = snapshot(store)
    assert client.post('/api/qas/'+target['id']+'/revisions',json={'versions':versions}).status_code==200
    if damage=='translation':
        def bad(result,*_):
            result['checks'][0]['translation_consistent']=False
            return result
        behavior['review']=bad
    elif damage=='evidence':
        with store.connect() as db:
            db.execute("UPDATE candidates SET evidence_hash='invalid' WHERE id=?",(target['id'],))
    elif damage=='source':
        from pathlib import Path
        Path(store.document(doc_id)['path']).write_text('changed')
    elif damage=='concurrent':
        with store.connect() as db:
            db.execute("UPDATE candidates SET review='{}' WHERE id=?",(target['id'],))
    elif damage=='duplicate':
        with store.connect() as db:
            rev = db.execute('SELECT * FROM qa_revisions').fetchone()
            changes=json.loads(rev['after_json'])
            others=[c for c in store.document(doc_id)['candidates'] if c['id'] not in {v['id'] for v in changes}]
            changes[0]['row']['input']=others[0]['row']['input']
            db.execute('UPDATE qa_revisions SET after_json=?',(encode(changes),))
    task = store.claim(worker.name,document_id=doc_id)
    if damage=='lease':
        with store.connect() as db:
            db.execute('UPDATE tasks SET lease_until=0 WHERE id=?',(task['id'],))
    worker.process(task)
    assert snapshot(store)==old
    state = client.get('/api/qas/'+target['id']).json()['revision']['status']
    assert state==('rejected' if damage=='translation' else 'queued' if damage=='lease' else 'failed')


def test_edit_validation_sessions_versions_and_fixed_fields(setup_pipeline):
    app,store,_,_,calls,*_ = setup_pipeline
    client,target,versions = prepared(setup_pipeline)
    url='/api/qas/'+target['id']+'/revisions'
    assert TestClient(app).post(url,json={'versions':versions}).status_code==403
    assert client.post(url+'?mode=demo',json={'versions':versions}).status_code==400
    assert client.post(url,json={'versions':versions[:1]}).status_code==400
    assert client.post(url,json={'versions':[versions[0],versions[0]]}).status_code==400
    changed=[dict(v) for v in versions];changed[0]['source']='forged'
    assert client.post(url,json={'versions':changed}).status_code==422
    changed=[dict(v) for v in versions];changed[0]['expected_hash']='stale'
    assert client.post(url,json={'versions':changed}).status_code==400
    changed=[dict(v) for v in versions];changed[0]['output']='   '
    assert client.post(url,json={'versions':changed}).status_code==400
    assert client.get('/api/qas/missing').status_code==400
    detail=client.get('/api/qas/'+target['id']).json()
    unchanged=[dict(id=v['id'],expected_hash=v['row_hash'],**{k:v['row'][k] for k in ('input','thinking','output')}) for v in detail['versions']]
    assert client.post(url,json={'versions':unchanged}).json()=={'unchanged':True}


def test_failed_request_can_retry_without_changing_old_data(setup_pipeline):
    _,store,doc_id,worker,_,behavior,_ = setup_pipeline
    client,target,versions = prepared(setup_pipeline)
    old = snapshot(store)
    def broken(*_):
        raise RuntimeError('test')
    behavior['review']=broken
    client.post('/api/qas/'+target['id']+'/revisions',json={'versions':versions})
    worker.process(store.claim(worker.name,document_id=doc_id))
    assert snapshot(store)==old
    assert client.get('/api/qas/'+target['id']).json()['revision']['status']=='failed'
    behavior.clear()
    assert client.post('/api/qas/'+target['id']+'/retry').status_code==200
    worker.process(store.claim(worker.name,document_id=doc_id))
    assert client.get('/api/qas/'+target['id']).json()['revision']['status']=='applied'
    assert client.post('/api/qas/'+target['id']+'/retry').status_code==400


def test_historical_single_and_incomplete_pair(setup_pipeline):
    _,store,doc_id,worker,*_ = setup_pipeline
    client,target,versions = prepared(setup_pipeline)
    with store.connect() as db:
        pair_id=db.execute('SELECT pair_id FROM candidate_pairs WHERE candidate_id=?',(target['id'],)).fetchone()[0]
        other=next(v['id'] for v in versions if v['id']!=target['id'])
        db.execute('DELETE FROM qas WHERE id=?',(other,))
    assert client.get('/api/qas/'+target['id']).status_code==400
    with store.connect() as db:
        db.execute('DELETE FROM candidate_pairs WHERE pair_id=?',(pair_id,))
    one=next(v for v in versions if v['id']==target['id'])
    one['thinking']+=' Historical revision.'
    assert client.post('/api/qas/'+target['id']+'/revisions',json={'versions':[one]}).status_code==200
    worker.process(store.claim(worker.name,document_id=doc_id))
    detail=client.get('/api/qas/'+target['id']).json()
    assert len(detail['versions'])==1 and detail['revision']['status']=='applied'
    assert store.document(doc_id)['qa_count']==9


def test_expired_revision_resumes_and_file_tasks_take_priority(setup_pipeline):
    _,store,doc_id,worker,calls,*_ = setup_pipeline
    client,target,versions = prepared(setup_pipeline)
    client.post('/api/qas/'+target['id']+'/revisions',json={'versions':versions})
    task=store.claim(worker.name,document_id=doc_id)
    with store.connect() as db:
        db.execute("UPDATE qa_revisions SET status='reviewing'")
        db.execute('UPDATE tasks SET lease_until=0 WHERE id=?',(task['id'],))
    other,_=store.import_file('priority.txt',b'new priority input','live')
    assert store.claim('other-worker')['document_id']==other
    calls.clear()
    worker.process(store.claim(worker.name,document_id=doc_id))
    assert [c[0] for c in calls]==['review']
    assert client.get('/api/qas/'+target['id']).json()['revision']['status']=='applied'


def test_rejected_draft_survives_reopening_and_can_be_corrected(setup_pipeline):
    _,store,doc_id,worker,_,behavior,_ = setup_pipeline
    client,target,versions = prepared(setup_pipeline)
    def failed(result,*_):
        result['checks'][0]['supported']=False
        return result
    behavior['review']=failed
    client.post('/api/qas/'+target['id']+'/revisions',json={'versions':versions})
    worker.process(store.claim(worker.name,document_id=doc_id))
    detail=client.get('/api/qas/'+target['id']).json()
    assert detail['revision']['status']=='rejected'
    assert detail['draft_versions'][0]['row']['thinking']==versions[0]['thinking']
    assert detail['versions'][0]['row']['thinking']!=versions[0]['thinking']
    behavior.clear()
    versions[0]['thinking']+='（已重新核实）'
    client.post('/api/qas/'+target['id']+'/revisions',json={'versions':versions})
    worker.process(store.claim(worker.name,document_id=doc_id))
    assert client.get('/api/qas/'+target['id']).json()['revision']['status']=='applied'


from test_generation_distribution import mixed, process


def test_visual_revision_sends_real_png_and_retains_frozen_evidence(mixed):
    from qaproject.pipeline import Worker
    from qaproject.qa_summary import edit_detail, queue_revision
    store,doc_id,_,calls,_ = mixed
    doc=process(store,doc_id)
    approve_document(store,doc_id)
    target=next(c for c in doc['candidates'] if c['row']['image'])
    detail=edit_detail(store,target['id'])
    versions=[dict(id=v['id'],expected_hash=v['row_hash'],**{k:v['row'][k] for k in ('input','thinking','output')}) for v in detail['versions']]
    versions[0]['thinking']+=' 人工补充核对。'
    queue_revision(store,target['id'],versions)
    calls.clear()
    worker=Worker(store)
    worker.process(store.claim(worker.name,document_id=doc_id))
    assert [c[0] for c in calls]==['review'] and len(calls[0][2])==1
    # The fixture decodes the HTTP data URL and compares actual PNG bytes.
    after=edit_detail(store,target['id'])
    assert after['revision']['status']=='applied'
    assert after['evidence']==detail['evidence']
    for old,new in zip(detail['versions'],after['versions']):
        assert all(old['row'][k]==new['row'][k] for k in ('source','image','language'))
