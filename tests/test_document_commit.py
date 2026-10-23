"""File-level commits use existing review gates and one rollback boundary."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from qaproject.pipeline import approve, reject
from qaproject.store import encode
from test_bilingual import setup_pipeline, run


def client_for(app):
    client = TestClient(app)
    client.headers['X-QA-Session'] = client.get('/api/bootstrap').json()['session']
    return client


@pytest.mark.parametrize('damage',[False,True])
def test_v07_endpoint_keeps_atomic_review_gates(setup_pipeline,damage):
    app,store,doc_id,_,calls,_,_ = setup_pipeline
    doc = run(setup_pipeline)
    call_count = len(calls)
    if damage:
        with store.connect() as db:
            db.execute("UPDATE candidates SET row_hash='stale' WHERE id=?",(doc['candidates'][-1]['id'],))
    response = client_for(app).post(f'/api/documents/{doc_id}/approve_all')
    assert response.status_code==(400 if damage else 200)
    assert store.document(doc_id)['qa_count']==(0 if damage else 10)
    if not damage:
        assert response.json()=={'approved':10,'failed':[]}
    assert len(calls)==call_count


def test_whole_file_commits_both_languages_once_and_only_selected_file(setup_pipeline):
    app,store,doc_id,_,calls,_,_ = setup_pipeline
    doc = run(setup_pipeline)
    other_id,_ = store.import_file('other.txt',b'other actual input','live')
    call_count = len(calls)
    before = {c['id']:c['row_hash'] for c in doc['candidates']}
    client = client_for(app)
    assert client.get('/api/documents?summary_only=true').json()[1]['pending_commit_count']==10
    response = client.post(f'/api/documents/{doc_id}/approve')
    assert response.status_code==200, response.text
    assert response.json()=={'approved':True,'committed_count':10,'qa_count':10}
    after = store.document(doc_id)
    assert after['qa_pair_count']==5 and after['status']=='completed'
    assert before=={c['id']:c['row_hash'] for c in after['candidates']}
    assert store.document(other_id)['qa_count']==0 and len(calls)==call_count
    assert client.post(f'/api/documents/{doc_id}/approve').json()['committed_count']==0
    assert store.document(doc_id)['qa_count']==10
    summary = next(d for d in client.get('/api/documents?summary_only=true').json() if d['id']==doc_id)
    assert summary['pending_commit_count']==0


@pytest.mark.parametrize('damage',['review','row_hash','source','evidence'])
def test_any_invalid_record_rolls_back_whole_batch(setup_pipeline, damage):
    app,store,doc_id,_,_,_,_ = setup_pipeline
    doc = run(setup_pipeline)
    with store.connect() as db:
        last = db.execute('SELECT * FROM candidates WHERE document_id=? ORDER BY created DESC,id DESC',(doc_id,)).fetchone()
        if damage=='review':
            review = json.loads(last['review']); review['supported']=False
            db.execute('UPDATE candidates SET review=? WHERE id=?',(encode(review),last['id']))
        elif damage=='row_hash':
            db.execute("UPDATE candidates SET row_hash='stale' WHERE id=?",(last['id'],))
        elif damage=='evidence':
            db.execute("UPDATE candidates SET evidence_hash='changed' WHERE id=?",(last['id'],))
    if damage=='source':
        Path(doc['path']).write_text('changed actual source')
    client = client_for(app)
    response = client.post(f'/api/documents/{doc_id}/approve')
    assert response.status_code==400
    assert store.document(doc_id)['qa_count']==0
    assert all(c['status']=='pending' for c in store.document(doc_id)['candidates'])
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='human_review_commit'").fetchone()[0]==0


def test_preserves_prior_commits_and_skips_excluded_pairs(setup_pipeline):
    app,store,doc_id,_,_,_,_ = setup_pipeline
    doc = run(setup_pipeline)
    pairs = list(dict.fromkeys(c['pair_id'] for c in doc['candidates']))
    for c in doc['candidates']:
        if c['pair_id']==pairs[0]:
            approve(store,c['id'],True,c['row_hash'])
        elif c['pair_id']==pairs[1]:
            reject(store,c['id'],c['row_hash'])
    result = client_for(app).post(f'/api/documents/{doc_id}/approve')
    assert result.status_code==200, result.text
    assert result.json()['committed_count']==6 and result.json()['qa_count']==8
    assert store.document(doc_id)['qa_pair_count']==4


def test_one_excluded_translation_cannot_create_partial_pair(setup_pipeline):
    app,store,doc_id,_,_,_,_ = setup_pipeline
    doc = run(setup_pipeline)
    c = doc['candidates'][-1]
    reject(store,c['id'],c['row_hash'])
    result = client_for(app).post(f'/api/documents/{doc_id}/approve')
    assert result.status_code==400 and '配对' in result.json()['detail']
    assert store.document(doc_id)['qa_count']==0


def test_blocks_busy_files_missing_files_and_requests_without_session(setup_pipeline):
    app,store,doc_id,_,_,_,_ = setup_pipeline
    run(setup_pipeline)
    client = client_for(app)
    assert TestClient(app).post(f'/api/documents/{doc_id}/approve').status_code==403
    assert client.post('/api/documents/missing/approve').status_code==400
    assert client.post(f'/api/documents/{doc_id}/approve?mode=demo').status_code==400
    with store.connect() as db:
        db.execute("UPDATE tasks SET status='queued' WHERE document_id=?",(doc_id,))
    assert client.post(f'/api/documents/{doc_id}/approve').status_code==400
    assert store.document(doc_id)['qa_count']==0
