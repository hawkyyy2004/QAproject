"""Generate a single whole-file topic plan in the existing generation request."""
import csv
import io
import json

from fastapi.testclient import TestClient
import pytest

from qaproject.glm import GLMClient
from qaproject.pipeline import Worker, CHECKS, BILINGUAL_CHECKS, approve
from qaproject.store import Store, encode
from test_document_topics import evidence, proposal, facts
from test_bilingual import pair, setup_pipeline, run


def test_multibatch_whole_file_topics_share_first_generation_and_keep_call_count(tmp_path,monkeypatch):
    store=Store(tmp_path/'live')
    doc_id,_=store.import_file('whole.txt',b'isolated evidence','live')
    units=[evidence(n,'Synthetic calibrated observation. '*220) for n in range(1,6)]
    store.update_document(doc_id,units=units)
    calls=[]
    read=[]
    def call(self,instruction,payload,group=(),**kw):
        if payload.get('analysis_phase')=='facts':
            calls.append('facts')
            read.extend(u['id'] for u in group)
            assert not store.document(doc_id)['topics']
            return facts(group),{}
        if 'document_facts' in payload:
            calls.append('combined')
            assert len(read)==5 and 'target' in payload
            actual=payload['units'][0]['id']
            return dict(summary='完整海洋观测。',topics=[dict(proposal('海洋观测',units[0]),unit_ids=read)],
                        pairs=[pair(actual,n) for n in range(payload['target'])],skipped=[]),{}
        if 'candidates' in payload:
            calls.append('review')
            return dict(checks=[dict(candidate_id=c['candidate_id'],note='双语、数值和范围符合证据。',
                        **{k:True for k in CHECKS+BILINGUAL_CHECKS}) for c in payload['candidates']]),{}
        calls.append('generation')
        return dict(pairs=[pair(group[0]['id'],3+n) for n in range(payload['target'])],skipped=[]),{}
    monkeypatch.setattr(GLMClient,'call',call)
    worker=Worker(store)
    worker.process(store.claim(worker.name))
    doc=store.document(doc_id)
    assert doc['status']=='waiting_review' and doc['bilingual_count']==5,doc['error']
    assert len(doc['topics'])==1 and all(c['status']=='pending' for c in doc['candidates'])
    assert len(calls)==5+4 and calls.count('combined')==1
    assert calls.count('generation')==1 and calls.count('review')==2
    for c in doc['candidates']:
        approve(store,c['id'],True,c['row_hash'])
    assert store.document(doc_id)['qa_count']==10


def test_seventeen_legacy_labels_are_automatically_reclassified_and_do_not_block_commit(setup_pipeline,monkeypatch):
    app,store,doc_id,_,_,_,_=setup_pipeline
    original=run(setup_pipeline)
    old_topic=original['topics'][0]
    extra=[store.topic(f'历史细方向{n}','旧版细分方向。','隔离历史兼容测试。') for n in range(16)]
    with store.connect() as db:
        db.executemany('INSERT INTO document_topics VALUES(?,?,?,?)',
            [(doc_id,t['id'],'[]',encode([original['units'][0]['id']])) for t in extra])
    # Topic count is not a quality/commit rejection rule.
    first=original['candidates'][0]
    approve(store,first['id'],True,first['row_hash'])
    with store.connect() as db:
        candidates_before=[tuple(r) for r in db.execute('SELECT * FROM candidates ORDER BY id')]
        qas_before=[tuple(r) for r in db.execute('SELECT * FROM qas ORDER BY id')]
    reopened=Store(store.root)
    Store(store.root)  # Repeated startup must not enqueue duplicate jobs.
    with reopened.connect() as db:
        tasks=db.execute("SELECT * FROM tasks WHERE status='queued'").fetchall()
    assert len(tasks)==1 and tasks[0]['stage']=='topic_reclassification'
    plans=[]
    def call(self,instruction,payload,group=(),**kw):
        plans.append(payload)
        assert 'existing_topics' in payload and len(payload['legacy_topics'])==17
        return dict(summary='整篇海洋科学观测。',topics=[dict(proposal('海洋科学',group[0]),unit_ids=[u['id'] for u in group])],
                    legacy_topic_map={t['id']:'海洋科学' for t in payload['legacy_topics']}),{}
    monkeypatch.setattr(GLMClient,'call',call)
    worker=Worker(reopened)
    worker.process(reopened.claim(worker.name))
    doc=reopened.document(doc_id)
    assert len(plans)==1 and [t['name'] for t in doc['topics']]==['海洋科学'],doc['error']
    with reopened.connect() as db:
        assert [tuple(r) for r in db.execute('SELECT * FROM candidates ORDER BY id')]==candidates_before
        assert [tuple(r) for r in db.execute('SELECT * FROM qas ORDER BY id')]==qas_before
    assert all(c['canonical_topic_id']==doc['topics'][0]['id'] for c in doc['candidates'])
    for c in doc['candidates']:
        approve(reopened,c['id'],True,c['row_hash'])
    app.state.stores['live']=reopened
    response=TestClient(app).get('/api/export',params={'topic_id':doc['topics'][0]['id']})
    exported=list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
    assert len(exported)==10 and all(row['source'].endswith('|'+old_topic['name']) for row in exported)


def test_new_files_take_priority_over_automatic_historical_reclassification(setup_pipeline):
    _,store,doc_id,_,_,_,_=setup_pipeline
    run(setup_pipeline)
    store.reclassify_topics(doc_id)
    new_id,_=store.import_file('new.txt',b'new file has priority','live')
    assert store.claim('priority-worker')['document_id']==new_id


def test_reclassification_does_not_repoint_history_after_source_changes(setup_pipeline,monkeypatch):
    from pathlib import Path
    _,store,doc_id,_,_,_,_=setup_pipeline
    doc=run(setup_pipeline)
    hashes={c['id']:c['row_hash'] for c in doc['candidates']}
    store.reclassify_topics(doc_id)
    Path(doc['path']).write_bytes(b'changed source version')
    monkeypatch.setattr(GLMClient,'call',lambda *a,**k:pytest.fail('Changed source must stop before model calls'))
    worker=Worker(store)
    worker.process(store.claim(worker.name))
    after=store.document(doc_id)
    assert after['status']=='failed' and '原文件已变更' in after['error']
    assert {c['id']:c['row_hash'] for c in after['candidates']}==hashes
