"""Durable QA vectors, complete checks and edit/generation failure isolation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qaproject.embedding import build_index, encode_only
from qaproject.semantic import (cache_connection, init_cache, encode_texts,
                               cross_document_duplicate, DedupUnavailable)
from qaproject.store import digest
from qaproject.pipeline import approve
from test_bilingual import setup_pipeline, run
from test_qa_summary import prepared, snapshot


class Encoder:
    encoded = []

    def __init__(self, directory):
        pass

    def encode(self, texts):
        self.encoded.extend(texts)
        vectors = np.zeros((len(texts), 512), dtype='<f4')
        for i, text in enumerate(texts):
            vectors[i, 0 if text.startswith('问题') else 1] = 1
        return vectors


def job(cache):
    return dict(cache=str(cache), model_dir='unused', model_id='test', catalog='v1',
                topics=[dict(id='topic', name='主题', definition='范围', aliases=[])], queries=[])


def test_upgrade_and_topic_rebuild_preserve_qa_cache(tmp_path):
    cache = tmp_path/'search.sqlite3'
    # Simulate a pre-upgrade shared table before any new code initializes it.
    with cache_connection(cache) as db:
        db.execute('CREATE TABLE vectors(hash TEXT PRIMARY KEY,vector BLOB NOT NULL)')
        vector = Encoder('unused').encode(['问题甲'])[0].tobytes()
        db.execute('INSERT INTO vectors VALUES(?,?)', (digest(['test','问题甲']), vector))
    build_index(job(cache), encoder_factory=Encoder)
    assert list(encode_texts(tmp_path/'missing','test',cache,['问题甲']))==['问题甲']
    with cache_connection(cache) as db:
        assert db.execute('SELECT COUNT(*) FROM qa_vectors').fetchone()[0]==1
        assert not db.execute('SELECT 1 FROM vectors WHERE hash=?', (digest(['test','问题甲']),)).fetchone()
    init_cache(cache)  # idempotent upgrade; do not keep copying search-only rows
    with cache_connection(cache) as db:
        assert db.execute('SELECT COUNT(*) FROM qa_vectors').fetchone()[0]==1


def child_runner(monkeypatch):
    calls=[]
    Encoder.encoded=[]
    monkeypatch.setattr('qaproject.embedding.LocalEmbedder', Encoder)

    def run_child(args, **kwargs):
        path=Path(args[-2])
        calls.append(json.loads(path.read_text(encoding='utf-8')))
        with monkeypatch.context() as patch:
            patch.setattr('qaproject.embedding.sys.argv', ['embedding',str(path),'encode-only'])
            encode_only()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr('qaproject.semantic.subprocess.run',run_child)
    return calls


def model_files(tmp_path):
    model=tmp_path/'model'
    model.mkdir()
    for name in ('model.onnx','tokenizer.json'):
        (model/name).write_bytes(b'isolated synthetic model placeholder')
    return model


def test_ten_thousand_vectors_reuse_after_rebuild_and_only_new_text_encodes(tmp_path,monkeypatch):
    cache=tmp_path/'search.sqlite3'
    model=model_files(tmp_path)
    calls=child_runner(monkeypatch)
    texts=[f'问题{i}' for i in range(10000)]
    assert len(encode_texts(model,'test',cache,texts))==10000
    assert len(Encoder.encoded)==10000 and all(len(c['texts'])<=256 for c in calls)
    assert len(calls)==40
    build_index(job(cache),encoder_factory=Encoder)
    calls.clear();Encoder.encoded=[]
    assert len(encode_texts(model,'test',cache,texts))==10000
    assert not calls and not Encoder.encoded
    assert len(encode_texts(model,'test',cache,[*texts,'问题新']))==10001
    assert len(calls)==1 and Encoder.encoded==['问题新']
    store=references(tmp_path,texts)
    assert cross_document_duplicate(store,model,'test',cache,'new',['问题新'],.93)=={'问题新':1.0}
    assert len(calls)==1, 'A 10,000-question comparison must reuse the warmed vectors.'
    assert cross_document_duplicate(store,model,'test',cache,'new',['不同方向'],.93)=={}
    assert len(calls)==2 and Encoder.encoded==['问题新','不同方向']
    assert not list(tmp_path.glob('dedup-job-*.json'))


def test_timeout_keeps_completed_vectors_for_retry(tmp_path,monkeypatch):
    import subprocess
    cache=tmp_path/'search.sqlite3'
    model=model_files(tmp_path)
    child_runner(monkeypatch)
    real_run=__import__('qaproject.semantic',fromlist=['subprocess']).subprocess.run
    count=0

    def interrupted(args, **kwargs):
        nonlocal count
        count+=1
        if count==2:
            raise subprocess.TimeoutExpired(args,kwargs['timeout'])
        return real_run(args,**kwargs)

    monkeypatch.setattr('qaproject.semantic.subprocess.run',interrupted)
    texts=[f'问题{i}' for i in range(300)]
    with pytest.raises(DedupUnavailable) as caught:
        encode_texts(model,'test',cache,texts)
    assert caught.value.reason=='timeout'
    assert len(Encoder.encoded)==256
    calls=child_runner(monkeypatch)
    assert len(encode_texts(model,'test',cache,texts))==300
    assert len(calls)==1 and len(Encoder.encoded)==44


@pytest.mark.parametrize('damage',['missing','nan','zero','partial','exit'])
def test_invalid_child_result_is_not_success(tmp_path,monkeypatch,damage):
    model=model_files(tmp_path)
    def bad(args,**kwargs):
        path=Path(args[-2]);payload=json.loads(path.read_text(encoding='utf-8'))
        raw=(np.full(512,np.nan,dtype='<f4') if damage=='nan' else np.zeros(512,dtype='<f4')).tobytes().hex()
        values={} if damage=='missing' else {digest(['test',payload['texts'][0]]):'00' if damage=='partial' else raw}
        path.write_text(json.dumps({'vectors':values}), encoding='utf-8')
        return SimpleNamespace(returncode=1 if damage=='exit' else 0)
    monkeypatch.setattr('qaproject.semantic.subprocess.run',bad)
    with pytest.raises(DedupUnavailable):
        encode_texts(model,'test',tmp_path/'cache.sqlite3',['问题'])


def references(tmp_path, texts):
    path=tmp_path/'business.sqlite3'
    with cache_connection(path) as db:
        db.executescript('CREATE TABLE candidates(id,status,document_id); CREATE TABLE qas(id,row_json);')
        for i,text in enumerate(texts):
            db.execute('INSERT INTO candidates VALUES(?,?,?)',(str(i),'approved','other'))
            db.execute('INSERT INTO qas VALUES(?,?)',(str(i),json.dumps({'input':text,'language':'zh'})))
    return SimpleNamespace(connect=lambda:cache_connection(path))


def test_exact_duplicate_needs_no_model_and_empty_pool_skips_encoding(tmp_path,monkeypatch):
    store=references(tmp_path,['相同问题'])
    def forbidden(*a,**kw):
        raise AssertionError('Exact match needs no encoding')
    monkeypatch.setattr('qaproject.semantic.encode_texts',forbidden)
    args=(store,tmp_path/'missing','test',tmp_path/'cache.sqlite3')
    assert cross_document_duplicate(*args,'new',['相同问题'],.93)=={'相同问题':1.0}
    assert cross_document_duplicate(*args,'other',['任意问题'],.93)=={}


def test_reference_failure_and_partial_vectors_are_incomplete(tmp_path,monkeypatch):
    store=references(tmp_path,['其他问题'])
    args=(store,tmp_path/'model','test',tmp_path/'cache.sqlite3','new',['问题'],.93)
    for result in (None,{}, {'问题':np.ones(512,dtype='<f4')}):
        monkeypatch.setattr('qaproject.semantic.encode_texts',lambda *a,**kw:result)
        assert cross_document_duplicate(*args) is None


@pytest.mark.parametrize('exception',[False,True])
def test_generation_incomplete_check_blocks_and_retry_reuses_model_responses(setup_pipeline,monkeypatch,exception):
    _,store,doc_id,worker,calls,_,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=1
    def unavailable(*a,**kw):
        if exception:
            raise RuntimeError('private path and provider content')
        return None
    monkeypatch.setattr('qaproject.pipeline.cross_document_duplicate',unavailable)
    doc=run(setup_pipeline)
    assert doc['status']=='failed' and '跨文档查重未完成' in doc['error'] and 'private' not in doc['error']
    for c in doc['candidates']:
        assert c['status']=='needs_review'
        with pytest.raises(ValueError):
            approve(store,c['id'],True,c['row_hash'])
    with store.connect() as db:
        event=json.loads(db.execute("SELECT detail FROM events WHERE kind='cross_dedup_unavailable'").fetchone()[0])
        assert event['stage']=='cross_dedup' and event['finish_reason']=='incomplete' and 'seconds' in event
        db.execute("UPDATE tasks SET status='queued' WHERE document_id=? AND status='failed'",(doc_id,))
    monkeypatch.setattr('qaproject.pipeline.cross_document_duplicate',lambda *a,**kw:{})
    calls.clear()
    worker.process(store.claim(worker.name,document_id=doc_id))
    assert store.document(doc_id)['status']=='waiting_review'
    assert all(kind!='generation' for kind,_,_ in calls)


@pytest.mark.parametrize('result',[None,{'重复问题':.98},{}])
def test_edit_checks_cross_document_duplicates_without_overwriting_original(setup_pipeline,monkeypatch,result):
    _,store,doc_id,worker,calls,_,_=setup_pipeline
    client,target,versions=prepared(setup_pipeline)
    old=snapshot(store)
    checked=[]
    def check(*args,**kwargs):
        checked.extend(args[5]);return result
    monkeypatch.setattr('qaproject.pipeline.cross_document_duplicate',check)
    assert client.post('/api/qas/'+target['id']+'/revisions',json={'versions':versions}).status_code==200
    calls.clear()
    worker.process(store.claim(worker.name,document_id=doc_id))
    rev=client.get('/api/qas/'+target['id']).json()['revision']
    assert checked and [kind for kind,_,_ in calls]==['review']
    if result is None:
        assert rev['status']=='failed' and '跨文档查重未完成' in rev['note']
    elif result:
        assert rev['status']=='rejected' and '高度相似' in rev['note']
    else:
        assert rev['status']=='applied' and snapshot(store)!=old
    if result is None or result:
        assert snapshot(store)==old


def test_missing_model_is_explicit_but_warm_cache_needs_no_model(tmp_path):
    cache=tmp_path/'cache.sqlite3'
    init_cache(cache)
    with pytest.raises(DedupUnavailable) as caught:
        encode_texts(tmp_path/'missing','test',cache,['问题'])
    assert caught.value.reason=='model_missing'
    with cache_connection(cache) as db:
        db.execute('INSERT INTO qa_vectors VALUES(?,?)',
                   (digest(['test','问题']),Encoder('unused').encode(['问题'])[0].tobytes()))
    assert list(encode_texts(tmp_path/'missing','test',cache,['问题']))==['问题']


def test_stop_at_batch_boundary_keeps_progress(tmp_path,monkeypatch):
    from qaproject.execution import WorkInterrupted
    cache=tmp_path/'cache.sqlite3'
    model=model_files(tmp_path)
    child_runner(monkeypatch)
    def check():
        if len(Encoder.encoded)>=256:
            raise WorkInterrupted()
    with pytest.raises(WorkInterrupted):
        encode_texts(model,'test',cache,[f'问题{i}' for i in range(300)],check=check)
    with cache_connection(cache) as db:
        assert db.execute('SELECT COUNT(*) FROM qa_vectors').fetchone()[0]==256
    assert not list(tmp_path.glob('dedup-job-*.json'))


def test_failed_rereview_invalidates_previous_passing_draft(setup_pipeline,monkeypatch):
    _,store,doc_id,worker,_,_,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=1
    doc=run(setup_pipeline)
    candidate=doc['candidates'][0]
    with store.connect() as db:
        db.execute("INSERT INTO tasks(id,document_id,status,stage,review_candidate_id) VALUES('rereview',?,'queued','reviewing',?)",
                   (doc_id,candidate['id']))
    monkeypatch.setattr('qaproject.pipeline.cross_document_duplicate',lambda *a,**kw:None)
    worker.process(store.claim(worker.name,document_id=doc_id))
    doc=store.document(doc_id)
    assert doc['status']=='failed'
    for c in doc['candidates']:
        assert c['status']=='needs_review' and not c['review'].get('binding')
        with pytest.raises(ValueError):
            approve(store,c['id'],True,c['row_hash'])
