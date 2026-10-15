import json
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

from fastapi.testclient import TestClient
import pytest

from qaproject.app import create_app
from qaproject.semantic import (SemanticSearch, SearchCoordinator, active_topics,
                               catalog_version, cache_connection)
from qaproject.store import Store


def topic_with_file(store, name, definition='用于测试的独立知识主题。', keywords=None):
    topic = store.topic(name, definition, '隔离检索测试资料支持此主题。')
    doc_id, _ = store.import_file(name+'.txt', (name+'对应的隔离测试证据。').encode(), 'live')
    store.link_topic(doc_id, topic['id'], keywords or [], [])
    with store.connect() as db:
        db.execute("UPDATE tasks SET status='succeeded' WHERE document_id=?", (doc_id,))
        db.execute("UPDATE documents SET status='completed' WHERE id=?", (doc_id,))
    return topic, doc_id


def service_for(store):
    return SemanticSearch(store, {'enabled': True, 'min_similarity': 0.65})


def cached_scores(service, key, scores):
    topics = active_topics(service.store)
    version = catalog_version(topics, service.model_id)
    values = [scores.get(t['name'], 0.1) for t in topics]
    with cache_connection(service.path) as db:
        db.execute('INSERT OR REPLACE INTO rankings VALUES(?,?,?)',
                   (key, version, struct.pack('<'+'f'*len(values), *values)))


def test_exact_name_first_and_narrow_term_keeps_document_scope(tmp_path):
    store = Store(tmp_path/'live')
    topic_with_file(store, 'A海洋探测')
    topic_with_file(store, '海洋探测')
    _, narrow_doc = topic_with_file(store, '海洋观测', keywords=['海浪频谱反演'])
    doc, _ = store.import_file('other.txt', b'other evidence', 'live')
    store.link_topic(doc, store.topic('海洋观测')['id'], ['海水温度'], [])
    assert store.search('海洋探测')[0]['topic']['name'] == '海洋探测'
    assert [d['id'] for r in store.search('海浪频谱') for d in r['documents']] == [narrow_doc]


def test_ranking_top_five_excludes_direct_and_unassociated_topics(tmp_path):
    store = Store(tmp_path/'live')
    anchor, _ = topic_with_file(store, '海洋探测')
    for name in ['海洋探索', '海洋遥感', '物理海洋', '海底测绘', '声学观测', '海洋生态', '数学']:
        topic_with_file(store, name)
    service = service_for(store)
    cached_scores(service, 't:'+anchor['id'], {'海洋探测': 1, '海洋探索': .94, '海洋遥感': .9,
                  '物理海洋': .85, '海底测绘': .8, '声学观测': .75, '海洋生态': .7, '数学': .4})
    response = service.recommend('海洋探测')
    assert [r['topic']['name'] for r in response['recommendations']] == [
        '海洋探索', '海洋遥感', '物理海洋', '海底测绘', '声学观测']
    assert all('documents' not in r for r in response['recommendations'])  # only counts, no file bodies
    assert not service.recommend('不存在的输入')['recommendations']
    assert not service.recommend('')['recommendations']


def test_alias_uses_anchor_cache_even_when_pdf_busy_and_model_missing(tmp_path):
    store = Store(tmp_path/'live')
    topic, _ = topic_with_file(store, '海洋观测')
    topic_with_file(store, '海洋探索')
    service = service_for(store)
    service.model_dir = tmp_path/'missing'
    coordinator = SearchCoordinator([service])
    cached_scores(service, 't:'+topic['id'], {'海洋观测': 1, '海洋探索': .9})
    store.import_file('busy.txt', b'new PDF work', 'live')
    assert coordinator.busy()
    result = service.recommend('海洋监测')
    assert result['status'] == 'ready'
    assert result['recommendations'][0]['topic']['name'] == '海洋探索'
    assert service.recommend('未知查询')['status'] == 'unavailable'


def test_changed_definition_and_removed_file_invalidate_rankings(tmp_path):
    store = Store(tmp_path/'live')
    anchor, _ = topic_with_file(store, '海洋探测')
    other, doc = topic_with_file(store, '海洋探索')
    service = service_for(store)
    cached_scores(service, 't:'+anchor['id'], {'海洋探测': 1, '海洋探索': .9})
    assert service.recommend('海洋探测')['recommendations']
    with store.connect() as db:
        db.execute('UPDATE topics SET definition=? WHERE id=?', ('新的主题范围', other['id']))
    assert not service.recommend('海洋探测')['recommendations']
    with store.connect() as db:
        db.execute('DELETE FROM document_topics WHERE document_id=?', (doc,))
    assert not service.recommend('海洋探测')['recommendations']


def test_busy_defers_new_queries_and_pending_queue_is_bounded(tmp_path, monkeypatch):
    store = Store(tmp_path/'live')
    topic_with_file(store, '海洋观测')
    service = service_for(store)
    monkeypatch.setattr(service, 'model_available', lambda: True)
    coordinator = SearchCoordinator([service])
    store.import_file('busy.txt', b'pending file', 'live')
    for i in range(140):
        assert service.recommend(f'未收录表达{i}')['status'] == 'busy'
    with cache_connection(service.path) as db:
        assert db.execute('SELECT COUNT(*) FROM queries').fetchone()[0] == 128
    assert coordinator.process is None


def test_model_inference_not_imported_by_application():
    result = subprocess.run([sys.executable, '-c',
        "import sys; import qaproject.app; print(any(m in sys.modules for m in ['numpy','onnxruntime','tokenizers']))"],
        env={'PYTHONPATH': str(Path(__file__).resolve().parents[1]/'src')}, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'False'


def test_interrupt_terminates_embedding_process(tmp_path):
    service = service_for(Store(tmp_path/'live'))
    coordinator = SearchCoordinator([service])
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    coordinator.process = process
    try:
        coordinator.interrupt()
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_queued_file_preempts_running_index_job(tmp_path, monkeypatch):
    import qaproject.semantic as semantic
    store = Store(tmp_path/'live')
    topic_with_file(store, '海洋观测')
    service = service_for(store)
    coordinator = SearchCoordinator([service])
    real_popen = subprocess.Popen
    monkeypatch.setattr(semantic.subprocess, 'Popen', lambda args, **kwargs:
                        real_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs))
    job = {'topics': active_topics(store)}
    thread = threading.Thread(target=coordinator.run_job, args=(service, job))
    thread.start()
    try:
        deadline = time.monotonic()+3
        while coordinator.process is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert coordinator.process is not None
        store.import_file('incoming.pdf', b'synthetic pending PDF', 'live')
        thread.join(timeout=3)
        assert not thread.is_alive() and coordinator.process is None
        with cache_connection(service.path) as db:
            assert db.execute('SELECT status FROM attempts ORDER BY id DESC LIMIT 1').fetchone()[0] == 'interrupted'
    finally:
        coordinator.close()
        thread.join(timeout=3)


def test_pdf_worker_cancels_index_before_reading_evidence(tmp_path, monkeypatch):
    from qaproject.pipeline import Worker
    store = Store(tmp_path/'live')
    calls = []
    worker = Worker(store, before_process=lambda: calls.append('cancel_embedding'))

    def document(doc_id):
        calls.append('read_evidence')
        raise ValueError('stop after ordering check')

    monkeypatch.setattr(store, 'document', document)
    with pytest.raises(ValueError, match='ordering check'):
        worker.process({'document_id': 'test'})
    assert calls == ['cancel_embedding', 'read_evidence']


def test_empty_and_workspace_isolation_api(tmp_path):
    app = create_app(tmp_path/'app', seed=True, workers=False)
    client = TestClient(app)
    response = client.get('/api/search/semantic?q=海洋').json()
    assert response['total_topics'] == 0 and response['recommendations'] == []
    assert client.get('/api/search/semantic?mode=invalid').status_code == 400
    assert client.get('/api/search/semantic', params={'q': '字'*201}).status_code == 400
    assert app.state.searches['live'].path != app.state.searches['demo'].path


def test_incremental_vectors_resume_and_query_reuse(tmp_path):
    np = pytest.importorskip('numpy')
    from qaproject.embedding import build_index
    store = Store(tmp_path/'live')
    topic_with_file(store, '海洋观测')
    topic_with_file(store, '海洋探索')
    service = service_for(store)
    calls = []

    class Encoder:
        def __init__(self, directory):
            pass

        def encode(self, texts):
            calls.extend(texts)
            v = np.zeros((len(texts), 512), dtype='<f4')
            v[:, 0] = 1
            return v

    topics = active_topics(store)
    job = {'topics': topics, 'queries': [], 'model_id': service.model_id,
           'catalog': catalog_version(topics, service.model_id), 'cache': str(service.path), 'model_dir': ''}
    build_index(job, Encoder)
    assert len(calls) == 2
    build_index(job, Encoder)
    assert len(calls) == 2
    service.recommend('新的搜索表达')
    job['queries'] = [{'key': 'q:新的搜索表达', 'text': '新的搜索表达'}]
    build_index(job, Encoder)
    assert len(calls) == 3
    assert service.recommend('新的搜索表达')['status'] == 'ready'
    assert service.recommend('新的搜索表达')['recommendations']
    with store.connect() as db:
        db.execute('UPDATE topics SET definition=? WHERE id=?', ('更新定义', topics[0]['id']))
    job['topics'] = active_topics(store)
    job['catalog'] = catalog_version(job['topics'], service.model_id)
    build_index(job, Encoder)
    assert len(calls) == 4
