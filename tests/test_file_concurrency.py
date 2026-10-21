"""Real queue / evidence / commits with isolated, simulated provider responses."""
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
import pytest

from qaproject.app import create_app
from qaproject.execution import ExecutionLimits, WorkInterrupted, execution_config
from qaproject.glm import GLMClient, ProviderError
from qaproject.pipeline import Worker, CHECKS, BILINGUAL_CHECKS, approve_document
from qaproject.settings import FIELDS, settings
from qaproject.store import Store, digest, encode, uid


@pytest.fixture
def limits():
    value = ExecutionLimits()
    yield value
    value.close()


def provider(monkeypatch, enter=None):
    calls = []
    lock = threading.Lock()
    cfg = dict(settings(), semantic_search={'enabled':False})
    monkeypatch.setattr('qaproject.pipeline.settings', lambda:cfg)

    def call(self, instruction, payload, units=(), **kwargs):
        kind = ('facts' if payload.get('analysis_phase')=='facts' else 'analysis' if 'existing_topics' in payload
                else 'review' if 'candidates' in payload else 'generation')
        doc = units[0]['id'].split(':')[0]
        with lock:
            calls.append((doc, kind))
        if enter:
            enter(doc, kind)
        if kind=='facts':
            return {'summary':'', 'facts':[dict(unit_id=u['id'],focus='合成观测资料。',keywords=['温度']) for u in units]}, {}
        result = {}
        if kind=='analysis':
            support = ([u['id'] for u in units] if 'document_facts' not in payload else
                       [ref for fact in payload['document_facts'] for ref in fact.get('unit_ids',[fact.get('unit_id')])])
            result.update(summary='隔离的并发观测。', topics=[dict(name='并发观测',definition='隔离观测领域。',
                new_reason='独立测试的观测领域。',unit_ids=support,keywords=['温度'])])
        if kind=='review':
            result['checks'] = [dict(candidate_id=c['candidate_id'],note='中英版本均符合原始观测证据。',
                **{k:True for k in CHECKS+BILINGUAL_CHECKS}) for c in payload['candidates']]
        elif kind=='generation' or 'target' in payload:
            start = len(payload['existing_questions'])//2
            result.update(pairs=[dict(unit_id=payload['units'][0]['id'],topic='并发观测',
                zh=dict(input=f'合成海域{doc}中指标{start+n}的观测结果是什么？',thinking='观测值为15。',output='观测值为15。'),
                en=dict(input=f'What is the measured value of indicator {start+n} in synthetic area {doc}?',
                        thinking='The measured value is 15.',output='The measured value is 15.'))
                for n in range(payload['target'])],skipped=[])
        return result, {'seconds':.001, 'model':'simulated'}

    monkeypatch.setattr(GLMClient, 'call', call)
    return calls, cfg


def test_two_files_overlap_third_queues_and_all_pairs_commit(tmp_path, monkeypatch, limits):
    store = Store(tmp_path/'shared')
    docs = [store.import_file(f'{n}.txt', f'合成观测{n}：温度、盐度等指标为15。'.encode(), 'live')[0] for n in range(3)]
    both = threading.Event()
    release = threading.Event()
    seen = set()
    lock = threading.Lock()

    def enter(doc, kind):
        if kind=='analysis':
            with lock:
                seen.add(doc)
                if len(seen)==2:
                    both.set()
            assert release.wait(5), 'two distinct file tasks must enter the model concurrently'

    calls, _ = provider(monkeypatch, enter)
    workers = [Worker(store, limits=limits) for _ in range(3)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = [pool.submit(w.run_once) for w in workers[:2]]
        try:
            assert both.wait(5)
            with store.connect() as db:
                assert db.execute("SELECT COUNT(*) FROM tasks WHERE status='running'").fetchone()[0]==2
                assert db.execute("SELECT COUNT(*) FROM tasks WHERE status='queued'").fetchone()[0]==1
            assert not workers[2].run_once()  # global file-slot limit, not merely a worker-list length
        finally:
            release.set()
        assert all(f.result(timeout=10) for f in running)
    assert workers[0].run_once()
    assert len({t['id'] for doc in docs for t in store.document(doc)['topics']})==1
    for doc_id in docs:
        doc = store.document(doc_id)
        assert doc['status']=='waiting_review', doc['error']
        assert doc['bilingual_count']==5 and len(doc['candidates'])==10
        assert all(c['unit_id'].startswith(doc_id+':') for c in doc['candidates'])
        assert all(c['row']['source'].startswith(doc['path']+'|') for c in doc['candidates'])
        result = approve_document(store, doc_id)
        assert result['committed_count']==10
        assert approve_document(store, doc_id)['committed_count']==0
        assert store.document(doc_id)['qa_pair_count']==5
        assert [k for d,k in calls if d==doc_id]==['analysis','generation','review','review','generation','review','review']
    app = create_app(store.root.parent/'api',workers=False)
    app.state.stores['live'] = store
    rows = list(csv.DictReader(io.StringIO(TestClient(app).get('/api/export').content.decode('utf-8-sig'))))
    assert len(rows)==30 and all(list(row)==FIELDS for row in rows)
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='model_error'").fetchone()[0]==0
        assert all(json.loads(r[0])['request_wait_seconds']>=0 for r in db.execute("SELECT detail FROM events WHERE kind='model_call'"))
    assert not limits.active_documents


@pytest.mark.parametrize('request_limit', [1,2])
def test_nested_analysis_and_files_share_request_ceiling(tmp_path, monkeypatch, request_limit):
    store = Store(tmp_path/'analysis')
    docs = [store.import_file(f'{n}.txt', str(n).encode(), 'live')[0] for n in range(2)]
    for doc in docs:
        units = []
        for n in range(4):
            u = dict(id=f'{doc}:p{n}:s1', text='温度观测资料。'*1000, image='',asset_id='',
                     page=n+1,seg=1,locator=f'page-{n+1:02d}-seg-01',language='zh')
            units.append(dict(u,hash=digest(u)))
        store.update_document(doc, units=units)
    lock = threading.Lock()
    active = peak = 0
    original = None
    calls, _ = provider(monkeypatch)
    original = GLMClient.call

    def measured(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak,active)
        try:
            time.sleep(.015)  # Simulated provider delay exercises overlapping in-flight requests.
            return original(*args,**kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(GLMClient,'call',measured)
    resources = ExecutionLimits(model_requests=request_limit)
    try:
        workers = [Worker(store,limits=resources) for _ in docs]
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert all(pool.map(lambda w:w.run_once(),workers))
        assert peak==request_limit and active==0
        for doc in docs:
            result = store.document(doc)
            assert result['status']=='waiting_review', result['error']
            assert result['bilingual_count']==5
            assert len([kind for d,kind in calls if d==doc and kind=='facts'])==4
    finally:
        resources.close()


def test_claims_exclude_same_document_and_preserve_task_priority(tmp_path):
    store = Store(tmp_path/'claims')
    a,_ = store.import_file('a.txt',b'a','live')
    b,_ = store.import_file('b.txt',b'b','live')
    extra = uid()
    with store.connect() as db:
        db.execute('INSERT INTO tasks(id,document_id,status,stage) VALUES(?,?,?,?)',(extra,a,'queued','quality_queued'))
    first = store.claim('first',a)
    assert store.claim('blocked',a) is None
    second = store.claim('second')
    assert second['document_id']==b
    store.task_update(first['id'],'first','done','succeeded')
    assert store.claim('third')['id']==extra


def test_expired_live_file_cannot_overlap_an_old_slot(tmp_path, limits):
    store = Store(tmp_path/'lease')
    a,_ = store.import_file('a.txt',b'a','live')
    b,_ = store.import_file('b.txt',b'b','live')
    first = limits.claim(store,'first')
    with store.connect() as db:
        db.execute('UPDATE tasks SET lease_until=0 WHERE id=?',(first['id'],))
    second = limits.claim(store,'second')
    assert first['document_id']==a and second['document_id']==b
    assert limits.claim(store,'third') is None
    limits.release(store,first)
    replacement = limits.claim(store,'third')
    assert replacement['id']==first['id']
    with pytest.raises(ValueError,match='租约'):
        store.task_update(first['id'],'first','late','succeeded')
    limits.release(store,second)
    limits.release(store,replacement)


def test_concurrent_duplicate_import_creates_one_original_and_task(tmp_path):
    store = Store(tmp_path/'imports')
    barrier = threading.Barrier(2)
    def import_one(n):
        barrier.wait(timeout=3)
        return store.import_file(f'{n}.txt', b'identical synthetic evidence', 'live')
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(import_one,range(2)))
    assert len({r[0] for r in result})==1 and sum(r[1] for r in result)==1
    assert len(list((store.root/'originals').glob('*/*')))==1
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]==1


def test_parser_is_single_thread_and_does_not_block_model_waits(limits):
    stop = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    thread_ids = []
    def parse():
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(3)
        return 'prepared'
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(limits.parse,parse,stop,lambda:None)
        assert entered.wait(2)
        second = pool.submit(limits.parse,parse,stop,lambda:None)
        with limits.request(stop,lambda:None):
            assert len(thread_ids)==1
        release.set()
        assert first.result(timeout=3)[0]==second.result(timeout=3)[0]=='prepared'
    assert len(thread_ids)==2 and len(set(thread_ids))==1


def test_native_parser_timeout_does_not_loop_or_poison_the_parser(limits):
    def failed():
        raise TimeoutError('isolated parser timeout')
    with pytest.raises(TimeoutError,match='isolated parser'):
        limits.parse(failed,threading.Event(),lambda:None)
    assert limits.parse(lambda:'next file',threading.Event(),lambda:None)[0]=='next file'


def test_stop_after_inflight_call_keeps_cache_and_requeues(tmp_path, monkeypatch, limits):
    store = Store(tmp_path/'stop')
    doc,_ = store.import_file('a.txt','合成观测值15。'.encode(),'live')
    entered = threading.Event()
    release = threading.Event()
    def enter(doc,kind):
        if kind=='analysis':
            entered.set()
            assert release.wait(3)
    calls,_ = provider(monkeypatch,enter)
    worker = Worker(store,limits=limits)
    worker.start()
    try:
        assert entered.wait(3)
        worker.stop.set()
    finally:
        release.set()
        worker.thread.join(timeout=5)
    assert not worker.thread.is_alive()
    assert store.document(doc)['status']=='queued'
    assert calls==[(doc,'analysis')]
    cached = {p.name:p.read_bytes() for p in (store.root/'runs').glob('*.json')}
    assert cached and not list((store.root/'runs').glob('*.tmp'))
    assert Worker(store,limits=limits).run_once()
    result = store.document(doc)
    assert result['status']=='waiting_review' and result['bilingual_count']==5, result['error']
    assert len([1 for _,kind in calls if kind=='analysis'])==1
    assert all((store.root/'runs'/name).read_bytes()==data for name,data in cached.items())


@pytest.mark.parametrize('failure',['provider','parser'])
def test_failure_releases_slots_and_other_file_completes(tmp_path, monkeypatch, limits, failure):
    store = Store(tmp_path/'failure')
    failed,_ = store.import_file('bad.pdf' if failure=='parser' else 'bad.txt',b'bad synthetic evidence','live')
    good,_ = store.import_file('good.txt',b'good synthetic evidence','live')
    def enter(doc,kind):
        if doc==failed:
            raise ProviderError('模拟超时','timeout',{'seconds':.01})
    provider(monkeypatch,enter)
    workers = [Worker(store,limits=limits) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(lambda w:w.run_once(),workers))
    assert store.document(failed)['status']=='failed'
    assert store.document(good)['bilingual_count']==5
    with limits.request(threading.Event(),lambda:None):
        pass
    assert not limits.active_documents


def test_two_pdfs_keep_full_pages_and_distinct_frozen_sources(tmp_path,monkeypatch,limits):
    import pymupdf
    store = Store(tmp_path/'pdfs')
    docs = []
    for n in range(2):
        with pymupdf.open() as pdf:
            for page in range(2):
                pdf.new_page().insert_text((40,60),f'Synthetic area {n}: temperature and salinity measurements are 15. Page {page+1}.')
            docs.append(store.import_file(f'{n}.pdf',pdf.tobytes(),'live')[0])
    provider(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(lambda _:Worker(store,limits=limits).run_once(),docs))
    for doc in docs:
        result = store.document(doc)
        assert result['status']=='waiting_review', result['error']
        assert result['pages']==2 and result['bilingual_count']==5
        assert {u['page'] for u in result['units']}=={1,2}
        assert not result['assets']
        assert all(c['row']['source'].startswith(result['path']+'|page-') for c in result['candidates'])


def test_two_images_send_their_own_original_bytes_in_generation_and_review(tmp_path,monkeypatch,limits):
    import base64
    import httpx
    import pymupdf
    from test_bilingual import pair
    from test_prototype import fake_provider
    store = Store(tmp_path/'images')
    originals = {}
    for n,extension in enumerate(('png','jpeg')):
        pix = pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,200,120),False)
        pix.clear_with(60+n*70)
        raw = pix.tobytes(extension)
        doc,_ = store.import_file(f'{n}.{extension}',raw,'live')
        originals[raw] = doc
    seen = {doc:[] for doc in originals.values()}
    barrier = threading.Barrier(2)
    cfg = dict(settings(),semantic_search={'enabled':False})
    monkeypatch.setattr('qaproject.pipeline.settings',lambda:cfg)
    def response(request):
        body = json.loads(request.content)
        content = body['messages'][1]['content']
        payload = json.loads(content[0]['text'])
        images = [p['image_url']['url'] for p in content if p['type']=='image_url']
        assert len(images)==1
        raw = base64.b64decode(images[0].split(',',1)[1])
        doc = originals[raw]
        kind = 'analysis' if 'existing_topics' in payload else 'review' if 'candidates' in payload else 'generation'
        if kind=='analysis':
            barrier.wait(timeout=4)
        elif kind=='review':
            assert all(v['qa']['image'].startswith(str(store.root/'assets/images'/doc))
                       for c in payload['candidates'] for v in c['versions'])
        seen[doc].append(kind)
        if kind=='generation':
            result = dict(pairs=[pair(payload['units'][0]['id'],n) for n in range(payload['target'])],skipped=[])
            return httpx.Response(200,json={'choices':[dict(finish_reason='stop',message={'content':encode(result)})]})
        body['messages'][1]['content'] = encode(payload)
        return fake_provider(httpx.Request('POST','https://fixture.invalid',content=encode(body)))
    client = httpx.Client
    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:client(transport=httpx.MockTransport(response)))
    monkeypatch.setattr('qaproject.glm.api_key',lambda:'isolated-placeholder')
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(lambda _:Worker(store,limits=limits).run_once(),originals))
    for raw,doc in originals.items():
        result = store.document(doc)
        assert result['status']=='waiting_review', result['error']
        assert result['bilingual_count']==2 and result['qa_count']==0
        assert seen[doc]==['analysis','generation','review','review']
        assert all(Path(c['row']['image']).read_bytes()==raw for c in result['candidates'])
        assert all('|image-0001|' in c['row']['source'] for c in result['candidates'])


def test_stopped_revision_stays_resumable_and_cannot_overwrite_old_rows(tmp_path,monkeypatch,limits):
    from qaproject.qa_summary import queue_revision, edit_detail
    store = Store(tmp_path/'revision')
    doc,_ = store.import_file('edit.txt','合成观测值15。'.encode(),'live')
    calls,_ = provider(monkeypatch)
    assert Worker(store,limits=limits).run_once()
    approve_document(store,doc)
    target = store.document(doc)['candidates'][0]
    detail = edit_detail(store,target['id'])
    versions = [dict(id=v['id'],expected_hash=v['row_hash'],**{k:v['row'][k] for k in ('input','thinking','output')})
                for v in detail['versions']]
    for v in versions:
        v['output'] += '（核对）' if '观测' in v['output'] else ' (verified)'
    queue_revision(store,target['id'],versions)
    entered = threading.Event()
    original_request = limits.request
    def request(*args,**kwargs):
        entered.set()
        return original_request(*args,**kwargs)
    monkeypatch.setattr(limits,'request',request)
    worker = Worker(store,limits=limits)
    # Fill both provider slots so the edit must wait before any request is made.
    limits.requests.acquire()
    limits.requests.acquire()
    try:
        worker.start()
        assert entered.wait(3)
        worker.stop.set()
        worker.thread.join(timeout=3)
        assert not worker.thread.is_alive()
    finally:
        limits.requests.release()
        limits.requests.release()
    after = edit_detail(store,target['id'])
    assert after['revision']['status']=='queued'
    assert store.document(doc)['status']=='completed'
    assert after['versions']==detail['versions']
    count = len(calls)
    assert Worker(store,limits=limits).run_once()
    assert len(calls)==count+1
    assert edit_detail(store,target['id'])['revision']['status']=='applied'


def test_waiting_request_renews_lease_and_can_stop_without_leaking_permit():
    limits = ExecutionLimits(model_requests=1)
    holder = threading.Event()
    stop = threading.Event()
    entered = threading.Event()
    renewed = threading.Event()
    def waiting():
        entered.set()
        with limits.request(stop,renewed.set):
            pytest.fail('stopped waiter must not issue a request')
    try:
        with limits.request(holder,lambda:None):
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(waiting)
                assert entered.wait(2) and renewed.wait(2)
                stop.set()
                with pytest.raises(WorkInterrupted):
                    result.result(timeout=2)
        with limits.request(holder,lambda:None):
            pass
    finally:
        limits.close()


@pytest.mark.parametrize('key,value',[('document_concurrency',0),('document_concurrency',3),
    ('model_request_concurrency',False),('model_request_concurrency','2')])
def test_invalid_concurrency_fails_explicitly(key,value):
    with pytest.raises(ValueError,match=key):
        execution_config({key:value})


@pytest.mark.parametrize('count',[1,2])
def test_application_owns_shared_limits_and_joins_slots_on_shutdown(tmp_path,monkeypatch,count):
    monkeypatch.setattr('qaproject.app.settings',lambda:dict(settings(),document_concurrency=count))
    app = create_app(tmp_path/'lifecycle')
    assert len(app.state.runners)==count
    assert len({w.name for w in app.state.runners})==count
    assert all(w.limits is app.state.execution_limits for w in app.state.runners)
    with TestClient(app):
        assert all(w.thread.is_alive() for w in app.state.runners)
    assert not any(w.thread.is_alive() for w in app.state.runners)
    assert not app.state.search_runner.thread.is_alive()


def test_cross_document_embedding_jobs_use_distinct_files(tmp_path,monkeypatch):
    from qaproject.semantic import encode_texts
    model = tmp_path/'model'
    model.mkdir()
    for name in ('model.onnx','tokenizer.json'):
        (model/name).write_bytes(b'isolated placeholder')
    barrier = threading.Barrier(2)
    paths = []
    def run(args,**kwargs):
        path = Path(args[-2])
        paths.append(path)
        barrier.wait(timeout=3)
        job = json.loads(path.read_text())
        job['vectors'] = {digest(['test',text]):(b'\x00'*2048).hex() for text in job['texts']}
        path.write_text(encode(job))
        return type('Result',(),{'returncode':0})()
    monkeypatch.setattr('qaproject.semantic.subprocess.run',run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda q:encode_texts(model,'test',tmp_path/'cache.sqlite3',[q]),['甲问题','乙问题']))
    assert list(results[0])==['甲问题'] and list(results[1])==['乙问题']
    assert len(set(paths))==2 and all(not p.exists() for p in paths)
