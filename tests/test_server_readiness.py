"""Optional directory intake, persistent queue, concurrent edits and recovery."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import threading
import time

from fastapi.testclient import TestClient
import pytest

from qaproject.app import create_app
from qaproject.library import LibraryIndex
from qaproject.persistence import RuntimeLease, backup_runtime, restore_runtime
from qaproject.pipeline import Worker, approve_document
from qaproject.settings import runtime_directory
from qaproject.store import Store, uid
from test_bilingual import setup_pipeline, run
from test_file_concurrency import provider
from test_qa_summary import prepared, snapshot


def client_for(app):
    client = TestClient(app)
    client.headers['x-qa-session'] = app.state.session
    return client


def test_unconfigured_directory_does_not_scan_or_change_upload_pipeline(tmp_path,monkeypatch):
    app = create_app(tmp_path/'data',workers=False)
    assert app.state.library.root is None
    assert not app.state.library.path.exists()
    client = client_for(app)
    listing = client.get('/api/library').json()
    assert listing['enabled'] is False and listing['items']==[]
    assert client.post('/api/library/refresh').json()['enabled'] is False
    calls,_ = provider(monkeypatch)
    result = client.post('/api/imports',files={'files':('a.txt','隔离观测值15。'.encode())}).json()[0]
    assert Worker(app.state.stores['live']).run_once()
    doc = app.state.stores['live'].document(result['id'])
    assert doc['bilingual_count']==5 and len(calls)==7
    assert not app.state.library.path.exists() and not app.state.library.thread.is_alive()
    assert client.get('/api/originals/'+doc['id']).status_code==200


def test_catalogue_pages_filter_names_without_reading_document_bodies(tmp_path,monkeypatch):
    root = tmp_path/'library'
    root.mkdir()
    for n in range(73):
        (root/f'{n:03d}.PDF').write_bytes(b'not parsed by the catalogue')
    (root/'100%观测.pdf').write_bytes(b'catalogue metadata only')
    (root/'ignored.exe').write_bytes(b'ignored')
    (root/'.private.pdf').write_bytes(b'ignored')
    (root/'escape.pdf').symlink_to(tmp_path/'outside.pdf')
    app = create_app(tmp_path/'data',workers=False,library_root=root)
    def forbidden(*a,**kw):
        raise AssertionError('Listing or scanning must not hash/read whole PDFs')
    monkeypatch.setattr('qaproject.library.file_hash',forbidden)
    app.state.library.scan()
    client = client_for(app)
    pages = [client.get('/api/library',params={'page':n,'page_size':30}).json() for n in range(1,4)]
    assert [len(p['items']) for p in pages]==[30,30,14]
    assert len({i['id'] for p in pages for i in p['items']})==74
    result = client.get('/api/library',params={'q':'100%'}).json()
    assert result['total']==1 and result['items'][0]['name']=='100%观测.pdf'
    assert client.get('/api/documents').json()==[]  # catalogue is not processed material
    assert client.get('/api/library?page_size=10001').status_code==422


def test_library_reference_processes_pdf_without_original_copy(tmp_path,monkeypatch):
    import pymupdf
    root = tmp_path/'library'
    root.mkdir()
    with pymupdf.open() as pdf:
        for n in range(2):
            pdf.new_page().insert_text((40,60),f'Synthetic area {n}: temperature, salinity and wind measurements are 15.')
        raw = pdf.tobytes()
    source = root/'paper.pdf'
    source.write_bytes(raw)
    app = create_app(tmp_path/'data',workers=False,library_root=root)
    app.state.library.scan()
    client = client_for(app)
    row = client.get('/api/library').json()['items'][0]
    result = client.post('/api/library/imports',json={'files':[{'id':row['id'],'version':row['version']}]}).json()[0]
    store = app.state.stores['live']
    assert store.document(result['id'])['path']==str(source)
    assert not list((store.root/'originals').iterdir())
    calls,_ = provider(monkeypatch)
    assert Worker(store).run_once()
    doc = store.document(result['id'])
    assert doc['bilingual_count']==5 and doc['pages']==2 and len(calls)==7, doc['error']
    assert all(c['row']['source'].startswith(str(source)+'|') for c in doc['candidates'])
    assert client.get('/api/originals/'+doc['id']).content==raw
    assert source.read_bytes()==raw
    assert approve_document(store,doc['id'])['committed_count']==10
    assert client.get('/api/library').json()['items'][0]['status']=='completed'
    before = snapshot(store)
    source.write_bytes(b'changed source')
    assert client.get('/api/originals/'+doc['id']).status_code==404
    assert snapshot(store)==before
    from qaproject.qa_summary import edit_detail, queue_revision
    target = doc['candidates'][0]
    detail = edit_detail(store,target['id'])
    versions = [dict(id=v['id'],expected_hash=v['row_hash'],**{k:v['row'][k] for k in ('input','thinking','output')}) for v in detail['versions']]
    versions[0]['thinking'] += '（核对）'
    queue_revision(store,target['id'],versions,detail['edit_version'])
    calls.clear()
    assert Worker(store).run_once()
    assert edit_detail(store,target['id'])['revision']['status']=='failed'
    assert not calls and snapshot(store)==before


def test_library_changes_and_path_attacks_do_not_create_tasks(tmp_path):
    root = tmp_path/'library'
    root.mkdir()
    path = root/'a.pdf'
    path.write_bytes(b'first file')
    app = create_app(tmp_path/'data',workers=False,library_root=root)
    app.state.library.scan()
    client = client_for(app)
    entry = client.get('/api/library').json()['items'][0]
    path.write_bytes(b'updated file')
    selection = dict(id=entry['id'],version=entry['version'])
    assert client.post('/api/library/imports',json={'files':[selection]}).json()[0]['error']
    assert client.post('/api/library/imports',json={'files':[dict(selection,path='/etc/passwd')]}).status_code==422
    app.state.library.scan()
    current = client.get('/api/library').json()['items'][0]
    path.unlink()
    outside = tmp_path/'outside.pdf'
    outside.write_bytes(b'outside')
    path.symlink_to(outside)
    assert client.post('/api/library/imports',json={'files':[dict(id=current['id'],version=current['version'])]}).json()[0]['error']
    assert client.get('/api/documents').json()==[]


@pytest.mark.parametrize('extension',['png','jpeg'])
def test_library_images_keep_real_source_locator_bytes_and_preview(tmp_path,monkeypatch,extension):
    from test_image_import import image_bytes
    root = tmp_path/'library'
    root.mkdir()
    raw = image_bytes(extension)
    source = root/('native.'+extension)
    source.write_bytes(raw)
    app = create_app(tmp_path/'data',workers=False,library_root=root)
    lib = app.state.library
    lib.scan()
    item = lib.list_files()['items'][0]
    result = lib.import_entry(item['id'],item['version'],25*1024*1024)
    provider(monkeypatch)
    store = app.state.stores['live']
    assert Worker(store).run_once()
    doc = store.document(result['id'])
    assert doc['status']=='waiting_review' and doc['bilingual_count']==2, doc['error']
    assert not list((store.root/'originals').iterdir())
    assert all(c['row']['source']==str(source)+'|image-0001|并发观测' for c in doc['candidates'])
    assert all(Path(c['row']['image']).read_bytes()==raw for c in doc['candidates'])
    assert client_for(app).get('/api/assets/'+doc['assets'][0]['id']).content==raw


def test_moved_original_is_reported_instead_of_reusing_a_broken_reference(tmp_path):
    root = tmp_path/'library'
    root.mkdir()
    path = root/'first.pdf'
    path.write_bytes(b'isolated unchanged bytes')
    app = create_app(tmp_path/'data',workers=False,library_root=root)
    lib = app.state.library
    lib.scan()
    item = lib.list_files()['items'][0]
    first = lib.import_entry(item['id'],item['version'],1024)
    # This is a metadata/history test, with workers disabled. Finish the
    # synthetic queue item before asking the file-priority index to scan again.
    with app.state.stores['live'].connect() as db:
        db.execute("UPDATE tasks SET status='succeeded' WHERE document_id=?",(first['id'],))
    path.rename(root/'moved.pdf')
    lib.scan()
    moved = lib.list_files()['items'][0]
    with pytest.raises(ValueError,match='历史资料'):
        lib.import_entry(moved['id'],moved['version'],1024)
    assert app.state.stores['live'].document(first['id'])['path']==str(path)


def test_reference_changed_before_parsing_stops_before_model(tmp_path,monkeypatch):
    root = tmp_path/'library'
    root.mkdir()
    path = root/'a.txt'
    path.write_text('原始观测值15。')
    app = create_app(tmp_path/'data',workers=False,library_root=root)
    lib = app.state.library
    lib.scan()
    entry = lib.list_files()['items'][0]
    result = lib.import_entry(entry['id'],entry['version'],1024*1024)
    path.write_text('已经替换的观测内容。')
    calls,_ = provider(monkeypatch)
    assert Worker(app.state.stores['live']).run_once()
    doc = app.state.stores['live'].document(result['id'])
    assert doc['status']=='failed' and '变更' in doc['error'] and not calls
    with app.state.stores['live'].connect() as db:
        error = json.loads(db.execute("SELECT detail FROM events WHERE kind='run_failed'").fetchone()[0])
        assert error['stage']=='source_validation'


def test_directory_scan_waits_for_file_work_without_touching_model(tmp_path):
    root = tmp_path/'library'
    root.mkdir()
    (root/'a.pdf').write_bytes(b'not parsed')
    store = Store(tmp_path/'data/live')
    busy = threading.Event()
    busy.set()
    lib = LibraryIndex(store,root,busy=busy.is_set)
    thread = threading.Thread(target=lib.scan)
    thread.start()
    try:
        time.sleep(.08)
        assert lib.list_files()['total']==0 and thread.is_alive()
        busy.clear()
        thread.join(timeout=2)
        assert not thread.is_alive() and lib.list_files()['total']==1
    finally:
        lib.close()
        thread.join(timeout=2)


def test_missing_directory_leaves_upload_available(tmp_path):
    app = create_app(tmp_path/'data',workers=False,library_root=tmp_path/'not-mounted')
    app.state.library.scan()
    client = client_for(app)
    assert client.get('/api/library').json()['error']
    assert client.post('/api/imports',files={'files':('a.txt',b'local upload remains available')}).json()[0]['created']


def test_closing_a_browser_client_does_not_cancel_background_queue(tmp_path,monkeypatch):
    entered, release = threading.Event(),threading.Event()
    def enter(doc,kind):
        if kind=='analysis':
            entered.set()
            assert release.wait(4)
    provider(monkeypatch,enter)
    app = create_app(tmp_path/'data')
    with TestClient(app) as service:
        browser = client_for(app)
        result = browser.post('/api/imports',files=[('files',(f'{n}.txt',f'合成海域{n}指标为15。'.encode())) for n in range(3)]).json()
        assert entered.wait(3)
        browser.close()
        queue = service.get('/api/queue').json()
        assert queue['running'] in (1,2) and queue['queued']>=1
        release.set()
        until = time.monotonic()+8
        while time.monotonic()<until:
            docs = service.get('/api/documents?summary_only=true').json()
            if all(d['status']=='waiting_review' for d in docs):
                break
            time.sleep(.03)
        assert len(docs)==3 and all(d['bilingual_count']==5 for d in docs)
        assert all(app.state.stores['live'].document(d['id'])['qa_count']==0 for d in result)


def test_exclusive_restart_recovers_unexpired_tasks_immediately(tmp_path,monkeypatch):
    root = tmp_path/'data'
    store = Store(root/'live')
    doc,_ = store.import_file('a.txt',b'isolated restart evidence','live')
    task = store.claim('old-process')
    store.update_document(doc,status='analyzing')
    monkeypatch.setattr(Worker,'start',lambda self:None)
    app = create_app(root)
    with TestClient(app):
        with store.connect() as db:
            current = db.execute('SELECT * FROM tasks WHERE id=?',(task['id'],)).fetchone()
            assert current['status']=='queued' and current['lease_until']==0 and current['worker'] is None
            assert db.execute("SELECT COUNT(*) FROM events WHERE kind='task_recovered'").fetchone()[0]==1
        assert store.document(doc)['status']=='queued'


def test_restart_preserves_quality_task_kind(tmp_path):
    store = Store(tmp_path/'data/live')
    doc,_ = store.import_file('a.txt',b'isolated quality task','live')
    with store.connect() as db:
        db.execute("UPDATE tasks SET stage='quality_queued' WHERE document_id=?",(doc,))
    task = store.claim('old-process')
    store.task_update(task['id'],'old-process','review')
    with RuntimeLease(store.root.parent):
        assert store.recover_abandoned()==1
    assert store.claim('recovered')['stage']=='quality_queued'


def test_second_application_cannot_start_on_the_same_persistent_directory(tmp_path):
    root = tmp_path/'data'
    a = create_app(root)
    with TestClient(a):
        b = create_app(root)
        with pytest.raises(ValueError,match='正在使用'):
            with TestClient(b):
                pytest.fail('the second scheduler must never start')
        assert not any(w.thread.is_alive() for w in b.state.runners)


def test_configured_data_directory_is_independent_of_launch_directory(tmp_path,monkeypatch):
    root = tmp_path/'persistent'
    monkeypatch.setenv('QA_RUNTIME_DIR',str(root))
    assert runtime_directory()==root
    assert create_app(workers=False).state.stores['live'].root==root/'live'
    assert runtime_directory(tmp_path/'explicit')==tmp_path/'explicit'


def test_stale_editor_conflicts_for_pending_and_completed_edits(setup_pipeline):
    app,store,doc,worker,calls,*_ = setup_pipeline
    client,target,versions = prepared(setup_pipeline)
    detail = client.get('/api/qas/'+target['id']).json()
    body = dict(versions=versions,expected_edit_version=detail['edit_version'])
    old = snapshot(store)
    assert client.post('/api/qas/'+target['id']+'/revisions',json=body).status_code==200
    count = len(calls)
    conflict = client.post('/api/qas/'+target['id']+'/revisions',json=body)
    assert conflict.status_code==409 and conflict.json()['code']=='edit_conflict'
    assert snapshot(store)==old and len(calls)==count
    worker.process(store.claim(worker.name,doc))
    new = snapshot(store)
    assert new!=old
    assert client.post('/api/qas/'+target['id']+'/revisions',json=body).status_code==409
    assert snapshot(store)==new


def test_two_editors_submit_one_draft_without_lost_updates(setup_pipeline):
    app,store,doc,worker,calls,*_ = setup_pipeline
    client,target,versions = prepared(setup_pipeline)
    detail = client.get('/api/qas/'+target['id']).json()
    body = dict(versions=versions,expected_edit_version=detail['edit_version'])
    gate = threading.Barrier(2)
    def submit(n):
        c = client_for(app)
        changed = [dict(v) for v in versions]
        changed[0]['thinking']+=str(n)
        gate.wait(timeout=3)
        return c.post('/api/qas/'+target['id']+'/revisions',json=dict(body,versions=changed)).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit,range(2)))==[200,409]
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM qa_revisions').fetchone()[0]==1


def test_backup_restore_preserves_rows_hashes_originals_and_revision_history(setup_pipeline,tmp_path):
    app,store,doc,worker,*_ = setup_pipeline
    client,target,versions = prepared(setup_pipeline)
    assert client.post('/api/qas/'+target['id']+'/revisions',json={'versions':versions}).status_code==200
    worker.process(store.claim(worker.name,doc))
    expected = snapshot(store)
    export = client.get('/api/export').content
    root = store.root.parent
    (root/'.env.local').write_text('not-a-real-key fixture; must not be included')
    saved = tmp_path/'snapshot'
    result = backup_runtime(root,saved)
    assert result['files']>0 and not list(saved.rglob('.env.local'))
    manifest = json.loads((saved/'manifest.json').read_text())
    assert all(not i['path'].endswith(('-wal','-shm')) for i in manifest['files'])
    shutil.move(str(store.root),str(tmp_path/'retained-old-live'))
    assert restore_runtime(saved,root)['restored']==result['files']
    restored = Store(root/'live')
    assert snapshot(restored)==expected and restored.document(doc)['qa_count']==10
    assert restored.original(doc)[0].is_file()
    with restored.connect() as db:
        assert db.execute('SELECT status FROM qa_revisions').fetchone()[0]=='applied'
    assert client.get('/api/export').content==export


def test_backup_restore_refuses_active_incomplete_or_overwriting_targets(tmp_path):
    root = tmp_path/'data'
    store = Store(root/'live')
    store.import_file('a.txt',b'backup data','live')
    with RuntimeLease(root):
        with pytest.raises(ValueError,match='正在使用'):
            backup_runtime(root,tmp_path/'blocked')
    saved = tmp_path/'snapshot'
    backup_runtime(root,saved)
    with pytest.raises(ValueError,match='已存在'):
        restore_runtime(saved,root)
    manifest = json.loads((saved/'manifest.json').read_text())
    target = saved/manifest['files'][0]['path']
    target.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='不完整'):
        restore_runtime(saved,root)
    assert store.documents()[0]['name']=='a.txt'


def test_allowlisted_internal_host_keeps_session_and_origin_checks(tmp_path,monkeypatch):
    from qaproject.settings import settings
    cfg = dict(settings(),access={'allowed_hosts':['qa.internal'],'allowed_origins':['https://qa.internal']})
    monkeypatch.setattr('qaproject.app.settings',lambda:cfg)
    app = create_app(tmp_path/'data',workers=False)
    client = TestClient(app,base_url='https://qa.internal')
    assert client.get('/api/bootstrap').status_code==200
    assert client.post('/api/library/refresh').status_code==403
    client.headers['x-qa-session']=app.state.session
    assert client.post('/api/library/refresh',headers={'Origin':'https://evil.invalid'}).status_code==403
    assert client.post('/api/library/refresh',headers={'Origin':'https://qa.internal'}).status_code==200
