"""Windows journal compatibility without serializing file workers or leaking errors."""
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
import threading

import pytest

from qaproject.store import Store
from qaproject.library import LibraryIndex
from qaproject.semantic import init_cache, cache_connection
from qaproject.sqlite_support import journal_mode, sqlite_diagnostics
from test_bilingual import setup_pipeline, run


@pytest.mark.parametrize('windows,expected',[(False,'WAL'),(True,'DELETE')])
def test_platform_default(monkeypatch,windows,expected):
    monkeypatch.delenv('QA_SQLITE_JOURNAL_MODE',raising=False)
    monkeypatch.setattr('qaproject.sqlite_support.WINDOWS',windows)
    assert journal_mode()==expected


@pytest.mark.parametrize('mode',['WAL','DELETE'])
def test_all_databases_use_selected_mode_and_preserve_existing_data(tmp_path,monkeypatch,mode):
    monkeypatch.setenv('QA_SQLITE_JOURNAL_MODE','WAL')
    store=Store(tmp_path/'live')
    doc_id,_=store.import_file('frozen.txt',b'frozen source','live')
    original=store.document(doc_id)
    root=tmp_path/'library';root.mkdir()
    LibraryIndex(store,root)
    init_cache(store.root/'search.sqlite3')
    with cache_connection(store.root/'search.sqlite3') as db:
        db.execute("INSERT INTO qa_vectors VALUES('qa',?)",(b'frozen vector',))
    monkeypatch.setenv('QA_SQLITE_JOURNAL_MODE',mode)
    reopened=Store(store.root)
    LibraryIndex(reopened,root)
    init_cache(store.root/'search.sqlite3')
    for path in (store.db,store.root/'search.sqlite3',store.root/'library.sqlite3'):
        with cache_connection(path) as db:
            assert db.execute('PRAGMA journal_mode').fetchone()[0].upper()==mode
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert reopened.document(doc_id)==original
    with cache_connection(store.root/'search.sqlite3') as db:
        assert db.execute("SELECT vector FROM qa_vectors WHERE hash='qa'").fetchone()[0]==b'frozen vector'


def test_connections_do_not_change_journal_mode_and_keep_foreign_keys(tmp_path,monkeypatch):
    monkeypatch.setenv('QA_SQLITE_JOURNAL_MODE','DELETE')
    store=Store(tmp_path/'live')
    statements=[]
    connect=sqlite3.connect
    def traced(*args,**kwargs):
        db=connect(*args,**kwargs);db.set_trace_callback(statements.append);return db
    monkeypatch.setattr('qaproject.store.sqlite3.connect',traced)
    for _ in range(4):
        with store.connect() as db:
            assert db.execute('PRAGMA foreign_keys').fetchone()[0]==1
            db.execute('SELECT COUNT(*) FROM documents').fetchone()
    assert not any('journal_mode' in sql.lower() for sql in statements)


@pytest.mark.parametrize('mode',['WAL','DELETE'])
@pytest.mark.parametrize('threads',[2,4,8])
def test_concurrent_short_transactions_and_readers(tmp_path,monkeypatch,mode,threads):
    monkeypatch.setenv('QA_SQLITE_JOURNAL_MODE',mode)
    store=Store(tmp_path/'live')
    barrier=threading.Barrier(threads+1)
    def write(worker):
        barrier.wait(timeout=10)
        for n in range(50):
            with store.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                db.execute('INSERT INTO events VALUES(?,NULL,?,?,?)',(f'{worker}-{n}','isolated','{}',0))
    def read():
        barrier.wait(timeout=10)
        for _ in range(100):
            with store.connect() as db:
                db.execute('SELECT COUNT(*) FROM events').fetchone()
    with ThreadPoolExecutor(max_workers=threads+1) as pool:
        futures=[pool.submit(write,i) for i in range(threads)]+[pool.submit(read)]
        for future in futures:
            future.result(timeout=30)
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM events').fetchone()[0]==threads*50
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    with pytest.raises(RuntimeError):
        with store.connect() as db:
            db.execute("INSERT INTO events VALUES('rollback',NULL,'isolated','{}',0)")
            raise RuntimeError('roll back')
    with store.connect() as db:
        assert not db.execute("SELECT 1 FROM events WHERE id='rollback'").fetchone()


def test_invalid_mode_fails_before_work(tmp_path,monkeypatch):
    monkeypatch.setenv('QA_SQLITE_JOURNAL_MODE','DELETE; DROP TABLE documents')
    with pytest.raises(ValueError,match='QA_SQLITE_JOURNAL_MODE'):
        Store(tmp_path/'live')


def test_sqlite_error_diagnostics_are_safe():
    error=sqlite3.OperationalError('private/path secret SQL and original document text')
    error.sqlite_errorcode=sqlite3.SQLITE_READONLY_DBMOVED
    error.sqlite_errorname='SQLITE_READONLY_DBMOVED'
    assert sqlite_diagnostics(error)==dict(sqlite_errorcode=sqlite3.SQLITE_READONLY_DBMOVED,
                                         sqlite_errorname='SQLITE_READONLY_DBMOVED')
    assert not sqlite_diagnostics(ValueError('secret'))
    error.sqlite_errorname='SQLITE_READONLY private/path'
    assert sqlite_diagnostics(error)==dict(sqlite_errorcode=sqlite3.SQLITE_READONLY_DBMOVED)


def test_pipeline_records_extended_sqlite_code_without_raw_message(setup_pipeline):
    _,store,_,_,_,behavior,_=setup_pipeline
    def fail(*args):
        error=sqlite3.OperationalError('secret private/path SQL')
        error.sqlite_errorcode=sqlite3.SQLITE_READONLY_DBMOVED
        error.sqlite_errorname='SQLITE_READONLY_DBMOVED'
        raise error
    behavior['analysis']=fail
    doc=run(setup_pipeline)
    assert doc['status']=='failed' and 'secret' not in doc['error']
    assert 'SQLITE_READONLY_DBMOVED' in doc['error']
    with store.connect() as db:
        event=json.loads(db.execute("SELECT detail FROM events WHERE kind='run_failed'").fetchone()[0])
    assert event['sqlite_errorname']=='SQLITE_READONLY_DBMOVED'
    assert event['sqlite_errorcode']==sqlite3.SQLITE_READONLY_DBMOVED
    assert 'secret' not in json.dumps(event) and 'private/path' not in json.dumps(event)


def test_encoding_child_inherits_selected_mode_without_credentials(tmp_path,monkeypatch):
    import struct
    from pathlib import Path
    from types import SimpleNamespace
    from qaproject.semantic import encode_texts
    from qaproject.store import digest
    monkeypatch.setenv('QA_SQLITE_JOURNAL_MODE','DELETE')
    monkeypatch.setenv('API_KEY','never-pass-this-isolated-test-value')
    model=tmp_path/'model';model.mkdir()
    for name in ('model.onnx','tokenizer.json'):
        (model/name).write_bytes(b'isolated placeholder')
    def child(args,**kwargs):
        assert kwargs['env']['QA_SQLITE_JOURNAL_MODE']=='DELETE'
        assert 'API_KEY' not in kwargs['env']
        path=Path(args[-2]);job=json.loads(path.read_text(encoding='utf-8'))
        path.write_text(json.dumps({'vectors':{digest(['test',text]):struct.pack('<512f',1.,*([0.]*511)).hex()
                                               for text in job['texts']}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('qaproject.semantic.subprocess.run',child)
    assert list(encode_texts(model,'test',tmp_path/'cache.sqlite3',['问题']))==['问题']


def test_edit_failure_records_sqlite_code_preserving_original(setup_pipeline):
    from test_qa_summary import prepared, snapshot
    _,store,doc_id,worker,_,behavior,_=setup_pipeline
    client,target,versions=prepared(setup_pipeline)
    old=snapshot(store)
    assert client.post('/api/qas/'+target['id']+'/revisions',json={'versions':versions}).status_code==200
    def fail(*args):
        error=sqlite3.OperationalError('secret source path and SQL')
        error.sqlite_errorcode=sqlite3.SQLITE_READONLY
        error.sqlite_errorname='SQLITE_READONLY'
        raise error
    behavior['review']=fail
    worker.process(store.claim(worker.name,document_id=doc_id))
    assert snapshot(store)==old
    with store.connect() as db:
        detail=db.execute("SELECT detail FROM events WHERE kind='qa_revision_failed'").fetchone()[0]
    assert json.loads(detail)['sqlite_errorname']=='SQLITE_READONLY'
    assert 'secret' not in detail and 'source path' not in detail


@pytest.mark.parametrize('mode', ['WAL', 'DELETE'])
def test_merged_dataset_readers_keep_frozen_evidence_under_both_journals(tmp_path, monkeypatch, mode):
    from test_datasets import WOCE, netcdf_file
    from qaproject.parsers import extract
    from qaproject.settings import settings
    from qaproject.store import digest

    monkeypatch.setenv('QA_SQLITE_JOURNAL_MODE', mode)
    store = Store(tmp_path/'merged-data')
    nc = netcdf_file(tmp_path/'ocean.nc')
    for name, raw, expected in [
        ('station.csv', WOCE.encode('utf-8'), {'csv_schema', 'csv_stats', 'csv_rows'}),
        ('ocean.nc', nc.read_bytes(), {'netcdf_schema', 'netcdf_variable'}),
    ]:
        doc_id, _ = store.import_file(name, raw, 'live')
        units, pages = extract(store, store.document(doc_id), settings())
        assert pages == 0 and units
        kinds = {u['evidence_kind'] for u in units}
        if name.endswith('.csv'):
            assert expected <= kinds
            assert any('STNNBR = 15' in u['text'] for u in units)
        else:
            assert any(u['locator'] == 'nc-schema' for u in units)
            assert any(u.get('variable') == 'temperature' for u in units)
        for unit in units:
            assert unit['page'] == 0 and not unit['image']
            frozen = dict(unit)
            assert frozen.pop('hash') == digest(frozen)
        store.update_document(doc_id, units=units)
        assert store.document(doc_id)['units'] == units
    with store.connect() as db:
        assert db.execute('PRAGMA journal_mode').fetchone()[0].upper() == mode
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
