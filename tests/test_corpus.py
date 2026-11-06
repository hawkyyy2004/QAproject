"""Small local fixtures; no model calls or production data."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import threading
import zipfile

import pytest

from qaproject import corpus
from qaproject.execution import ExecutionLimits
from qaproject.store import Store, digest, encode


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path/'runtime')


@pytest.fixture
def worker(store):
    class Worker:
        name = 'corpus-test'
        stop = threading.Event()
        limits = ExecutionLimits()
    w = Worker(); w.store = store
    yield w
    w.limits.close()


def imported(store, name='paper.txt', content=b'Real ocean evidence.'):
    return store.import_file(name, content, 'live', task_stage='corpus')[0]


def run(worker, doc_id):
    task = worker.store.claim(worker.name, document_id=doc_id)
    assert task
    return corpus.process(worker, task)


def test_contract_schema_direct_upload_and_idempotence(store, worker, monkeypatch):
    corpus.init_schema(store); corpus.init_schema(store)
    doc = imported(store)
    before = store.document(doc)
    calls = []
    original = corpus.extract
    def extract(*args, **kwargs):
        calls.append(args[2]); return original(*args, **kwargs)
    monkeypatch.setattr(corpus, 'extract', extract)
    result = run(worker, doc)
    assert result and result['status'] == 'ready'
    version = corpus.verify(store, result['id'])
    assert version['units'][0]['text'] == 'Real ocean evidence.'
    assert version['kind'] == 'paper'
    assert calls[0]['asset_namespace'].startswith('corpus-')
    assert store.document(doc) == before
    assert corpus.enqueue(store, doc)['status'] == 'completed'
    assert store.claim(worker.name) is None
    assert corpus.prepared_units(store, doc)[0] == version['units']
    listing = corpus.listing(store)
    assert listing['total'] == 1
    assert not {'units','units_json','manifest_json','markdown'} & listing['items'][0].keys()
    assert corpus.jobs(store)['items'][0]['version_id'] == result['id']
    with store.connect() as db:
        with pytest.raises(sqlite3.IntegrityError, match='immutable'):
            db.execute("UPDATE corpus_versions SET title='changed'")


def test_web_metadata_versioned_without_touching_old_evidence(store, worker):
    doc = imported(store, 'capture.txt')
    corpus.register_source(store, doc, 'web', 'First title', 'https://example.org/a', 'site')
    corpus.enqueue(store, doc, kind=None, metadata={'extract_knowledge':True})
    first = run(worker, doc)
    old = corpus.verify(store, first['id'])
    assert old['units'][0]['page_urls'] == 'https://example.org/a'
    assert corpus.jobs(store)['items'][0]['metadata']['extract_knowledge']
    corpus.register_source(store, doc, 'web', 'Second title', 'https://example.org/b')
    corpus.enqueue(store, doc)
    second = run(worker, doc)
    assert first['id'] != second['id']
    assert corpus.verify(store, first['id']) == old
    assert corpus.verify(store, second['id'])['units'][0]['page_titles'] == 'Second title'
    assert corpus.listing(store, kind='web', page_size=1)['pages'] == 2
    assert corpus.listing(store, q='Second')['total'] == 1


def test_source_missing_history_readable_but_verify_export_fail(store, worker):
    doc = imported(store); result = run(worker, doc)
    path = Path(store.document(doc)['path']); path.unlink()
    assert not corpus.detail(store, result['id'])['source_valid']
    with pytest.raises(ValueError): corpus.verify(store, result['id'])
    with pytest.raises(ValueError): corpus.export_bundle(store, result['id'])
    with pytest.raises(ValueError): corpus.prepared_units(store, doc)


def test_source_changed_rejects_initial_processing(store, worker):
    doc = imported(store)
    Path(store.document(doc)['path']).write_text('Replacement', encoding='utf-8')
    assert run(worker, doc) is None
    assert corpus.jobs(store)['items'][0]['status'] == 'failed'
    assert corpus.listing(store)['total'] == 0


def test_zip_actual_relative_paths_and_hashes_no_persistent_original_copy(store, worker):
    import pymupdf
    pdf = pymupdf.open(); page = pdf.new_page(); page.insert_text((50,50), 'Real ocean evidence.')
    raw = pdf.tobytes(); pdf.close()
    doc = imported(store, 'paper.pdf', raw); result = run(worker, doc)
    version_dir = store.root/'corpus/versions'/result['id']
    assert not (version_dir/'originals').exists()
    data = corpus.export_bundle(store, result['id'])
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        row = next(csv.DictReader(io.StringIO(archive.read('papers.csv').decode('utf-8-sig'))))
        assert list(row) == corpus.PAPER_FIELDS
        assert archive.read(row['pdf_path']) == raw
        assert archive.read(row['md_path']).decode().startswith('# paper.pdf')
        manifest = json.loads(archive.read('manifest.json'))
        for path, expected in manifest['files'].items():
            assert not Path(path).is_absolute() and '..' not in Path(path).parts
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected
    assert not (version_dir/'originals').exists()


def test_archived_file_tamper_is_rejected(store, worker):
    doc = imported(store); result = run(worker, doc)
    (store.root/'corpus/versions'/result['id']/'document.md').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='篡改'): corpus.verify(store, result['id'])
    with pytest.raises(ValueError): corpus.export_bundle(store, result['id'])


def frozen_image(store, doc):
    path = store.root/'assets/images'/doc/'image.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'full original image bytes')
    asset = store.add_asset(doc, path, 1, 'original')
    unit = dict(id=doc+':p1:s1', text='Scan page: use actual visual evidence.', page=1, seg=1,
                image=str(path), asset_id=asset, locator='page-01-seg-01')
    unit['hash'] = digest(unit)
    store.update_document(doc, units=[unit], pages=1)
    return unit


def test_frozen_image_copied_isolated_and_old_qa_changes_cannot_corrupt_version(store, worker, monkeypatch):
    doc = imported(store); raw = frozen_image(store, doc)
    monkeypatch.setattr(corpus, 'extract', lambda *a,**kw: pytest.fail('must reuse frozen evidence'))
    first = run(worker, doc)
    version = corpus.verify(store, first['id']); unit = version['units'][0]
    assert unit['image'] != raw['image'] and unit['asset_id'] != raw['asset_id']
    assert unit['hash'] != raw['hash']
    assert Path(unit['image']).read_bytes() == Path(raw['image']).read_bytes()
    Path(raw['image']).write_bytes(b'QA overwrite')
    assert corpus.verify(store, first['id'])['units'] == [unit]
    assert store.document(doc)['units'] == [raw]
    with zipfile.ZipFile(io.BytesIO(corpus.export_bundle(store, first['id']))) as archive:
        assert archive.read('images/0001.png') == b'full original image bytes'
        assert 'images/0001.png' in archive.read('document.md').decode()
    Path(unit['image']).write_bytes(b'bad')
    with pytest.raises(ValueError): corpus.verify(store, first['id'])


def test_reuse_rejects_asset_hash_or_unit_hash_tampering(store, worker):
    doc = imported(store); raw = frozen_image(store, doc)
    Path(raw['image']).write_bytes(b'bad')
    assert run(worker, doc) is None
    assert corpus.listing(store)['total'] == 0
    corpus.enqueue(store, doc)
    raw['text'] = 'edited without rehash'; store.update_document(doc, units=[raw])
    assert run(worker, doc) is None


def test_expired_lease_cannot_parse_publish_or_change_job(store, worker, monkeypatch):
    doc = imported(store); corpus.enqueue(store, doc)
    task = store.claim(worker.name, document_id=doc)
    with store.connect() as db: db.execute('UPDATE tasks SET lease_until=0 WHERE id=?', (task['id'],))
    extract = corpus.extract
    monkeypatch.setattr(corpus, 'extract', lambda *a,**kw: pytest.fail('expired lease must not parse'))
    assert corpus.process(worker, task) is None
    assert corpus.listing(store)['total'] == 0
    assert corpus.jobs(store)['items'][0]['status'] == 'queued'
    monkeypatch.setattr(corpus, 'extract', extract)
    assert run(worker, doc)


def test_parse_checkpoint_resumes_after_archive_failure(store, worker, monkeypatch):
    doc = imported(store); publish = corpus._publish
    monkeypatch.setattr(corpus, '_publish', lambda *a,**kw: (_ for _ in ()).throw(OSError('simulated')))
    assert run(worker, doc) is None
    monkeypatch.setattr(corpus, '_publish', publish)
    monkeypatch.setattr(corpus, 'extract', lambda *a,**kw: pytest.fail('must resume frozen units'))
    corpus.enqueue(store, doc)
    assert run(worker, doc)
    assert corpus.listing(store)['total'] == 1


def test_knowledge_frozen_units_reused(store, worker, monkeypatch):
    doc = imported(store); raw = dict(id='frozen',text='Observed salinity.',page=0,seg=1,image='',asset_id='',locator='seg-01')
    raw['hash'] = digest(raw)
    with store.connect() as db:
        db.execute("INSERT INTO knowledge_jobs(document_id,task_id,status,units,updated) VALUES(?,?,'completed',?,0)", (doc,'history',encode([raw])))
    monkeypatch.setattr(corpus, 'extract', lambda *a,**kw: pytest.fail('must reuse knowledge frozen units'))
    assert corpus.verify(store, run(worker, doc)['id'])['units'][0]['text'] == raw['text']
    assert store.document(doc)['units'] == []


def test_cleaning_preserves_raw_and_span_mapping():
    units = []
    for n in range(1,4):
        unit = dict(id=str(n),text=f'Page {n} of 3\nA real fact {n}.\n',page=n,seg=1,image='',asset_id='',locator=f'page-{n}-seg-1')
        unit['hash'] = digest(unit); units.append(unit)
    cleaned, mapping = corpus._clean(units)
    for raw, clean, evidence in zip(units,cleaned,mapping):
        assert clean['text'].startswith('A real fact')
        assert clean['text'] == ''.join(raw['text'][a:b] for a,b in evidence['retained_spans'])
        assert raw['text'].startswith('Page')
    short, _ = corpus._clean(units[:2])
    assert short[0]['text'] == units[0]['text']


def test_reference_original_is_read_only_and_never_copied(store, worker, tmp_path):
    from qaproject.source_files import signature
    root = tmp_path/'library'; root.mkdir(); source = root/'server.txt'
    source.write_bytes(b'Server original evidence.'); source.chmod(0o444)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    doc, _ = store.import_reference(source, original_hash, root, signature(source), task_stage='corpus')
    result = run(worker, doc)
    assert result
    assert not (store.root/'originals'/doc).exists()
    assert source.read_bytes() == b'Server original evidence.'
    assert corpus.detail(store, result['id'])['source_path'] == str(source)
    assert corpus.export_bundle(store, result['id'])
    assert not (store.root/'originals'/doc).exists()


def test_safe_path_rejects_traversal_and_symlink_escape(tmp_path):
    for path in ('../secret', '/secret', 'C:/secret', 'a\\b'):
        with pytest.raises(ValueError): corpus._safe(tmp_path,path)
    root = tmp_path/'archive'; root.mkdir()
    (root/'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError): corpus._safe(root,'escape/file')


def test_enqueue_preserves_chain_requests_and_partial_source_metadata(store, worker):
    doc = imported(store)
    corpus.register_source(store, doc, 'web', 'Title', 'https://example.org/real', 'original')
    corpus.register_source(store, doc, None, source_label='revised')
    assert corpus._source(store, store.document(doc)) == dict(kind='web', title='Title',
                    source_url='https://example.org/real', source_label='revised')
    corpus.enqueue(store, doc, generate_after=True, metadata={'extract_knowledge':True})
    job = corpus.enqueue(store, doc, generate_after=False)
    assert job['generate_after'] and job['metadata']['extract_knowledge']
    result = run(worker, doc)
    with store.connect() as db:
        db.execute('UPDATE corpus_jobs SET generate_after=0 WHERE document_id=?', (doc,))
    job = corpus.enqueue(store, doc, generate_after=True)
    assert job['generate_after'] and job['status'] == 'completed'
    assert job['version_id'] == result['id']
    assert store.claim(worker.name) is None


def test_lease_lost_during_parse_cannot_commit(store, worker, monkeypatch):
    doc = imported(store)
    extract = corpus.extract
    def stolen(*args,**kw):
        result = extract(*args,**kw)
        with store.connect() as db:
            db.execute("UPDATE tasks SET worker='another-worker' WHERE document_id=?", (doc,))
        return result
    monkeypatch.setattr(corpus, 'extract', stolen)
    assert run(worker, doc) is None
    assert corpus.listing(store)['total'] == 0
    with store.connect() as db:
        assert db.execute('SELECT worker FROM tasks WHERE document_id=?', (doc,)).fetchone()[0] == 'another-worker'


def test_version_database_binding_rejects_tamper(store, worker):
    doc = imported(store); result = run(worker, doc)
    with store.connect() as db:
        db.execute('DROP TRIGGER corpus_versions_no_update')
        db.execute("UPDATE corpus_versions SET units_json='[]' WHERE id=?", (result['id'],))
    with pytest.raises(ValueError, match='hash'): corpus.verify(store, result['id'])


def test_knowledge_reuse_preserves_known_document_page_count(store, worker):
    doc = imported(store)
    unit = dict(id='last',text='Recorded on an actual PDF page.',page=7,seg=1,image='',asset_id='',locator='page-07-seg-01')
    unit['hash'] = digest(unit)
    store.update_document(doc, pages=9)
    with store.connect() as db:
        db.execute("INSERT INTO knowledge_jobs(document_id,task_id,status,units,updated) VALUES(?,?,'completed',?,0)", (doc,'history',encode([unit])))
    result = run(worker, doc)
    assert corpus.prepared_units(store, doc)[1] == 9
    assert corpus.detail(store, result['id'])['units'][0]['page'] == 7


def test_direct_image_upload_uses_shared_parser_and_infers_kind(store, worker):
    import pymupdf
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0,0,3,3), False)
    pix.clear_with(123); raw = pix.tobytes('png')
    doc = imported(store, 'real.png', raw)
    result = run(worker, doc)
    version = corpus.verify(store, result['id'])
    assert version['kind'] == 'image'
    image = next(u for u in version['units'] if u['image'])
    assert Path(image['image']).read_bytes() == raw
    assert image['page'] == 0 and image['locator'] == 'image-0001'
    with zipfile.ZipFile(io.BytesIO(corpus.export_bundle(store, result['id']))) as archive:
        unit = next(u for u in json.loads(archive.read('units.json')) if u['image'])
        assert archive.read(unit['image']) == raw
        manifest = json.loads(archive.read('manifest.json'))
        assert 'units.json' in manifest['files']


@pytest.mark.parametrize('kind', ['web','image','data','paper'])
def test_non_pdf_exports_do_not_claim_pdf_index(store, worker, kind):
    doc = imported(store)
    corpus.register_source(store, doc, kind, source_url='https://example.org/a' if kind=='web' else '')
    result = run(worker, doc)
    with zipfile.ZipFile(io.BytesIO(corpus.export_bundle(store, result['id']))) as archive:
        assert 'papers.csv' not in archive.namelist()
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['index'] is None
        if kind == 'web':
            assert list(csv.DictReader(io.StringIO(archive.read('Web/wiki_updated_full.csv').decode('utf-8-sig')))) == []
            assert manifest['warnings']


def test_web_export_only_version_bound_reviewed_facts_with_clean_fulltext(store, worker):
    doc = imported(store, content=b'First fact.\n\nSecond fact.')
    corpus.register_source(store, doc, 'web', 'Ocean page', 'https://example.org/ocean')
    result = run(worker, doc); version = corpus.verify(store, result['id'])
    for i,unit in enumerate(version['units']):
        row = dict(entity_name='Salinity', entity_type='Property', entity_description='Fact '+str(i),
                   page_urls=version['source_url'], page_titles=version['title'], full_text=unit['text'], full_text_length=len(unit['text']))
        review = dict(row_hash=digest(row), evidence_hash=unit['hash'], **{k:True for k in
                      ('supported','scope_correct','type_correct','topic_correct','language_correct')})
        with store.connect() as db:
            db.execute('INSERT INTO knowledge_entries VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                       (str(i),doc,unit['id'],unit['hash'],encode(row),digest(row),'Ocean',unit['text'],'approved',encode(review),i))
    with store.connect() as db:
        db.execute("INSERT INTO knowledge_jobs(document_id,task_id,status,updated) VALUES(?,?,'completed',0)", (doc,'knowledge'))
    with zipfile.ZipFile(io.BytesIO(corpus.export_bundle(store,result['id']))) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read('Web/wiki_updated_full.csv').decode('utf-8-sig'))))
        assert len(rows) == 1 and list(rows[0]) == corpus.WEB_FIELDS
        assert rows[0]['entity_description'] == 'Fact 0\n\nFact 1'
        assert rows[0]['full_text'] == 'First fact.\n\nSecond fact.'
        assert int(rows[0]['full_text_length']) == len(rows[0]['full_text'])
        assert rows[0]['page_urls'] == 'https://example.org/ocean'
        assert not json.loads(archive.read('manifest.json'))['warnings']
    with store.connect() as db:
        db.execute("UPDATE knowledge_entries SET row_hash='tampered' WHERE id='0'")
    with pytest.raises(ValueError, match='审核绑定'): corpus.export_bundle(store,result['id'])
