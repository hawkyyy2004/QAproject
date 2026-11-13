"""Original MinerU figures feed Corpus, QA and knowledge without page-shot regression."""
import io
import json
from pathlib import Path
import zipfile

import pymupdf
import pytest
from fastapi.testclient import TestClient

from qaproject import corpus, corpus_processing, knowledge, layout
from qaproject.app import create_app
from qaproject.execution import ExecutionLimits
from qaproject.parsers import extract
from qaproject.pipeline import Worker
from qaproject.settings import settings
from qaproject.store import Store
from test_layout_manifest import graphics_pdf, manifest_for, loader


def figures(store, doc):
    return manifest_for(store, doc, [{'page':1,'blocks':[
        {'kind':'text','text':'Ocean observations refer to Figure 1 and Figure 2.'},
        {'kind':'figure','bbox':[.1,.2,.9,.4],'path':'crops/page-01-fig-01.png','caption':'Figure 1 salinity'},
        {'kind':'figure','bbox':[.1,.5,.9,.7],'path':'crops/page-01-fig-02.png','caption':'Figure 2 temperature'}]}])


def archive(worker, store):
    assert worker.run_once()
    jobs=corpus.jobs(store)['items']
    assert jobs[0]['status']=='completed',jobs[0]['error']
    return corpus.verify(store,jobs[0]['version_id'])


def test_two_original_figures_shared_with_qa_knowledge_preview_and_export(tmp_path,monkeypatch):
    store=Store(tmp_path/'data')
    doc,_=store.import_file('two-figures.pdf',graphics_pdf(),'demo',task_stage='corpus')
    target=figures(store,doc)
    corpus.enqueue(store,doc)
    with store.connect() as db:
        parent=db.execute('SELECT task_id FROM corpus_jobs WHERE document_id=?',(doc,)).fetchone()[0]
    store.ensure_qa_task(doc,after_task_id=parent)
    monkeypatch.setattr(layout,'prepare',lambda *a:pytest.fail('Frozen layout must be reused'))
    monkeypatch.setattr(pymupdf.Page,'get_pixmap',lambda *a,**kw:pytest.fail('No full-page rendering'))
    worker=Worker(store,limits=ExecutionLimits())
    try:
        version=archive(worker,store)
        images=[u for u in version['units'] if u['image']]
        assert len(images)==2 and all(u['evidence_kind']=='pdf_figure' for u in images)
        assert [u['locator'] for u in images]==['page-01-fig-01','page-01-fig-02']
        assert all(u['layout_engine']=='mineru' and u['bbox'] for u in images)
        assert version['manifest']['image_count']==2
        assert '完整原图' not in version['markdown']
        assert version['markdown'].count('![page-01-fig-')==2
        monkeypatch.setattr('qaproject.pipeline.extract',lambda *a,**kw:pytest.fail('QA must not parse again'))
        assert worker.run_once()
        assert store.document(doc)['status']=='parsed_demo'
        assert store.document(doc)['units']==version['units']
        with store.connect() as db:
            assert db.execute('SELECT version_id FROM qa_corpus_inputs WHERE document_id=?',(doc,)).fetchone()[0]==version['id']
        monkeypatch.setattr(knowledge,'extract',lambda *a,**kw:pytest.fail('Knowledge must not parse again'))
        assert knowledge.prepare(store,store.document(doc),settings(),lambda:None)[0]==version['units']
        app=create_app(tmp_path/'api',workers=False);app.state.stores['live']=store
        client=TestClient(app)
        for image in images:
            preview=client.get('/api/assets/'+image['asset_id'])
            assert preview.status_code==200 and preview.content==Path(image['image']).read_bytes()
        # Frozen assets are independent of the disposable layout cache.
        for image in images:
            source=target/'crops'/('page-01-fig-01.png' if image['locator'].endswith('01') else 'page-01-fig-02.png')
            assert Path(image['image']).read_bytes()==source.read_bytes()
        with zipfile.ZipFile(io.BytesIO(corpus.export_official_bundle(store,version['id']))) as package:
            paths=[p for p in package.namelist() if '/images/' in p and p.endswith('.png')]
            assert len(paths)==2
            assert all(package.read(p)==Path(images[n]['image']).read_bytes() for n,p in enumerate(paths))
    finally:worker.limits.close()


def test_archive_retry_reuses_figures_and_structure(tmp_path,monkeypatch):
    store=Store(tmp_path/'data');doc,_=store.import_file('figures.pdf',graphics_pdf(),'live',task_stage='corpus')
    figures(store,doc)
    original=corpus._publish
    monkeypatch.setattr(corpus,'_publish',lambda *a,**kw:(_ for _ in ()).throw(OSError('isolated archive error')))
    worker=Worker(store,limits=ExecutionLimits())
    try:
        assert worker.run_once() and corpus.jobs(store)['items'][0]['status']=='failed'
        monkeypatch.setattr(corpus,'_publish',original)
        monkeypatch.setattr(corpus,'extract',lambda *a,**kw:pytest.fail('No second evidence preparation'))
        monkeypatch.setattr(corpus_processing,'mineru_structure',lambda *a,**kw:pytest.fail('Reuse completed structure'))
        corpus.enqueue(store,doc)
        version=archive(worker,store)
        assert version['manifest']['image_count']==2
    finally:worker.limits.close()


@pytest.mark.parametrize('problem',['missing-crop','missing-page'])
def test_incomplete_mineru_evidence_fails_without_page_shot_or_downstream_release(tmp_path,monkeypatch,problem):
    store=Store(tmp_path/'data');doc,_=store.import_file('figures.pdf',graphics_pdf(),'live',task_stage='corpus')
    target=figures(store,doc)
    if problem=='missing-crop':(target/'crops/page-01-fig-02.png').unlink()
    else:
        raw=json.loads((target/'manifest.json').read_text())
        raw['pdf']['pages']=2
        (target/'manifest.json').write_text(json.dumps(raw))
    parent=corpus.enqueue(store,doc)['task_id'];store.ensure_qa_task(doc,after_task_id=parent)
    monkeypatch.setattr(pymupdf.Page,'get_pixmap',lambda *a,**kw:pytest.fail('Must fail, not render a page'))
    worker=Worker(store,limits=ExecutionLimits())
    try:
        assert worker.run_once()
        assert corpus.jobs(store)['items'][0]['status']=='failed'
        assert corpus.listing(store)['total']==0
        assert store.claim('downstream') is None
        assert not store.document(doc)['units']
    finally:worker.limits.close()


def test_new_version_repairs_requested_legacy_page_archive_without_touching_qa(tmp_path,monkeypatch):
    store=Store(tmp_path/'data');doc,_=store.import_file('figures.pdf',graphics_pdf(),'live',task_stage='corpus')
    worker=Worker(store,limits=ExecutionLimits())
    try:
        old=archive(worker,store)  # No MinerU in this isolated fixture: legacy native archive.
        assert any(u['evidence_kind']=='pdf_page' for u in old['units'])
        store.update_document(doc,units=old['units'],pages=1)
        before=store.document(doc)
        figures(store,doc)
        corpus.enqueue(store,doc)
        new=archive(worker,store)
        assert new['id']!=old['id'] and new['manifest']['image_count']==2
        assert corpus.verify(store,old['id'])==old
        assert store.document(doc)['units']==before['units']
        assert all(Path(u['image']).is_file() for u in old['units'] if u['image'])
    finally:worker.limits.close()


def test_unpublished_legacy_checkpoint_rebuilt_from_cached_mineru_without_screenshot(tmp_path,monkeypatch):
    store=Store(tmp_path/'data');doc,_=store.import_file('figures.pdf',graphics_pdf(),'live',task_stage='corpus')
    old_units,pages=extract(store,store.document(doc),dict(settings(),pdf_layout='off'))
    from qaproject.store import encode
    corpus.enqueue(store,doc)
    with store.connect() as db:
        db.execute('UPDATE corpus_jobs SET units_json=?,pages=?,metadata_json=? WHERE document_id=?',
            (encode(old_units),pages,encode({'processing':{'visual_preparation_complete':True,'parser_selection':{'resolved':'mineru'}}}),doc))
    figures(store,doc)
    monkeypatch.setattr(pymupdf.Page,'get_pixmap',lambda *a,**kw:pytest.fail('No new page shot'))
    worker=Worker(store,limits=ExecutionLimits())
    try:
        new=archive(worker,store)
        assert new['manifest']['image_count']==2
        with store.connect() as db:
            metadata=json.loads(db.execute('SELECT metadata_json FROM corpus_jobs WHERE document_id=?',(doc,)).fetchone()[0])
        assert metadata['superseded_preparations'][0]['units']==old_units
        assert all(Path(u['image']).is_file() for u in old_units if u['image'])
    finally:worker.limits.close()


def test_original_layout_builder_reuses_completed_raw_parse_for_figures(tmp_path,monkeypatch):
    module=loader();path=tmp_path/'figures.pdf';path.write_bytes(graphics_pdf())
    root=tmp_path/'layout';target=root/module.sha256_file(path);target.mkdir(parents=True)
    raw={'pages':[{'page_idx':0,'blocks':[
        {'type':'text','bbox':[.1,.05,.9,.1],'content':'Ocean observations.'},
        {'type':'image','bbox':[.1,.2,.9,.5],'content':[
            {'type':'image_caption','content':'Figure 1 ocean observations'}]}]}]}
    (target/'mineru.json').write_text(json.dumps(raw))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:pytest.fail('Completed MinerU parse must not run twice'))
    from types import SimpleNamespace
    result=module.process(path,SimpleNamespace(layout_dir=root,force=False,rebuild=True,raw_json=None,
        mineru_python='unused',tier='basic',dpi=144))
    assert result['status']=='ok' and result['crops']==1
    manifest=json.loads((target/'manifest.json').read_text())
    assert manifest['pages'][0]['blocks'][1]['kind']=='figure'
    assert (target/manifest['pages'][0]['blocks'][1]['path']).is_file()


def test_layout_builder_rejects_unparsed_pages_instead_of_native_fallback(tmp_path):
    module=loader();path=tmp_path/'partial.pdf'
    with pymupdf.open() as pdf:
        pdf.new_page();pdf.new_page().insert_text((40,80),'Real evidence on the second page.')
        pdf.save(path)
    warnings=[]
    assert module.build_manifest(path,{'pages':[{'page_idx':0,'blocks':[]}]},tmp_path,144,'basic',module.sha256_file(path),warnings.append) is None
    assert warnings and '页面不完整' in warnings[0]


def test_knowledge_only_uses_the_same_original_mineru_parser(tmp_path,monkeypatch):
    store=Store(tmp_path/'data');doc,_=store.import_file('figures.pdf',graphics_pdf(),'live',task_stage='knowledge')
    figures(store,doc)
    monkeypatch.setattr(pymupdf.Page,'get_pixmap',lambda *a,**kw:pytest.fail('Knowledge must not force native page shots'))
    units,pages=knowledge.prepare(store,store.document(doc),settings(),lambda:None)
    assert pages==1 and len([u for u in units if u['image']])==2
    assert all(u['evidence_kind']=='pdf_figure' for u in units if u['image'])
