"""One preparation feeds independent, unchanged QA/knowledge consumers."""
import io
import json
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from qaproject.app import create_app
from qaproject import corpus,corpus_processing,corpus_quality,knowledge
from qaproject.pipeline import Worker
from qaproject.execution import ExecutionLimits
from test_file_concurrency import provider


def test_default_qa_archives_first_and_reads_native_pdf_once(tmp_path,monkeypatch):
    app=create_app(tmp_path/'data',workers=False);client=TestClient(app);client.headers['x-qa-session']=app.state.session
    with pymupdf.open() as pdf:
        p=pdf.new_page();p.insert_text((40,80),'Ocean observations show temperature and salinity values of 15.')
        raw=pdf.tobytes()
    doc=client.post('/api/imports',files={'files':('ocean.pdf',raw)}).json()[0]['id'];store=app.state.stores['live']
    opens=[];original=pymupdf.open
    def opened(*args,**kw):
        opens.append(1);return original(*args,**kw)
    monkeypatch.setattr(pymupdf,'open',opened)
    monkeypatch.setattr(corpus_processing,'pdf_structure',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Must capture structure during the first pass')))
    calls,_=provider(monkeypatch)
    w=Worker(store,limits=ExecutionLimits())
    try:
        assert w.run_once()  # Corpus
        assert len(opens)==1 and not calls
        version=corpus.listing(store)['items'][0]['id']
        assert corpus.verify(store,version)['manifest']['processing']['structure']['pages']
        monkeypatch.setattr('qaproject.pipeline.extract',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('QA must reuse Corpus')))
        assert w.run_once()  # QA uses exactly the existing GLM generation/review pipeline.
        assert len(opens)==1
        result=store.document(doc)
        assert result['bilingual_count']==5 and len(calls)==7,result['error']
        with store.connect() as db:
            assert db.execute('SELECT version_id FROM qa_corpus_inputs WHERE document_id=?',(doc,)).fetchone()[0]==version
    finally:w.limits.close()


def test_failed_archive_blocks_consumers_retry_reuses_preparation(tmp_path,monkeypatch):
    app=create_app(tmp_path/'data',workers=False);client=TestClient(app);client.headers['x-qa-session']=app.state.session
    doc=client.post('/api/imports?processing=qa&processing=knowledge',files={'files':('sea.txt',b'Ocean evidence for a shared preparation.')}).json()[0]['id'];store=app.state.stores['live']
    extract=corpus.extract;count=[]
    def parsed(*args,**kw):count.append(1);return extract(*args,**kw)
    monkeypatch.setattr(corpus,'extract',parsed)
    publish=corpus._publish
    monkeypatch.setattr(corpus,'_publish',lambda *a,**kw:(_ for _ in ()).throw(OSError('isolated archive failure')))
    w=Worker(store,limits=ExecutionLimits())
    try:
        assert w.run_once()
        assert corpus.jobs(store)['items'][0]['status']=='failed' and len(count)==1
        assert store.claim('other-worker',document_id=doc) is None  # Neither consumer may silently parse again.
        with store.connect() as db:
            assert db.execute("SELECT count(*) FROM tasks WHERE document_id=? AND status='queued'",(doc,)).fetchone()[0]==2
        monkeypatch.setattr(corpus,'_publish',publish)
        corpus.enqueue(store,doc)
        assert w.run_once() and len(count)==1
        task=store.claim('consumer',document_id=doc)
        assert task['stage']=='queued'
        store.task_update(task['id'],'consumer','failed','failed')
        # Knowledge can run despite a downstream QA failure; it reuses Corpus too.
        task=store.claim('knowledge-consumer',document_id=doc)
        assert task['stage']=='knowledge'
        monkeypatch.setattr(knowledge,'extract',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Knowledge must reuse Corpus')))
        prepared=knowledge.prepare(store,store.document(doc),{},lambda:None)
        assert prepared[0]==corpus.prepared_units(store,doc)[0] and len(count)==1
    finally:w.limits.close()


def test_quality_failure_checkpoints_before_model_retry(tmp_path,monkeypatch):
    from qaproject.store import Store
    store=Store(tmp_path/'data');doc,_=store.import_file('sea.txt',b'Frozen scientific evidence.','live',task_stage='corpus')
    refine=corpus_quality.refine
    monkeypatch.setattr(corpus_quality,'refine',lambda *a,**kw:(_ for _ in ()).throw(ValueError('isolated model unavailable')))
    w=Worker(store,limits=ExecutionLimits())
    try:
        assert w.run_once()
        monkeypatch.setattr(corpus_quality,'refine',refine)
        monkeypatch.setattr(corpus,'extract',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Prepared evidence must survive a quality failure')))
        corpus.enqueue(store,doc)
        assert w.run_once() and corpus.listing(store)['total']==1
    finally:w.limits.close()


def test_later_knowledge_request_waits_for_already_pending_corpus(tmp_path):
    app=create_app(tmp_path/'data',workers=False);client=TestClient(app);client.headers['x-qa-session']=app.state.session
    doc=client.post('/api/imports',files={'files':('sea.txt',b'Shared original evidence.')}).json()[0]['id']
    store=app.state.stores['live'];knowledge.enqueue(store,doc)
    task=store.claim('preparing',document_id=doc);assert task['stage']=='corpus'
    store.task_update(task['id'],'preparing','corpus','failed')
    assert store.claim('consumer',document_id=doc) is None
    with store.connect() as db:
        assert db.execute('SELECT count(*) FROM task_dependencies').fetchone()[0]==2


def test_default_inherits_existing_mineru_policy_and_does_not_extract_native_body(tmp_path,monkeypatch):
    from qaproject import layout
    from test_layout_manifest import manifest_for
    app=create_app(tmp_path/'data',workers=False);client=TestClient(app);client.headers['x-qa-session']=app.state.session
    with pymupdf.open() as pdf:
        p=pdf.new_page();p.insert_text((40,80),'Native content should not replace MinerU content.')
        raw=pdf.tobytes()
    doc=client.post('/api/imports',files={'files':('sea.pdf',raw)}).json()[0]['id'];store=app.state.stores['live']
    monkeypatch.setattr(layout,'runner',lambda config:'existing-mineru-python')
    manifest_for(store,doc,[dict(page=1,blocks=[dict(kind='text',text='MinerU salinity observation 35 PSU.')])])
    monkeypatch.setattr(layout,'prepare',lambda *a:(_ for _ in ()).throw(AssertionError('Use the existing MinerU manifest')))
    original=pymupdf.Page.get_text
    def get_text(self,option='text',*a,**kw):
        assert option!='text','The native preparer must not generate a second body before MinerU.'
        return original(self,option,*a,**kw)
    monkeypatch.setattr(pymupdf.Page,'get_text',get_text)
    w=Worker(store,limits=ExecutionLimits())
    try:assert w.run_once()
    finally:w.limits.close()
    version=corpus.verify(store,corpus.listing(store)['items'][0]['id'])
    assert version['manifest']['processing']['parser_selection']['resolved']=='mineru'
    assert any(u['text']=='MinerU salinity observation 35 PSU.' for u in version['units'])
    assert corpus_processing.configured_engine(store,store.document(doc),{'pdf_layout':'off'})[0]=='native'
    assert corpus_processing.configured_engine(store,store.document(doc),{'pdf_layout':'auto'},'native')[0]=='native'
