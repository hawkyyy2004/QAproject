"""Isolated real-GLM comparison on identical frozen evidence; never commits live QA."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from qaproject.store import Store
    from qaproject.pipeline import Worker, approve
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document-id',required=True)
    parser.add_argument('--baseline-ref',default='HEAD')
    args=parser.parse_args()
    folder=ROOT/'runtime/document-topic-check'/str(time.time_ns())
    folder.mkdir(parents=True)
    baseline=folder/'baseline_qaproject'
    baseline.mkdir()
    revision=subprocess.run(['git','rev-parse',args.baseline_ref],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    for name in ('__init__.py','settings.py','store.py','parsers.py','glm.py','pipeline.py','taxonomy.py'):
        raw=subprocess.run(['git','show',revision+':src/qaproject/'+name],cwd=ROOT,check=True,capture_output=True).stdout
        (baseline/name).write_bytes(raw)
    spec=importlib.util.spec_from_file_location('baseline_qaproject',baseline/'__init__.py',submodule_search_locations=[str(baseline)])
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    from baseline_qaproject import settings as before_settings
    from baseline_qaproject.store import Store as BeforeStore
    from baseline_qaproject.pipeline import Worker as BeforeWorker
    before_settings.ROOT=ROOT
    with sqlite3.connect('file:'+str(ROOT/'runtime/live/state.sqlite3')+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        source=dict(db.execute('SELECT * FROM documents WHERE id=?',(args.document_id,)).fetchone())
        assets=[dict(r) for r in db.execute('SELECT * FROM assets WHERE document_id=?',(args.document_id,))]
    seed=Store(folder/'seed')
    doc_id,_=seed.import_file(source['name'],Path(source['path']).read_bytes(),'live')
    seed.update_document(doc_id,units=json.loads(source['units']),pages=source['pages'])
    with seed.connect() as db:
        db.executemany('INSERT INTO assets VALUES(?,?,?,?,?,?)',
            [(a['id'],doc_id,a['path'],a['hash'],a['page'],a['label']) for a in assets])
    for label in ('baseline','updated'):
        shutil.copytree(seed.root,folder/label)
    report=dict(source_document_id=args.document_id,baseline_revision=revision,frozen_evidence_reused=True,
                response_cache_reused=False,includes_pdf_reparse=False,live_qa_changed=False,runs={})
    report_file=folder/'report.json'
    report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('report='+str(report_file),flush=True)
    for label,store_type,worker_type in (('baseline',BeforeStore,BeforeWorker),('updated',Store,Worker)):
        store=store_type(folder/label)
        worker=worker_type(store)
        print(label+' started',flush=True)
        worker.process(store.claim(worker.name,document_id=doc_id))
        doc=store.document(doc_id)
        with store.connect() as db:
            calls=[json.loads(r[0]) for r in db.execute("SELECT detail FROM events WHERE kind IN ('model_call','model_error') ORDER BY created")]
        result=dict(status=doc['status'],seconds=doc['seconds'],timings=doc['timings'],pages=doc['pages'],
                    topics=[t['name'] for t in doc['topics']],pairs=doc['bilingual_count'],rows=len(doc['candidates']),
                    pending_pairs=len({c['pair_id'] for c in doc['candidates'] if c['status']=='pending'}),
                    committed=doc['qa_count'],error=doc['error'],calls=calls)
        report['runs'][label]=result
        report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(label+' '+json.dumps({k:v for k,v in result.items() if k!='calls'},ensure_ascii=False),flush=True)
        if label=='updated' and doc['status']=='waiting_review':
            for c in doc['candidates']:
                approve(store,c['id'],True,c['row_hash'],dry_run=True,verification_cache=set())
    return 0 if report['runs']['updated']['status']=='waiting_review' else 1


if __name__=='__main__':
    raise SystemExit(main())
