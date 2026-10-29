"""Offline SQLite concurrency and synthetic file timing; never calls a provider."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def measure(args):
    sys.path.insert(0,str(Path(args.implementation_root)/'src'))
    os.environ['QA_SQLITE_JOURNAL_MODE']=args.mode
    from qaproject.store import Store
    from qaproject.settings import settings
    from qaproject.pipeline import Worker, CHECKS, BILINGUAL_CHECKS, approve_document
    from qaproject.execution import ExecutionLimits
    from qaproject.glm import GLMClient
    import qaproject.pipeline as pipeline
    import pymupdf
    cfg=dict(settings(),semantic_search={'enabled':False,'cross_document_dedup':False},pdf_layout='off')
    pipeline.settings=lambda:cfg
    calls=[]
    def call(self,instruction,payload,units=(),**kwargs):
        kind='facts' if payload.get('analysis_phase')=='facts' else 'analysis' if 'existing_topics' in payload else 'review' if 'candidates' in payload else 'generation'
        calls.append(kind)
        time.sleep(.05)  # Simulated, identical provider wait; not a real GLM measurement.
        if kind=='facts':
            return {'summary':'','facts':[dict(unit_id=u['id'],focus='温度观测记录',keywords=['温度']) for u in units]},{}
        if kind=='analysis':
            return dict(summary='合成温度记录',topics=[dict(name='海洋观测',keywords=['温度'],unit_ids=[u['id'] for u in units])]),{}
        if kind=='review':
            return dict(checks=[dict(candidate_id=c['candidate_id'],note='模拟中英核对通过',**{k:True for k in CHECKS+BILINGUAL_CHECKS}) for c in payload['candidates']]),{}
        n=len(payload['existing_questions'])//2
        return dict(pairs=[dict(unit_id=units[0]['id'],topic=payload['topics'][0]['name'],
            zh=dict(input=f'合成观测{n+i}中温度是多少？',thinking='记录温度15摄氏度。',output='15摄氏度。'),
            en=dict(input=f'What is the temperature in synthetic observation {n+i}?',thinking='The record is 15 degrees Celsius.',output='15 degrees Celsius.')) for i in range(payload['target'])],skipped=[]),{}
    GLMClient.call=call
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    # Raw concurrent short transactions, with exact row count and integrity checks.
    raw=[]
    for count in (2,4,8):
        store=Store(output/f'writes-{count}')
        barrier=threading.Barrier(count)
        def writes(k):
            barrier.wait()
            for n in range(50):
                with store.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    db.execute('INSERT INTO events VALUES(?,NULL,?,?,?)',(f'{k}-{n}','isolated','{}',0))
        began=time.perf_counter()
        with ThreadPoolExecutor(max_workers=count) as pool:
            list(pool.map(writes,range(count)))
        with store.connect() as db:
            assert db.execute('SELECT COUNT(*) FROM events').fetchone()[0]==50*count
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        raw.append(dict(threads=count,writes=50*count,seconds=time.perf_counter()-began,errors=0))
    runs=[]
    for round_no in range(args.rounds):
        store=Store(output/f'files-{round_no}')
        docs=[]
        registration=time.perf_counter()
        for n in range(2):
            pdf=pymupdf.open()
            for page_no in range(12):
                page=pdf.new_page()
                page.insert_text((40,70),f'Isolated station {n} page {page_no}. Temperature is 15 degrees Celsius. '*4)
            raw_pdf=pdf.tobytes();pdf.close()
            doc_id,_=store.import_file(f'isolated-{n}.pdf',raw_pdf,'live')
            docs.append(doc_id)
        registration=time.perf_counter()-registration
        limits=ExecutionLimits(2,2)
        workers=[Worker(store,limits=limits) for _ in range(2)]
        before=len(calls)
        began=time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda worker:worker.run_once(),workers))
        finally:
            limits.close()
        processing=time.perf_counter()-began
        states=[store.document(d) for d in docs]
        assert all(d['status']=='waiting_review' and d['bilingual_count']==5 for d in states),[(d['status'],d['error']) for d in states]
        commit=time.perf_counter()
        for d in docs:
            approve_document(store,d)
        commit=time.perf_counter()-commit
        export=time.perf_counter()
        with store.connect() as db:
            exported=[tuple(r) for r in db.execute('SELECT row_json FROM qas')]
        export=time.perf_counter()-export
        assert len(exported)==20
        runs.append(dict(registration=registration,processing=processing,commit=commit,
                         export_query=export,stages=[d['timings'] for d in states],calls=calls[before:],
                         pages=[d['pages'] for d in states],pairs=[d['bilingual_count'] for d in states]))
    print(json.dumps(dict(sqlite=__import__('sqlite3').sqlite_version,mode=args.mode,raw=raw,files=runs,
        medians={k:statistics.median(r[k] for r in runs) for k in ('registration','processing','commit','export_query')},
        scope='Local database + two synthetic 12-page PDFs; simulated 50ms provider waits. Not Windows or real GLM acceptance.'),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root')
    parser.add_argument('--implementation-root',default=str(ROOT),help=argparse.SUPPRESS)
    parser.add_argument('--mode',choices=['WAL','DELETE'],default='WAL')
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--output',default=str(ROOT/'runtime/sqlite-journal-check'/str(time.time_ns())))
    args=parser.parse_args()
    if args.baseline_root:
        out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
        results={}
        for label,root,mode in [('before_wal',Path(args.baseline_root).resolve(),'WAL'),('after_wal',ROOT,'WAL'),('after_delete',ROOT,'DELETE')]:
            done=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--implementation-root',str(root),
                '--mode',mode,'--rounds',str(args.rounds),'--output',str(out/label)],check=True,capture_output=True,
                text=True,encoding='utf-8',errors='replace')
            results[label]=json.loads(done.stdout)
        (out/'report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v['medians'] for k,v in results.items()},ensure_ascii=False,indent=2))
    else:
        measure(args)


if __name__=='__main__':
    main()
