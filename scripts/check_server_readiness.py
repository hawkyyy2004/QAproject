"""Isolated before/after timing with synthetic PDFs and simulated GLM latency."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def measure(args):
    sys.path.insert(0,str(args.baseline if args.variant=='before' else ROOT/'src'))
    from qaproject import settings as config_module
    config_module.ROOT = ROOT
    cfg = json.loads((args.baseline/'settings.json' if args.variant=='before' else ROOT/'config/settings.json').read_text())
    cfg['semantic_search'] = {'enabled':False,'cross_document_dedup':False}
    config_module.settings = lambda:cfg
    from qaproject.store import Store, encode, digest
    from qaproject.pipeline import Worker, CHECKS, BILINGUAL_CHECKS
    from qaproject.glm import GLMClient
    calls = []
    def call(self,instruction,payload,units=(),**kwargs):
        kind = 'facts' if payload.get('analysis_phase')=='facts' else 'analysis' if 'existing_topics' in payload else 'review' if 'candidates' in payload else 'generation'
        calls.append(kind)
        time.sleep(.05)  # Same controlled latency on both versions; never a real provider call.
        if kind=='facts':
            return {'summary':'','facts':[dict(unit_id=u['id'],focus='合成温度观测领域。',keywords=['温度']) for u in units]}, {'seconds':.05}
        result = {}
        if kind=='analysis':
            support = [ref for fact in payload.get('document_facts',[]) for ref in fact.get('unit_ids',[fact.get('unit_id')])]
            result.update(summary='合成温度观测。',topics=[dict(name='海洋观测',keywords=['温度'],unit_ids=support or [u['id'] for u in units])])
        if kind=='review':
            result['checks'] = [dict(candidate_id=c['candidate_id'],note='模拟核对中英数值和范围。',**{k:True for k in CHECKS+BILINGUAL_CHECKS}) for c in payload['candidates']]
        elif kind=='generation' or 'target' in payload:
            start = len(payload['existing_questions'])//2
            result.update(pairs=[dict(unit_id=payload['units'][0]['id'],topic='海洋观测',
                zh=dict(input=f'合成海域指标{start+n}的观测值是什么？',thinking='合成观测值为15。',output='15。'),
                en=dict(input=f'What is the observed value of synthetic indicator {start+n}?',thinking='The synthetic value is 15.',output='15.')) for n in range(payload['target'])],skipped=[])
        return result,{'seconds':.05}
    GLMClient.call = call
    store = Store(args.workspace)
    library = None
    if args.variant=='after' and args.reference:
        from qaproject.library import LibraryIndex
        from qaproject.source_files import signature
        library = LibraryIndex(store,args.pdf.parent)
        path = args.pdf.resolve()
        entry_id,version = digest([str(library.root),path.name]),signature(path)
        library.flush([(entry_id,str(library.root),path.name,path.name,path.name.casefold(),path.stat().st_size,
                        path.stat().st_mtime,version,'benchmark')])
    raw = None if library else args.pdf.read_bytes()
    began = time.perf_counter()
    if library:
        result = library.import_entry(entry_id,version,25*1024*1024)
        doc_id = result['id']
        assert not list((store.root/'originals').iterdir())
    else:
        doc_id,_ = store.import_file('sample.pdf',raw,'live')
    registration = time.perf_counter()-began
    stop = threading.Event()
    browse = None
    listing = []
    if args.variant=='after' and args.browse:
        from qaproject.library import LibraryIndex
        library = library or LibraryIndex(store,args.pdf.parent)
        rows = [(digest([str(library.root),str(n)]),str(library.root),f'paper-{n:05d}.pdf',f'paper-{n:05d}.pdf',f'paper-{n:05d}.pdf',1000,0.,'metadata','bench') for n in range(20000)]
        library.flush(rows)
        def browse_loop():
            while not stop.wait(.2):
                started = time.perf_counter()
                library.list_files('paper',page=100,page_size=30)
                listing.append(time.perf_counter()-started)
        browse = threading.Thread(target=browse_loop)
        browse.start()
    started = time.perf_counter()
    worker = Worker(store)
    worker.run_once()
    total = time.perf_counter()-started
    stop.set()
    if browse:
        browse.join()
    doc = store.document(doc_id)
    assert doc['status']=='waiting_review' and doc['bilingual_count']==5, doc['error']
    images = list((store.root/'assets/images').rglob('*.png'))
    print(json.dumps(dict(registration_seconds=registration,total_seconds=total,timings=doc['timings'],
        pages=doc['pages'],units=len(doc['units']),images=len(images),image_bytes=sum(p.stat().st_size for p in images),
        pairs=doc['bilingual_count'],calls=calls,catalogue_queries=len(listing),
        catalogue_query_p95=sorted(listing)[max(0,int(len(listing)*.95)-1)] if listing else None)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True,help='Pre-change qaproject snapshot directory under runtime')
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--variant',choices=['before','after'])
    parser.add_argument('--workspace',type=Path)
    parser.add_argument('--pdf',type=Path)
    parser.add_argument('--browse',action='store_true')
    parser.add_argument('--reference',action='store_true',help='After variant reads its original directly from the directory')
    args = parser.parse_args()
    if args.variant:
        measure(args)
        return
    if not 1<=args.rounds<=10:
        parser.error('rounds must be 1–10')
    import pymupdf
    output = args.output or ROOT/'runtime/server-readiness-check'/str(time.time_ns())
    output.mkdir(parents=True,exist_ok=True)
    cases = {}
    for name,mixed in [('text',False),('mixed',True)]:
        path = output/(name+'.pdf')
        with pymupdf.open() as pdf:
            for n in range(60):
                page = pdf.new_page()
                page.insert_textbox((40,40,550,680),('Synthetic area: temperature, salinity, wind and waves were measured at 15.\n')*12)
                if mixed and n%6==0:
                    page.draw_rect((100,500,400,650))
                    page.insert_text((100,675),'Figure 1. Synthetic comparison of measured conditions.')
            pdf.save(path)
        cases[name] = path
    report = dict(scope='Fresh tasks and evidence; OS caches may be warm. Real local PDF preparation with simulated 50ms GLM calls. No real GLM, no live data. After variant additionally browses a synthetic cached 20,000-row catalogue at 5 queries/s. Embedding disabled equally in both isolated variants.',cases={})
    for name,path in cases.items():
        results = {'before':[],'after':[]}
        for n in range(args.rounds):
            for variant in (('before','after') if n%2==0 else ('after','before')):
                command = [sys.executable,str(Path(__file__).resolve()),'--baseline',str(args.baseline.resolve()),'--variant',variant,
                    '--workspace',str(output/name/f'{variant}-{n}'),'--pdf',str(path),'--browse']
                if args.reference:
                    command.append('--reference')
                completed = subprocess.run(command,cwd=ROOT,text=True,capture_output=True,check=True,timeout=120,
                                           env={k:v for k,v in os.environ.items() if k!='API_KEY'})
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                results[variant].append(result)
                print(json.dumps(dict(case=name,round=n,variant=variant,total=result['total_seconds']),ensure_ascii=False),flush=True)
        for n in range(args.rounds):
            before,after = results['before'][n],results['after'][n]
            for field in ('pages','units','images','image_bytes','pairs'):
                assert before[field]==after[field],field
            assert sorted(before['calls'])==sorted(after['calls'])
        medians = {variant:statistics.median(r['total_seconds'] for r in values) for variant,values in results.items()}
        preparation = {variant:statistics.median(r['timings']['parsing'] for r in values) for variant,values in results.items()}
        report['cases'][name] = dict(runs=results,median_total_seconds=medians,median_preparation_seconds=preparation,
            total_change_percent=round((medians['after']/medians['before']-1)*100,2),
            preparation_change_percent=round((preparation['after']/preparation['before']-1)*100,2))
    destination = output/'report.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(str(destination),flush=True)


if __name__=='__main__':
    main()
