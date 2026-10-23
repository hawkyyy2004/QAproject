"""Queue existing unsuccessful bilingual groups through the application's sole worker.

Only --queue writes task requests; no direct edits to candidates, QA or CSV.
Run --report after completion to validate controlled commits in an isolated database copy.
Company credentials are read by the running app only and never copied into reports.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--queue',action='store_true')
    action.add_argument('--report',action='store_true')
    args=parser.parse_args()
    folder=Path((ROOT/'runtime/quality-delivery/latest-path.txt').read_text())
    if args.queue:
        with urllib.request.urlopen('http://127.0.0.1:8765/api/bootstrap') as r:
            session=json.load(r)['session']
        with sqlite3.connect('file:'+str(ROOT/'runtime/live/state.sqlite3')+'?mode=ro',uri=True) as db:
            ids=[r[0] for r in db.execute("SELECT DISTINCT document_id FROM candidates WHERE status='needs_review'")]
        queued=[]
        for doc_id in ids:
            request=urllib.request.Request('http://127.0.0.1:8765/api/documents/'+doc_id+'/optimize',
                                           method='POST',headers={'X-QA-Session':session})
            with urllib.request.urlopen(request) as r:
                if not json.load(r).get('queued'):raise ValueError('优化任务未成功登记。')
            queued.append(doc_id)
        (folder/'queued.json').write_text(json.dumps({'time':time.time(),'documents':queued}))
        print(json.dumps({'queued_documents':len(queued),'report_folder':str(folder)}))
        return

    from qaproject.pipeline import approve
    from qaproject.store import Store
    source=sqlite3.connect('file:'+str(ROOT/'runtime/live/state.sqlite3')+'?mode=ro',uri=True)
    source.row_factory=sqlite3.Row
    if source.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0]:
        raise SystemExit('处理尚未完成。')
    isolated=folder/'commit-validation'
    isolated.mkdir(exist_ok=True)
    with sqlite3.connect(isolated/'state.sqlite3') as copied:source.backup(copied)
    store=Store(isolated)
    before=json.loads((folder/'before.json').read_text())
    baseline={c['id']:c for d in before for c in d['candidates']}
    queued=json.loads((folder/'queued.json').read_text())
    report={'documents':[],'unchanged_healthy_rows':0,'actual_committed_before':source.execute('SELECT COUNT(*) FROM qas').fetchone()[0]}
    for summary in source.execute('SELECT id,name,status,seconds,timings,error FROM documents'):
        doc=store.document(summary['id'])
        for c in doc['candidates']:
            old=baseline.get(c['id'])
            if old and old['status']=='pending':
                if c['row_hash']!=old['row_hash'] or c['evidence_hash']!=old['evidence_hash']:
                    raise ValueError('已通过候选被不必要地改写。')
                report['unchanged_healthy_rows']+=1
            if c['status']=='pending':approve(store,c['id'],True,c['row_hash'])
            elif c['status']!='approved':raise ValueError('仍有未通过的交付候选。')
        item=dict(summary)
        item['timings']=json.loads(item['timings'])
        item.update(delivered_pairs=doc['bilingual_count'],discarded_pairs=doc['discarded_pair_count'],
                    delivered_rows=len(doc['candidates']),isolated_committed=store.document(doc['id'])['qa_count'],
                    optimized=doc['id'] in queued['documents'])
        if item['optimized']:
            events=[json.loads(r[0]) for r in source.execute(
                "SELECT detail FROM events WHERE document_id=? AND kind='model_call' AND created>=? ORDER BY rowid",(doc['id'],queued['time']))]
            item['calls']=[{k:e.get(k) for k in ('stage','purpose','model','seconds','images')} for e in events]
        report['documents'].append(item)
    report['actual_committed_after']=source.execute('SELECT COUNT(*) FROM qas').fetchone()[0]
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'report':str(folder/'report.json'),'healthy_rows_unchanged':report['unchanged_healthy_rows'],
        'delivered_pairs':sum(d['delivered_pairs'] for d in report['documents']),
        'isolated_commit_rows':sum(d['isolated_committed'] for d in report['documents']),
        'actual_committed':report['actual_committed_after']},ensure_ascii=False))


if __name__=='__main__':main()
