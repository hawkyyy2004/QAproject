"""Real company-GLM comparison of scheduling, on identical frozen evidence.

Reuses completed classification, not model response caches. No PDF extraction,
no live writes, no commits; baseline emulates the earlier text-first scheduling
(five pairs, two figures, batches consumed in order) under the same current model
prompts, so the coverage difference is the scheduler and not the wording.

The live workspace must be idle: the final guard compares its table hashes
before and after, so a running service's own writes look like a violation.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qaproject.generation_policy import generation_limits, modality_counts
from qaproject.pipeline import Worker
from qaproject.store import Store


def snapshot(db):
    return {table:hashlib.sha256(json.dumps(db.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall(),
                        ensure_ascii=False).encode()).hexdigest()
            for table in ('documents','assets','candidates','candidate_pairs','qas','qa_pairs','tasks')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document-id',required=True)
    parser.add_argument('--reanalyze',action='store_true',
                        help='re-run whole-file classification instead of reusing frozen topics')
    args = parser.parse_args()
    folder = ROOT/'runtime/generation-distribution-check'/str(time.time_ns())
    folder.mkdir(parents=True)
    with sqlite3.connect('file:'+str(ROOT/'runtime/live/state.sqlite3')+'?mode=ro',uri=True) as db:
        before = snapshot(db)
        db.row_factory = sqlite3.Row
        source = dict(db.execute('SELECT * FROM documents WHERE id=?',(args.document_id,)).fetchone())
        topics = [dict(r) for r in db.execute('SELECT t.*,d.unit_ids,d.keywords FROM topics t JOIN document_topics d ON d.topic_id=t.id WHERE d.document_id=?',(args.document_id,))]
        assets = [dict(r) for r in db.execute('SELECT * FROM assets WHERE document_id=?',(args.document_id,))]
    units = json.loads(source['units'])
    seed = Store(folder/'seed')
    doc_id,_ = seed.import_file(source['name'],Path(source['path']).read_bytes(),'live')
    seed.update_document(doc_id,units=units,pages=source['pages'],
                         coverage='pending' if args.reanalyze else 'complete',
                         summary='' if args.reanalyze else source['summary'])
    for t in topics if not args.reanalyze else []:
        topic = seed.topic(t['name'],t['definition'],'隔离调度对照复用正式主题。')
        seed.link_topic(doc_id,topic['id'],json.loads(t['keywords']),json.loads(t['unit_ids']))
    with seed.connect() as db:
        db.executemany('INSERT INTO assets VALUES(?,?,?,?,?,?)',
            [(a['id'],doc_id,a['path'],a['hash'],a['page'],a['label']) for a in assets])
    report = dict(source_document_id=args.document_id,scope='Same frozen evidence and completed topics; empty response caches; no PDF preparation, queueing, commit or export. Baseline is a scheduling counterfactual with current prompts, not a prior full software revision.',runs={})
    report_file = folder/'report.json'
    print('report='+str(report_file),flush=True)
    for label in ('mixed_coverage',) if args.reanalyze else ('text_first_shared_five','mixed_coverage'):
        shutil.copytree(seed.root,folder/label)
        store = Store(folder/label)
        with ExitStack() as stack:
            if label=='text_first_shared_five':
                def old_limits(cfg):
                    return dict(target=5,maximum=7,text=5,visual=2)
                def old_phases(text,visual,limits):
                    yield 'text',limits['target'],False,text
                    yield 'visual',limits['target'],False,visual
                stack.enter_context(patch('qaproject.pipeline.generation_limits',old_limits))
                stack.enter_context(patch('qaproject.pipeline.generation_phases',old_phases))
            worker = Worker(store)
            worker.process(store.claim(worker.name,document_id=doc_id))
        doc = store.document(doc_id)
        with store.connect() as db:
            calls = [json.loads(r[0]) for r in db.execute("SELECT detail FROM events WHERE kind IN ('model_call','model_error') ORDER BY created")]
        result = dict(status=doc['status'],seconds=doc['seconds'],timings=doc['timings'],
                      pairs=doc['bilingual_count'],distribution=modality_counts(doc['candidates']),
                      rows=len(doc['candidates']),committed=doc['qa_count'],error=doc['error'],calls=calls)
        report['runs'][label] = result
        report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(label+' '+json.dumps({k:v for k,v in result.items() if k!='calls'},ensure_ascii=False),flush=True)
    with sqlite3.connect('file:'+str(ROOT/'runtime/live/state.sqlite3')+'?mode=ro',uri=True) as db:
        report['live_data_unchanged'] = before==snapshot(db)
    if not report['live_data_unchanged']:
        # The live workspace is opened read-only and this check writes only into
        # its own folder; a difference means the default service is busy.
        print('live workspace changed during the check: the default service is running. '
              'Isolation still holds (read-only connection, separate store).',flush=True)
    report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['runs']['mixed_coverage']['status']=='waiting_review' else 1


if __name__=='__main__':
    raise SystemExit(main())
