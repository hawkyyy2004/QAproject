"""Run one imported document in an isolated workspace using the configured GLM."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    from qaproject.store import Store
    from qaproject.pipeline import Worker
    parser = argparse.ArgumentParser()
    parser.add_argument('--document-id', required=True)
    parser.add_argument('--name', default='cold-trial')
    parser.add_argument('--recover-live', action='store_true', help='续跑指定真实任务，候选仍须人工核对')
    args = parser.parse_args()
    if not args.name.replace('-', '').isalnum():
        parser.error('name must contain only letters, digits and hyphens')
    original = Store(ROOT/'runtime/live').document(args.document_id)
    folder = ROOT/'runtime/performance-fix'/args.name
    folder.mkdir(parents=True, exist_ok=True)
    store = Store(ROOT/'runtime/live') if args.recover_live else Store(folder/'workspace')
    if args.recover_live:
        doc_id = original['id']
    else:
        doc_id, created = store.import_file(original['name'], Path(original['path']).read_bytes(), 'live')
        if not created:
            parser.error('Use a new name for a cold run; previous evidence is preserved')
    (folder/'document-id.txt').write_text(doc_id)
    worker = Worker(store)
    task = store.claim(worker.name, document_id=doc_id)
    if not task:
        parser.error('The selected task is not queued; it may already be running or finished')
    worker.process(task)
    doc = store.document(doc_id)
    with store.connect() as db:
        calls = [json.loads(r[0]) for r in db.execute("SELECT detail FROM events WHERE document_id=? AND kind IN ('model_call','model_error') ORDER BY created",(doc_id,))]
    report = {'source_document_id':original['id'], 'document_id':doc_id, 'status':doc['status'],
              'pages':doc['pages'], 'seconds':doc['seconds'], 'timings':doc['timings'],
              'candidates':len(doc['candidates']), 'committed':doc['qa_count'], 'error':doc['error'],
              'calls':calls, 'cold':not args.recover_live}
    (folder/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if doc['status']=='waiting_review' else 1


if __name__=='__main__':
    raise SystemExit(main())
