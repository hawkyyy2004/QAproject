"""Re-prepare frozen evidence for a document that has no approved QA yet.

Evidence is frozen on purpose: an approved QA is bound to the exact units and
image hashes it was reviewed against. So this only touches documents whose
answers are still drafts, and only after the evidence pipeline itself changed
(for example the MinerU layout now crops figures instead of rendering pages).

    python scripts/rebuild_evidence.py --data-dir runtime/live --all-unapproved
    python scripts/rebuild_evidence.py --data-dir runtime/live --document <id> --apply

Without ``--apply`` nothing is written; the run only reports what would change.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from qaproject.settings import runtime_directory  # noqa: E402
from qaproject.store import Store, uid  # noqa: E402


def blockers(store, doc_id):
    with store.connect() as db:
        approved = db.execute("SELECT COUNT(*) FROM candidates WHERE document_id=? AND status='approved'",
                              (doc_id,)).fetchone()[0]
        committed = db.execute('SELECT COUNT(*) FROM qas WHERE id IN '
                               '(SELECT id FROM candidates WHERE document_id=?)', (doc_id,)).fetchone()[0]
        applied = db.execute("SELECT COUNT(*) FROM qa_revisions WHERE document_id=? AND status='applied'",
                             (doc_id,)).fetchone()[0]
        active = db.execute("SELECT COUNT(*) FROM tasks WHERE document_id=? AND status IN ('queued','running')",
                            (doc_id,)).fetchone()[0]
    reasons = []
    if approved or committed:
        reasons.append(f'已有{approved}条入库问答，证据冻结')
    if applied:
        reasons.append(f'已有{applied}条生效的问答修订，证据冻结')
    if active:
        reasons.append('该文件正在排队或处理中')
    return reasons


def rebuild(store, doc_id, apply=False):
    doc = store.document(doc_id)
    reasons = blockers(store, doc_id)
    if reasons:
        return {'document': doc_id, 'name': doc['name'], 'skipped': '；'.join(reasons)}
    folder = store.root / 'assets/images' / doc_id
    files = len(list(folder.glob('*'))) if folder.is_dir() else 0
    plan = {'document': doc_id, 'name': doc['name'], 'units': doc['unit_count'],
            'asset_files': files, 'status': doc['status']}
    if not apply:
        return {**plan, 'would_reset': True}
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('DELETE FROM candidate_pairs WHERE candidate_id IN '
                   '(SELECT id FROM candidates WHERE document_id=?)', (doc_id,))
        db.execute('DELETE FROM candidates WHERE document_id=?', (doc_id,))
        db.execute('DELETE FROM qa_pairs WHERE id NOT IN (SELECT pair_id FROM candidate_pairs)')
        db.execute("DELETE FROM qa_revisions WHERE document_id=? AND status!='applied'", (doc_id,))
        db.execute('DELETE FROM document_topics WHERE document_id=?', (doc_id,))
        db.execute('DELETE FROM document_search_terms WHERE document_id=?', (doc_id,))
        db.execute('DELETE FROM assets WHERE document_id=?', (doc_id,))
        db.execute("UPDATE documents SET units='[]', unit_count=0, pages=0, summary='', coverage='pending', "
                   "status='queued', error='', progress='等待重新准备证据（MinerU 版面）', seconds=0, timings='{}' "
                   'WHERE id=?', (doc_id,))
        db.execute('INSERT INTO tasks(id,document_id,status) VALUES(?,?,?)', (uid(), doc_id, 'queued'))
        db.execute('INSERT INTO events VALUES(?,?,?,?,?)',
                   (uid(), doc_id, 'evidence_reset', '{"reason":"evidence pipeline changed"}', time.time()))
    if folder.is_dir():
        shutil.rmtree(folder)
    return {**plan, 'requeued': True}


def main():
    parser = argparse.ArgumentParser(description='重建无入库问答文件的证据')
    parser.add_argument('--data-dir', type=Path, help='数据目录，默认沿用 settings/runtime')
    parser.add_argument('--document', action='append', default=[], help='文件ID，可重复')
    parser.add_argument('--all-unapproved', action='store_true', help='所有尚无入库问答的文件')
    parser.add_argument('--apply', action='store_true', help='真正写入；不加只做检查')
    args = parser.parse_args()
    store = Store(args.data_dir or runtime_directory())
    ids = list(args.document)
    if args.all_unapproved:
        with store.connect() as db:
            ids += [row[0] for row in db.execute(
                "SELECT id FROM documents WHERE id NOT IN "
                "(SELECT document_id FROM candidates WHERE status='approved')")]
    if not ids:
        raise SystemExit('请给出 --document <ID> 或 --all-unapproved。')
    for doc_id in dict.fromkeys(ids):
        print(rebuild(store, doc_id, args.apply))


if __name__ == '__main__':
    main()
