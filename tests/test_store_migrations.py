"""Older workspaces keep their rows and reach the same schema on repeated startup."""
import pytest

from qaproject.store import Store


@pytest.mark.parametrize('version', ['oldest', 'partial', 'current'])
def test_reopening_old_workspace_preserves_data_and_column_defaults(tmp_path, version):
    store = Store(tmp_path/'live')
    doc_id, _ = store.import_file('source.txt', b'frozen source', 'live')
    store.update_document(doc_id, units=[{'id': 'u1'}, {'id': 'u2'}])
    with store.connect() as db:
        tables = ['tasks', 'qa_pairs', 'documents']
        expected = {t: [tuple(r) for r in db.execute(f'PRAGMA table_info({t})')] for t in tables}
        original = dict(db.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone())
        task = dict(db.execute('SELECT * FROM tasks WHERE document_id=?', (doc_id,)).fetchone())
        if version != 'current':
            removals = {
                'tasks': ['review_candidate_id', 'revision_id', 'initial_stage'],
                'qa_pairs': ['repair_attempts'],
                'documents': ['unit_count', 'source_kind', 'source_root', 'source_signature', 'progress', 'timings'],
            }
            for table, columns in removals.items():
                # A partially upgraded release already has the first added column.
                for column in columns[1 if version == 'partial' else 0:]:
                    db.execute(f'ALTER TABLE {table} DROP COLUMN {column}')
    for _ in range(2):
        reopened = Store(store.root)
        with reopened.connect() as db:
            assert {t: [tuple(r) for r in db.execute(f'PRAGMA table_info({t})')] for t in tables} == expected
            assert dict(db.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()) == original
            assert dict(db.execute('SELECT * FROM tasks WHERE document_id=?', (doc_id,)).fetchone()) == task
        assert reopened.document(doc_id)['units'] == [{'id': 'u1'}, {'id': 'u2'}]
        assert (store.root/'originals'/doc_id/'source.txt').read_bytes() == b'frozen source'
