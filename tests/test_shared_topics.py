"""Shared broad labels coexist with distinctive labels and scoped fine-term search."""
import pytest

from qaproject.glm import ANALYZE, FACTS, GLMClient
from qaproject.pipeline import Worker, approve
from qaproject.store import CORE_TOPICS, Store, encode, classification_catalog
from test_document_topics import evidence, proposal
from test_bilingual import run, setup_pipeline


@pytest.mark.parametrize("shared_name", ["物理海洋", "海洋声学"])
def test_two_papers_share_broad_topic_and_keep_a_unique_one(tmp_path, monkeypatch, shared_name):
    store = Store(tmp_path/'live')
    ids, calls = [], []
    for n, word in enumerate(('海冰同化', '环流诊断')):
        doc_id, _ = store.import_file(f'{n}.txt', f'关键词：{word}；{shared_name}。'.encode(), 'live')
        unit = evidence(n+1, f'关键词：{word}；{shared_name}。研究该领域的不同过程。')
        store.update_document(doc_id, units=[unit])
        ids.append(doc_id)
    distinctive = '极地海冰可预报性'
    def call(self, instruction, payload, units=(), **kwargs):
        calls.append(payload)
        if 'existing_topics' in payload:
            # Compact catalog retains meanings and shares stable names across papers.
            assert isinstance(payload['existing_topics'], dict)
            assert (shared_name in payload['existing_topics']) == (shared_name=='物理海洋' or bool(store.document(ids[0])['topics']))
            word = '海冰同化' if '海冰同化' in units[0]['text'] else '环流诊断'
            topics = [proposal(shared_name, units[0], keywords=[word])]
            if word=='海冰同化':
                topics.append(proposal(distinctive, units[0], keywords=['可预报性']))
            return dict(summary='海洋动力研究。', topics=topics), {}
        return dict(pairs=[], skipped=[]), {}
    monkeypatch.setattr(GLMClient, 'call', call)
    worker = Worker(store)
    for doc_id in ids:
        worker.process(store.claim(worker.name, document_id=doc_id))
    documents = [store.document(doc_id) for doc_id in ids]
    assert all(d['coverage']=='complete' and 1<=len(d['topics'])<=5 for d in documents)
    shared = next(t for t in store.topics() if t['name']==shared_name)
    assert shared['documents']==2
    assert distinctive in {t['name'] for t in documents[0]['topics']}
    assert distinctive not in {t['name'] for t in documents[1]['topics']}
    assert {d['id'] for d in store.search(shared_name)[0]['documents']}==set(ids)
    assert {d['id'] for d in store.search('海冰同化')[0]['documents']}=={ids[0]}
    assert len([p for p in calls if 'existing_topics' in p])==2  # One existing analysis per short paper.
    # Reopening doesn't fold an unlisted unique label into an assumed parent.
    assert distinctive in {t['name'] for t in Store(store.root).document(ids[0])['topics']}


def test_legacy_specific_topics_share_parents_without_any_model_calls(tmp_path):
    store = Store(tmp_path/'live')
    physical = store.topic('物理海洋')
    ids = []
    for n, name in enumerate(('南海环流', '太平洋年代际振荡')):
        doc_id, _ = store.import_file(f'{n}.txt', str(n).encode(), 'live')
        old = store.topic(name, '旧版具体方向。', '隔离迁移测试。')
        unit = evidence(n+1)
        store.update_document(doc_id, units=[unit], coverage='complete', status='waiting_review')
        store.link_topic(doc_id, old['id'], [name+'指标'], [unit['id']])
        with store.connect() as db:
            db.execute("UPDATE tasks SET status='succeeded' WHERE document_id=?", (doc_id,))
        ids.append(doc_id)
    reopened = Store(store.root)
    assert all([t['id'] for t in reopened.document(d)['topics']]==[physical['id']] for d in ids)
    assert {d['id'] for d in reopened.search('物理海洋')[0]['documents']}==set(ids)
    assert {d['id'] for d in reopened.search('南海环流')[0]['documents']}=={ids[0]}
    with reopened.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM tasks WHERE status='queued'").fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='shared_topics_reused'").fetchone()[0]==2
    Store(store.root)
    with reopened.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='shared_topics_reused'").fetchone()[0]==2


def test_parent_reuse_preserves_candidates_sources_reviews_and_old_redirects(setup_pipeline):
    _, store, doc_id, _, _, _, _ = setup_pipeline
    doc = run(setup_pipeline)
    first = doc['candidates'][0]
    approve(store,first['id'],True,first['row_hash'])
    old = store.topic('南海环流', '南海的海洋环流研究。', '模拟旧分类。')
    support = encode([u['id'] for u in doc['units']])
    with store.connect() as db:
        db.execute('UPDATE document_topics SET topic_id=? WHERE document_id=?', (old['id'],doc_id))
        db.execute('UPDATE document_search_terms SET topic_id=? WHERE document_id=?', (old['id'],doc_id))
        # Simulate a previously reviewed file-level source redirect.
        db.execute('INSERT INTO document_topic_redirects VALUES(?,?,?,?,?)',
                   (doc_id,doc['topics'][0]['id'],old['id'],support,0))
        before = [tuple(r) for r in db.execute('SELECT * FROM candidates ORDER BY id')]
        qas_before = [tuple(r) for r in db.execute('SELECT * FROM qas ORDER BY id')]
    reopened = Store(store.root)
    after = reopened.document(doc_id)
    assert [t['name'] for t in after['topics']]==['物理海洋']
    assert {c['canonical_topic_id'] for c in after['candidates']}=={after['topics'][0]['id']}
    with reopened.connect() as db:
        assert [tuple(r) for r in db.execute('SELECT * FROM candidates ORDER BY id')]==before
        assert [tuple(r) for r in db.execute('SELECT * FROM qas ORDER BY id')]==qas_before
    for c in after['candidates']:
        approve(reopened, c['id'], True, c['row_hash'])
    assert reopened.document(doc_id)['qa_count']==10


def test_known_parent_migration_defers_a_live_lease(tmp_path):
    store = Store(tmp_path/'live')
    doc_id, _ = store.import_file('active.txt', b'active', 'live')
    old = store.topic('南海环流', '已知方向。', '隔离测试。')
    store.link_topic(doc_id, old['id'], [], ['frozen'])
    task = store.claim('existing-worker')
    assert Store(store.root).document(doc_id)['topics'][0]['id']==old['id']
    with store.connect() as db:
        db.execute("UPDATE tasks SET status='succeeded' WHERE id=?", (task['id'],))
    assert Store(store.root).document(doc_id)['topics'][0]['name']=='物理海洋'


def test_guidance_prioritizes_paper_keywords_and_permits_distinctive_topics():
    assert '原文“关键词/Keywords”' in FACTS
    assert '至少包含一个' in ANALYZE and '允许同时有独有主题' in ANALYZE
    assert '可用原文核心关键词作主题' in ANALYZE
    assert '不要所有论文都归海洋观测' in ANALYZE
    assert {'物理海洋','海洋生态','海洋探索'} <= {t[0] for t in CORE_TOPICS}


def test_compact_catalog_keeps_unique_definitions_but_omits_known_fine_directions(tmp_path):
    store = Store(tmp_path/'live')
    store.topic('南海环流','较细的区域环流方向。','历史测试。')
    unique = store.topic('独立交叉领域','有明确边界的特色主题。','其他类别无法覆盖。')
    topics = store.topics()
    catalog = classification_catalog(topics)
    assert '南海环流' not in catalog
    assert catalog[unique['name']]==unique['definition'] and '物理海洋' in catalog
    import json
    previous = [{k:t[k] for k in ('name','definition','aliases')} for t in topics]
    assert len(json.dumps(catalog,ensure_ascii=False))<len(json.dumps(previous,ensure_ascii=False))
