"""A file has at most ten active, deduplicated fine search terms across all topics."""
import json

import pytest

from qaproject.store import Store, encode, search_term_key, limit_search_terms, uid
from qaproject.taxonomy import TopicSelection
from test_document_topics import evidence, proposal


def assert_bounded(store, doc_id):
    detail = store.document(doc_id)
    detail_words = [w for t in detail['topics'] for w in t['keywords']]
    summary = next(d for d in store.documents(summary_only=True) if d['id']==doc_id)
    summary_words = [w for t in summary['topics'] for w in t['keywords']]
    keys = [search_term_key(w) for w in detail_words]
    assert len(keys)==len(set(keys))<=10
    assert set(summary_words)==set(detail_words)=={t['term'] for t in detail['search_terms']}
    return detail


def test_full_document_terms_prioritize_core_and_merge_spelling_with_evidence(tmp_path):
    store = Store(tmp_path/'live')
    units = [evidence(n) for n in (1,2,3)]
    selection = TopicSelection(store.topics(), units)
    selection.add(dict(summary='',topics=[proposal('数学',units[0],1,[f'次要词{n}' for n in range(20)])]), [units[0]])
    selection.add(dict(summary='',topics=[proposal('海洋观测',units[1],3,['SONAR',*[f'核心词{n}' for n in range(12)]])]), [units[1]])
    selection.add(dict(summary='',topics=[proposal('海洋观测',units[2],3,[' sonar ','SoNaR'])]), [units[2]])
    selected, terms = selection.select()
    assert len(terms)==10 and not any(w.startswith('次要词') for w in terms)
    assert terms['SONAR']=={units[1]['id'],units[2]['id']}
    assert len({search_term_key(w) for w in terms})==10
    assert all(search_term_key(w) in {search_term_key(x) for x in terms} for t in selected for w in t['keywords'])
    # Retaining/merging a word must not give the math topic new evidence permission.
    assert next(t for t in selected if t['name']=='数学')['unit_ids']=={units[0]['id']}


def test_persistence_caps_both_stores_even_if_selector_is_bypassed(tmp_path):
    store = Store(tmp_path/'live')
    doc_id,_ = store.import_file('synthetic.txt',b'synthetic terms','live')
    task = store.claim('worker')
    units = [evidence(n) for n in (1,2)]
    selected = [dict(proposal(name,unit),keywords={f'raw{n}' for n in range(40)},unit_ids={unit['id']})
                for name,unit in zip(('海洋观测','数学'),units)]
    terms = {f'词{n}':{units[n%2]['id']} for n in range(30)}
    store.apply_topic_selection(doc_id,selected,terms,'Synthetic',task['id'],'worker')
    detail = assert_bounded(store,doc_id)
    assert len(detail['search_terms'])==10
    assert store.search('raw39')==[] and store.search('词29')==[]
    assert {d['id'] for r in store.search('词9') for d in r['documents']}=={doc_id}
    assert detail['search_terms'][0]['unit_ids']==[units[0]['id']]


def test_incremental_topics_share_budget_and_new_support_does_not_expand_to_whole_topic(tmp_path):
    store = Store(tmp_path/'live')
    doc_id,_ = store.import_file('incremental.txt',b'incremental','live')
    topic = store.topic('海洋观测')
    store.link_topic(doc_id,topic['id'],[f'关键词{n}' for n in range(8)],['u1'])
    store.link_topic(doc_id,store.topic('数学')['id'],[f'关键词{n}' for n in range(8,18)],['math'])
    detail = assert_bounded(store,doc_id)
    assert len(detail['search_terms'])==10
    store.link_topic(doc_id,topic['id'],[],['unrelated'])
    store.link_topic(doc_id,topic['id'],[' 关键词0 '],['u2'])
    detail = assert_bounded(store,doc_id)
    assert next(t for t in detail['search_terms'] if t['term']=='关键词0')['unit_ids']==['u1','u2']
    second,_ = store.import_file('second.txt',b'second','live')
    store.link_topic(second,topic['id'],[f'另词{n}' for n in range(20)],['other'])
    assert len(assert_bounded(store,second)['search_terms'])==10
    assert {d['id'] for r in store.search('关键词9') for d in r['documents']}=={doc_id}


def test_existing_mixed_stores_are_limited_once_without_reading_evidence_or_changing_qa(tmp_path, monkeypatch):
    store = Store(tmp_path/'legacy')
    doc_id,_ = store.import_file('legacy.txt',b'legacy','live')
    topic = store.topic('海洋观测')
    candidate_id = uid()
    with store.connect() as db:
        db.execute('INSERT INTO document_topics VALUES(?,?,?,?)',
                   (doc_id,topic['id'],encode([f'旧词{n}' for n in range(16)]),encode(['whole-topic'])))
        db.executemany('INSERT INTO document_search_terms VALUES(?,?,?,?)',
                       [(doc_id,f'旧词{n}',topic['id'],encode([f'exact{n}'])) for n in range(8)])
        db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                   (candidate_id,doc_id,'u1',topic['id'],'{}','unchanged-hash','frozen-evidence','fp','approved','{}',0))
        db.execute('INSERT INTO qas VALUES(?,?,?)',(candidate_id,'unchanged formal snapshot',0))
        # Any evidence-body parsing would fail; this migration must use metadata only.
        db.execute("UPDATE documents SET units='not-json-evidence' WHERE id=?",(doc_id,))
        db.execute("DELETE FROM schema_migrations WHERE name='document_search_terms_limit_10_v1'")
    reopened = Store(store.root)
    summaries = reopened.documents(summary_only=True)
    assert sum(len(t['keywords']) for t in summaries[0]['topics'])==10
    with reopened.connect() as db:
        assert db.execute('SELECT row_json FROM qas').fetchone()[0]=='unchanged formal snapshot'
        assert db.execute('SELECT row_hash FROM candidates').fetchone()[0]=='unchanged-hash'
        rows = {r['term']:json.loads(r['unit_ids']) for r in db.execute('SELECT * FROM document_search_terms')}
        assert len(rows)==10 and rows['旧词0']==['exact0']
        history = json.loads(db.execute("SELECT detail FROM events WHERE kind='document_search_terms_limited'").fetchone()[0])
        assert len(json.loads(history['previous_topics'][0]['keywords']))==16
    monkeypatch.setattr(Store,'_limit_document_terms',staticmethod(lambda *args:pytest.fail('Migration repeated')))
    Store(store.root)


def test_deduplication_keeps_symbols_that_distinguish_real_terms():
    terms = limit_search_terms({'C':['c'], 'C++':['cpp'], '2.5 kHz':['low'], '25 kHz':['high']})
    assert len(terms)==4 and terms['C']=={'c'} and terms['C++']=={'cpp'}
