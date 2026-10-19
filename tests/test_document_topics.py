"""Document-level limits, whole-document ranking, provenance and request size."""
import json

import httpx
import pytest

from qaproject.glm import GLMClient, ProviderError
from qaproject.pipeline import Worker
from qaproject.settings import settings
from qaproject.store import Store, digest, encode, uid
from qaproject.taxonomy import TopicSelection


def evidence(n, text='合成观测温度为15摄氏度。'):
    unit = dict(id=f'long-persistent-document-identifier:p{n}:s1', text=text, image='',
                asset_id='', locator=f'page-{n:04d}-seg-0001', language='zh')
    return dict(unit, hash=digest(unit))


def proposal(name, unit, importance=2, keywords=None):
    return dict(name=name, definition='隔离测试的稳定领域定义。', new_reason='现有主题不能覆盖的隔离测试领域。',
                importance=importance, keywords=keywords or [name+'细词'], unit_ids=[unit['id']])


def test_twenty_four_batch_topics_become_five_without_extra_model_call(tmp_path, monkeypatch):
    store = Store(tmp_path/'live')
    doc_id, _ = store.import_file('synthetic.txt', b'synthetic classification test', 'live')
    units = [evidence(n, '隔离观测证据。'*900) for n in range(1, 13)]
    store.update_document(doc_id, units=units)
    calls = []

    def call(self, instruction, payload, units=(), **kwargs):
        if 'existing_topics' in payload:
            n = int(units[0]['locator'].split('-')[1])
            calls.append(n)
            # The core topic appears late: do not keep the first five proposals.
            main = '海洋观测' if n >= 11 else f'次要领域{n}'
            result = dict(summary='隔离观测。', topics=[proposal(main, units[0], 3 if n >= 11 else 1),
                                                      proposal(f'测试领域{n}', units[0], 2)])
        else:
            result = {'pairs':[], 'skipped':['仅验证分类，不补造考点。']}
        return result, {'seconds':0, 'model':'mock'}

    monkeypatch.setattr(GLMClient, 'call', call)
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    doc = store.document(doc_id)
    assert doc['coverage']=='complete' and len(doc['topics'])==5, doc['error']
    assert '海洋观测' in {t['name'] for t in doc['topics']}
    assert sorted(calls)==list(range(1, 13))  # Whole-document evidence, no consolidation call.
    assert len([t for t in store.topics() if t['documents']])==5
    assert not any(t['name']=='测试领域12' for t in store.topics())  # No orphan formal topics.
    terms = {t['term']: t['unit_ids'] for t in doc['search_terms']}
    assert len(terms)==10
    assert terms['海洋观测细词']==sorted([u['id'] for u in units[-2:]])
    assert '测试领域12细词' not in terms
    # Fine-term support never becomes formal-topic generation permission.
    fine_hit = store.search('海洋观测细词')
    assert {d['id'] for r in fine_hit for d in r['documents']}=={doc_id}
    second, _ = store.import_file('other.txt', b'unrelated evidence', 'live')
    store.link_topic(second, store.topic('海洋观测')['id'], ['另一检索方向'], [])
    assert {d['id'] for r in store.search('海洋观测细词') for d in r['documents']}=={doc_id}
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='document_topics_selected'").fetchone()[0]==1


def test_limit_is_per_document_not_whole_library_and_updates_are_allowed(tmp_path):
    store = Store(tmp_path/'live')
    for n in range(2):
        doc_id, _ = store.import_file(f'{n}.txt', str(n).encode(), 'live')
        for m in range(5):
            topic = store.topic(f'独立领域{n}{m}', '独立测试领域。', '测试定义无法归入其他领域。')
            store.link_topic(doc_id, topic['id'], ['真实细词'], ['evidence'])
        store.link_topic(doc_id, topic['id'], ['补充细词'], ['later'])
        sixth = store.topic('第六领域', '第六个测试领域。', '仅用于验证拒绝超限。')
        with pytest.raises(ValueError, match='最多5个'):
            store.link_topic(doc_id, sixth['id'], [], [])
        assert len(store.document(doc_id)['topics'])==5
    assert len([t for t in store.topics() if t['documents']])==10


def test_alias_reuse_and_coverage_count_real_locations_not_views(tmp_path):
    store = Store(tmp_path/'live')
    unit = evidence(1)
    view = dict(unit, id=unit['id']+':text', companion_of=unit['id'])
    other = evidence(2)
    selection = TopicSelection(store.topics(), [unit, view, other])
    selection.add(dict(summary='', topics=[proposal('海洋监测', unit)]), [unit])
    selection.add(dict(summary='', topics=[proposal('海洋观测', view)]), [view])
    selection.add(dict(summary='', topics=[proposal('数学', other, 3)]), [other])
    selected, _ = selection.select()
    assert len(selected)==2 and selected[0]['name']=='数学'
    assert selected[1]['name']=='海洋观测' and len(selected[1]['locations'])==1
    assert selected[1]['unit_ids']=={unit['id'], view['id']}


def test_empty_classification_is_failure_not_success_or_placeholder(tmp_path, monkeypatch):
    store = Store(tmp_path/'live')
    doc_id, _ = store.import_file('empty-topics.txt', '不应伪造分类。'.encode(), 'live')
    monkeypatch.setattr(GLMClient, 'call', lambda *a, **kw: ({'summary':'无法归类。', 'topics':[]}, {}))
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    doc = store.document(doc_id)
    assert doc['status']=='failed' and doc['coverage']!='complete'
    assert '未识别到' in doc['error'] and not doc['candidates'] and not doc['topics']


def test_failed_analysis_never_publishes_partial_topics_and_resumes_cached_batches(tmp_path, monkeypatch):
    store = Store(tmp_path/'live')
    doc_id, _ = store.import_file('failure.txt', b'isolated failure', 'live')
    units = [evidence(n, 'Synthetic evidence '*500) for n in (1, 2)]
    store.update_document(doc_id, units=units)
    calls = []
    fail = True

    def call(self, instruction, payload, units=(), **kw):
        if 'existing_topics' not in payload:
            return {'pairs':[], 'skipped':[]}, {}
        n = int(units[0]['locator'].split('-')[1])
        calls.append(n)
        if n==2 and fail:
            raise ProviderError('测试超时。', 'timeout', {})
        return dict(summary='隔离摘要。', topics=[proposal('海洋观测', units[0])]), {}

    monkeypatch.setattr(GLMClient, 'call', call)
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    assert store.document(doc_id)['status']=='failed'
    assert not store.document(doc_id)['topics']
    assert not any(t['documents'] for t in store.topics())
    fail = False
    with store.connect() as db:
        db.execute("UPDATE tasks SET status='queued' WHERE document_id=?", (doc_id,))
    worker.process(store.claim(worker.name))
    assert len(store.document(doc_id)['topics'])==1
    assert calls.count(1)==1 and calls.count(2)==2


def test_stale_lease_cannot_register_topics_or_replace_associations(tmp_path):
    store = Store(tmp_path/'live')
    doc_id, _ = store.import_file('stale.txt', b'stale synthetic', 'live')
    task = store.claim('old-worker')
    unit = evidence(1)
    selection = TopicSelection(store.topics(), [unit])
    selection.add(dict(summary='', topics=[proposal('尚未登记领域', unit)]), [unit])
    selected, terms = selection.select()
    with store.connect() as db:
        db.execute('UPDATE tasks SET lease_until=0 WHERE id=?', (task['id'],))
    with pytest.raises(ValueError, match='租约'):
        store.apply_topic_selection(doc_id, selected, terms, '', task['id'], 'old-worker')
    assert not any(t['name']=='尚未登记领域' for t in store.topics())
    assert store.document(doc_id)['coverage']=='pending'


def test_history_is_not_reclassified_without_preserving_candidate_references(tmp_path):
    store = Store(tmp_path/'live')
    doc_id, _ = store.import_file('history.txt', b'history', 'live')
    task = store.claim('worker')
    topic = store.topic('数学')
    store.link_topic(doc_id, topic['id'], [], [])
    with store.connect() as db:
        db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                   (uid(),doc_id,'unit',topic['id'],'{}','unchanged','evidence','fingerprint','rejected','{}',0))
    unit = evidence(1)
    selection = TopicSelection(store.topics(), [unit])
    selection.add(dict(summary='', topics=[proposal('全新领域', unit)]), [unit])
    selected, terms = selection.select()
    with pytest.raises(ValueError, match='历史候选'):
        store.apply_topic_selection(doc_id, selected, terms, '', task['id'], 'worker')
    assert [t['name'] for t in store.document(doc_id)['topics']]==['数学']
    assert not any(t['name']=='全新领域' for t in store.topics())
    with pytest.raises(ValueError, match='超过5个'):
        selection.select([f'history{n}' for n in range(6)])


def test_request_local_ids_reduce_payload_and_restore_real_evidence_refs(monkeypatch):
    unit = evidence(1, 'Original text with unit_id=u1 is unchanged.')
    payload = {'units':[{k:unit[k] for k in ('id','text','language')}],
               'topics':[{'name':'海洋观测','unit_ids':[unit['id']]}]}
    seen = []

    def response(request):
        body = json.loads(request.content)
        sent = json.loads(body['messages'][1]['content'])
        assert sent['units'][0]['id']=='u1'
        assert sent['units'][0]['text']==unit['text']
        assert sent['topics'][0]['unit_ids']==['u1']
        seen.append(len(body['messages'][1]['content']))
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop','message':{'content':encode(
            {'pairs':[{'unit_id':'u1','zh':{'input':'Keep literal u1'}}]})}}]})

    real_client = httpx.Client
    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:real_client(transport=httpx.MockTransport(response)))
    result, _ = GLMClient(settings(), key='test-placeholder').call('Return JSON', payload, [unit])
    assert result['pairs'][0]['unit_id']==unit['id']
    assert result['pairs'][0]['zh']['input']=='Keep literal u1'
    assert payload['units'][0]['id']==unit['id']  # Frozen caller data is untouched.
    assert seen[0] < len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))


def test_generation_omits_unapproved_units_but_analysis_keeps_them(tmp_path, monkeypatch):
    store = Store(tmp_path/'live')
    doc_id, _ = store.import_file('scope.txt', b'synthetic generation scope', 'live')
    units = [evidence(1), evidence(2)]
    store.update_document(doc_id, units=units)
    analyzed, generated = [], []

    def call(self, instruction, payload, units=(), **kwargs):
        if 'existing_topics' in payload:
            analyzed.extend(u['id'] for u in units)
            result = dict(summary='隔离支持范围。', topics=[proposal('海洋观测', units[0])])
        else:
            generated.extend(u['id'] for u in units)
            assert [u['id'] for u in payload['units']]==[units[0]['id']]
            result = {'pairs':[], 'skipped':[]}
        return result, {}

    monkeypatch.setattr(GLMClient, 'call', call)
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    assert analyzed==[u['id'] for u in units]
    assert generated==[units[0]['id']]


def test_legacy_over_limit_classification_is_reselected_not_reused(tmp_path, monkeypatch):
    store = Store(tmp_path/'live')
    doc_id, _ = store.import_file('legacy.txt', b'synthetic historical classification', 'live')
    unit = evidence(1)
    store.update_document(doc_id, units=[unit], coverage='complete')
    historical = [store.topic(f'历史领域{n}', '测试历史领域。', '旧版登记。') for n in range(7)]
    # Recreate pre-limit history; runtime writes always use the guarded API.
    with store.connect() as db:
        db.executemany('INSERT INTO document_topics VALUES(?,?,?,?)',
                       [(doc_id,t['id'],'[]',encode([unit['id']])) for t in historical])
    calls = []

    def call(self, instruction, payload, units=(), **kwargs):
        if 'existing_topics' in payload:
            calls.append('analysis')
            return dict(summary='按全文重新筛选。', topics=[proposal(t['name'], unit) for t in historical]), {}
        return {'pairs':[], 'skipped':[]}, {}

    monkeypatch.setattr(GLMClient, 'call', call)
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    doc = store.document(doc_id)
    assert len(doc['topics'])==5 and calls==['analysis']
    assert len([t for t in store.topics() if t['name'].startswith('历史领域')])==7  # Stable historical IDs.
    with store.connect() as db:
        event = json.loads(db.execute("SELECT detail FROM events WHERE kind='document_topics_selected'").fetchone()[0])
    assert len(event['previous'])==7


def test_first_page_priority_uses_real_source_order_for_appended_text_view(tmp_path):
    store = Store(tmp_path/'live')
    units = [evidence(n) for n in range(1, 7)]
    view = dict(units[0], id=units[0]['id']+':text', companion_of=units[0]['id'])
    selection = TopicSelection(store.topics(), [*units, view])
    # Text companions are appended during parsing, but still belong to page 1.
    for n, unit in enumerate([*units[1:], view], 1):
        selection.add(dict(summary='', topics=[proposal(f'测试领域{n}', unit)]), [unit])
    selected, _ = selection.select()
    assert selected[0]['name']=='测试领域6'
    assert '测试领域5' not in {t['name'] for t in selected}


def test_invalid_new_topic_is_deferred_without_blocking_valid_core_or_faking_reason(tmp_path):
    store = Store(tmp_path/'live')
    unit = evidence(1)
    selection = TopicSelection(store.topics(), [unit])
    invalid = dict(proposal('新方向', unit), new_reason='')
    selection.add(dict(summary='', topics=[invalid, proposal('海洋观测', unit)]), [unit])
    selected, terms = selection.select()
    assert [t['name'] for t in selected]==['海洋观测']
    assert terms['新方向']=={unit['id']} and terms['新方向细词']=={unit['id']}
    assert selection.deferred[0]['name']=='新方向'
    only_invalid = TopicSelection(store.topics(), [unit])
    only_invalid.add(dict(summary='', topics=[invalid]), [unit])
    with pytest.raises(ValueError, match='未识别到'):
        only_invalid.select()
