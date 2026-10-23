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


def facts(units):
    return dict(summary='隔离观测内容。', facts=[dict(unit_id=u['id'],focus='温度观测方法与结果。',keywords=['温度']) for u in units])


def test_many_evidence_batches_generate_formal_topics_once_after_all_facts(tmp_path, monkeypatch):
    store = Store(tmp_path/'live')
    doc_id, _ = store.import_file('synthetic.txt', b'synthetic classification test', 'live')
    units = [evidence(n, '隔离观测证据。'*900) for n in range(1,13)]
    store.update_document(doc_id, units=units)
    seen, plans = [], []
    def call(self, instruction, payload, units=(), **kwargs):
        if payload.get('analysis_phase')=='facts':
            assert 'existing_topics' not in payload and 'topics' not in payload
            assert not store.document(doc_id)['topics']
            seen.extend(u['id'] for u in units)
            return facts(units), {}
        if 'existing_topics' in payload:
            assert len(seen)==12 and len(payload['document_facts'])==12
            assert len(payload['units'])==1  # First generation shares the single formal-topic request.
            plans.append(payload)
            return dict(summary='整篇海洋观测。', topics=[dict(proposal('海洋观测', units[0]),
                unit_ids=[u['unit_id'] for u in payload['document_facts']])],pairs=[],skipped=[]), {}
        return {'pairs':[], 'skipped':[]}, {}
    monkeypatch.setattr(GLMClient,'call',call)
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    doc = store.document(doc_id)
    assert doc['coverage']=='complete' and len(doc['topics'])==1, doc['error']
    assert set(seen)=={u['id'] for u in units} and len(plans)==1
    assert set(doc['topics'][0]['unit_ids'])==set(seen)
    assert {t['name'] for t in store.topics() if t['documents']}=={'海洋观测'}
    assert not any(t['name'].startswith('温度观测') for t in store.topics())


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
    assert len(selected)==2 and selected[0]['name']=='海洋观测'
    assert len(selected[0]['locations'])==1
    assert selected[0]['unit_ids']=={unit['id'], view['id']}


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
        if payload.get('analysis_phase')=='facts':
            n = int(units[0]['locator'].split('-')[1])
            calls.append(n)
            if n==2 and fail:
                raise ProviderError('测试超时。','timeout',{})
            return facts(units), {}
        if 'existing_topics' in payload:
            return dict(summary='整篇观测。',topics=[dict(proposal('海洋观测',units[0]),unit_ids=[f['unit_id'] for f in payload['document_facts']])],pairs=[],skipped=[]), {}
        return {'pairs':[],'skipped':[]}, {}

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
            return dict(summary='整篇直接概括。', topics=[proposal('海洋观测',unit)],
                        legacy_topic_map={t['id']:'海洋观测' for t in historical}), {}
        return {'pairs':[], 'skipped':[]}, {}

    monkeypatch.setattr(GLMClient, 'call', call)
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    doc = store.document(doc_id)
    assert len(doc['topics'])==1 and calls==['analysis']
    assert len([t for t in store.topics() if t['name'].startswith('历史领域')])==7  # Stable historical IDs.
    with store.connect() as db:
        event = json.loads(db.execute("SELECT detail FROM events WHERE kind='document_topics_selected'").fetchone()[0])
    assert len(event['previous'])==7


def test_more_than_five_model_labels_are_not_truncated_or_published(tmp_path,monkeypatch):
    store=Store(tmp_path/'live')
    doc_id,_=store.import_file('plan.txt',b'synthetic correction','live')
    calls=[]
    def call(self,instruction,payload,units=(),**kw):
        if 'existing_topics' in payload:
            calls.append(payload)
            if len(calls)==1:
                return dict(summary='错误输出。',topics=[proposal(f'过细方向{n}',units[0]) for n in range(8)]),{}
            assert 'automatic_regeneration' in payload
            return dict(summary='概括为海洋观测。',topics=[proposal('海洋观测',units[0])]),{}
        return {'pairs':[],'skipped':[]},{}
    monkeypatch.setattr(GLMClient,'call',call)
    worker=Worker(store)
    worker.process(store.claim(worker.name))
    doc=store.document(doc_id)
    assert [t['name'] for t in doc['topics']]==['海洋观测'],doc['error']
    assert len(calls)==2 and not any(t['name'].startswith('过细方向') for t in store.topics())


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
