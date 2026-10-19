"""FACTS coverage failures must not poison caches or discard completed work."""
from collections import Counter
import json

import pytest

from qaproject.glm import FACTS, GLMClient, ProviderError
from qaproject.pipeline import compact_units, content_facts, CHECKS, BILINGUAL_CHECKS, approve
from qaproject.settings import settings
from qaproject.store import digest, encode
from test_request_recovery import setup_document, facts, analysis, events, process
from test_bilingual import pair


def cache_file(store, group, *, legacy=False):
    config = settings()
    request = dict(kind='analysis', prompt=FACTS,
                   payload={'analysis_phase':'facts', 'units':compact_units(group)},
                   units=group, config={k:config.get(k) for k in
                   ('base_url','text_model','vision_model','reasoning_effort','json_mode')}, review_attempt=None)
    if legacy:
        request['max_tokens'] = 2048
    return store.root/'runs'/(digest(request)+'.json')


def finish(payload, group):
    if 'document_facts' in payload:
        ids = [ref for f in payload['document_facts'] for ref in f.get('unit_ids',[f.get('unit_id')])]
        assert len(ids)==33 and len(set(ids))==33
        return dict(analysis([{'id':ref} for ref in ids]), pairs=[], skipped=[]), {}
    return {'pairs':[], 'skipped':[]}, {}


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'unknown', 'empty', 'format', 'focus'])
def test_invalid_facts_split_before_cache_and_cover_every_unit(tmp_path, monkeypatch, fault):
    store, doc_id, units = setup_document(tmp_path, count=33)
    calls = []
    def call(self, instruction, payload, group=(), **kwargs):
        if payload.get('analysis_phase')!='facts':
            return finish(payload, group)
        calls.append(tuple(u['id'] for u in group))
        result = {'summary':'隔离观测。', 'facts':[{'unit_ids':[u['id'] for u in group],
                  'focus':'温度观测方法与结果。', 'keywords':['温度']}]}
        if len(group)==32:
            if fault=='missing':
                result['facts'][0]['unit_ids'].pop()
            elif fault=='duplicate':
                result['facts'].append(result['facts'][0])
            elif fault=='unknown':
                result['facts'][0]['unit_ids'][0]='not-sent'
            elif fault=='empty':
                result['facts']=[]
            elif fault=='format':
                result=[]
            else:
                result['facts'][0]['focus']=' '
        return result, {'seconds':0.01, 'finish_reason':'stop', 'usage':{'completion_tokens':40}}
    monkeypatch.setattr(GLMClient, 'call', call)
    doc = process(store, doc_id)
    assert doc['coverage']=='complete' and doc['status']=='insufficient', doc['error']
    assert sorted(map(len,calls))==[1,16,16,32]
    assert not cache_file(store, units[:32]).exists()
    for group in (units[:16],units[16:32],units[32:]):
        content_facts(json.loads(cache_file(store,group).read_text())['result'], group)
    error, = events(store,'model_error')
    assert error['code']=='facts' and error['error_category']=='facts_validation'
    assert error['seconds']==0.01 and error['finish_reason']=='stop'
    assert len(events(store,'request_split'))==1


@pytest.mark.parametrize('legacy', [False, True])
def test_old_bad_cache_is_quarantined_and_good_batch_is_reused(tmp_path, monkeypatch, legacy):
    store, doc_id, units = setup_document(tmp_path, count=33)
    good = cache_file(store, units[:32])
    original = encode({'result':facts(units[:32]), 'stats':{}})
    good.write_text(original)
    bad = cache_file(store, units[32:], legacy=legacy)
    rejected = encode({'result':{'facts':[]}, 'stats':{}})
    bad.write_text(rejected)
    calls = []
    def call(self, instruction, payload, group=(), **kwargs):
        if payload.get('analysis_phase')=='facts':
            calls.append([u['id'] for u in group])
            return facts(group), {}
        return finish(payload, group)
    monkeypatch.setattr(GLMClient, 'call', call)
    doc = process(store, doc_id)
    assert doc['coverage']=='complete', doc['error']
    assert calls==[[units[32]['id']]] and good.read_text()==original
    quarantined = list(bad.parent.glob(bad.stem+'.invalid-*.json'))
    assert len(quarantined)==1 and quarantined[0].read_text()==rejected
    assert len(events(store,'cache_invalid'))==1
    content_facts(json.loads(cache_file(store,units[32:]).read_text())['result'], units[32:])


def test_resume_after_split_timeout_reuses_successful_children(tmp_path, monkeypatch):
    store, doc_id, units = setup_document(tmp_path, count=33)
    calls = Counter()
    fail = True
    def call(self, instruction, payload, group=(), **kwargs):
        if payload.get('analysis_phase')!='facts':
            return finish(payload, group)
        key = tuple(u['id'] for u in group)
        calls[key] += 1
        if len(group)==32:
            return {'facts':[]}, {}
        if key==tuple(u['id'] for u in units[16:32]) and fail:
            raise ProviderError('隔离超时。','timeout',{'seconds':0.01})
        return facts(group), {}
    monkeypatch.setattr(GLMClient, 'call', call)
    doc = process(store, doc_id)
    assert doc['status']=='failed' and not doc['topics']
    fail = False
    with store.connect() as db:
        db.execute("UPDATE tasks SET status='queued' WHERE document_id=?",(doc_id,))
    doc = process(store, doc_id)
    assert doc['coverage']=='complete', doc['error']
    assert calls[tuple(u['id'] for u in units[:32])]==1
    assert calls[tuple(u['id'] for u in units[:16])]==1
    assert calls[tuple(u['id'] for u in units[16:32])]==2
    assert calls[(units[32]['id'],)]==1


def test_persistent_invalid_single_unit_stops_after_one_retry_without_cache(tmp_path, monkeypatch):
    store, doc_id, units = setup_document(tmp_path, count=33)
    cache_file(store,units[:32]).write_text(encode({'result':facts(units[:32]),'stats':{}}))
    calls = []
    def call(self, instruction, payload, group=(), **kwargs):
        calls.append(group)
        return {'facts':[]}, {'seconds':0.01,'finish_reason':'stop'}
    monkeypatch.setattr(GLMClient,'call',call)
    doc = process(store, doc_id)
    assert doc['status']=='failed' and doc['coverage']!='complete' and not doc['topics']
    assert len(calls)==2 and all(g==units[32:] for g in calls)
    assert not cache_file(store,units[32:]).exists()
    assert len(events(store,'facts_request_retry'))==1


def test_facts_recovery_continues_through_bilingual_review_and_controlled_commit(tmp_path, monkeypatch):
    store, doc_id, units = setup_document(tmp_path, count=33)
    def call(self, instruction, payload, group=(), **kwargs):
        if payload.get('analysis_phase')=='facts':
            return (facts(group[:-1]) if len(group)==32 else facts(group)), {}
        if 'document_facts' in payload:
            result, stats = finish(payload,group)
            result['pairs']=[pair(payload['units'][0]['id'], n) for n in range(payload['target'])]
            return result, stats
        if 'candidates' in payload:
            return {'checks':[dict(candidate_id=c['candidate_id'], note='中英等义且符合实际证据。',
                                   **{k:True for k in CHECKS+BILINGUAL_CHECKS})
                              for c in payload['candidates']]}, {}
        return {'pairs':[pair(group[0]['id'],3+n) for n in range(payload['target'])], 'skipped':[]}, {}
    monkeypatch.setattr(GLMClient,'call',call)
    doc = process(store,doc_id)
    assert doc['status']=='waiting_review' and doc['bilingual_count']==5, doc['error']
    assert len(events(store,'request_split'))==1
    for c in doc['candidates']:
        approve(store,c['id'],True,c['row_hash'])
    assert store.document(doc_id)['qa_count']==10
