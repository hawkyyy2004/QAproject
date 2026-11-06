"""A topic plan that cites evidence never sent must not poison the cache."""
import json

import pytest

from qaproject.glm import ANALYZE, GLMClient
from qaproject.pipeline import response_cacheable, topic_plan_supported
from test_request_recovery import setup_document, facts, process


def test_topic_plan_with_invented_ids_is_not_cacheable():
    group = [{'id':'doc:p0:s1'}]
    invented = {'summary':'水下试验场地。','topics':[
        {'name':'水下试验场地','unit_ids':['doc:p0:s1','u2','u3','u4']}]}
    assert not topic_plan_supported(invented, group)
    assert not response_cacheable('topic_plan', ANALYZE, invented, group)
    valid = {'summary':'水下试验场地。','topics':[{'name':'水下试验场地','unit_ids':['doc:p0:s1']}]}
    assert topic_plan_supported(valid, group)
    assert response_cacheable('topic_plan', ANALYZE, valid, group)
    empty = {'summary':'无知识。','topics':[]}
    assert not response_cacheable('topic_plan', ANALYZE, empty, group)


def test_rejected_plan_never_reaches_the_cache(tmp_path, monkeypatch):
    store, doc_id, units = setup_document(tmp_path, count=2)
    calls = []
    def call(self, instruction, payload, group=(), **kw):
        if payload.get('analysis_phase')=='facts':
            return facts(group), {}
        if 'document_facts' in payload or payload.get('analysis_phase')=='document':
            calls.append(list(group[0]) if group else [])
            return {'summary':'水下试验场地。','topics':[{'name':'水下试验场地','definition':'试验水池',
                    'new_reason':'独立检索价值','importance':2,'keywords':['水池'],
                    'unit_ids':[units[0]['id'],'u2']}]}, {}
        return {'pairs':[], 'skipped':[]}, {}
    monkeypatch.setattr(GLMClient,'call',call)
    doc = process(store, doc_id)
    assert doc['status']=='failed' and doc['error']=='主题缺少当前证据支持。', doc['error']
    # Three real attempts, each reaching the model instead of a cached replay.
    assert len(calls)==3
    cached = [json.loads(path.read_text(encoding='utf-8'))['result']
              for path in (store.root/'runs').glob('*.json')]
    assert not any(isinstance(item,dict) and 'topics' in item for item in cached)
