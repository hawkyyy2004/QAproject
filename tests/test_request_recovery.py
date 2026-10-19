"""Fault-injection checks: full evidence, bounded recovery and durable child caches."""
import json
from pathlib import Path

import httpx
import pytest

from qaproject.glm import GLMClient, ProviderError
from qaproject.pipeline import Worker, CHECKS, BILINGUAL_CHECKS
from qaproject.settings import settings
from qaproject.store import Store, digest, encode


def setup_document(tmp_path, images=0, count=2):
    store = Store(tmp_path/'workspace')
    doc_id, _ = store.import_file('synthetic.txt', '隔离观测：温度为15摄氏度。'.encode(), 'live')
    units = []
    for n in range(count):
        image = tmp_path/f'{n}.png'
        if n<images:
            image.write_bytes(b'actual-image-fixture-'+bytes([n]))
        unit = dict(id=f'persistent-unit-{n}', text='隔离海域温度为15摄氏度。', image=str(image) if n<images else '',
                    language='zh', locator=f'page-{n+1:04d}-seg-0001', asset_id='')
        if n<images:
            unit['asset_id']=store.add_asset(doc_id,image,n+1,'isolated image fixture')
        units.append(dict(unit, hash=digest(unit)))
    store.update_document(doc_id, units=units)
    return store, doc_id, units


def analysis(units):
    return {'summary':'隔离温度观测。', 'topics':[{'name':'海洋观测','unit_ids':[u['id'] for u in units],
            'keywords':['温度'], 'importance':3}]}


def events(store, kind):
    with store.connect() as db:
        return [json.loads(x[0]) for x in db.execute('SELECT detail FROM events WHERE kind=? ORDER BY created',(kind,))]


def process(store, doc_id):
    worker = Worker(store)
    worker.process(store.claim(worker.name,document_id=doc_id))
    return store.document(doc_id)


def test_analysis_expands_budget_without_repeating_complete_batches(tmp_path, monkeypatch):
    store, doc_id, units = setup_document(tmp_path)
    calls = []
    def call(self, instruction, payload, group=(), **kw):
        if 'existing_topics' not in payload:
            return {'pairs':[], 'skipped':[]}, {}
        calls.append((kw['max_tokens'],[u['id'] for u in group]))
        if kw['max_tokens']<16384:
            raise ProviderError('截断','length',{'finish_reason':'length', 'usage':{'completion_tokens':8192}})
        return analysis(group), {}
    monkeypatch.setattr(GLMClient,'call',call)
    doc = process(store,doc_id)
    assert doc['coverage']=='complete' and doc['status']=='insufficient', doc['error']
    assert calls==[(8192,[u['id'] for u in units]),(16384,[u['id'] for u in units])]
    assert len(events(store,'model_error'))==1 and len(events(store,'token_budget_retry'))==1
    assert len(events(store,'model_call'))==2  # One complete analysis and one empty generation.


def test_multi_image_400_splits_and_resume_reuses_children(tmp_path, monkeypatch):
    store, doc_id, units = setup_document(tmp_path, images=2)
    seen = []
    fail_second = True
    def call(self, instruction, payload, group=(), **kw):
        if 'existing_topics' not in payload:
            return {'pairs':[], 'skipped':[]}, {}
        seen.append([u['id'] for u in group])
        assert all(Path(u['image']).read_bytes().startswith(b'actual-image') for u in group)
        if len(group)>1:
            raise ProviderError('拒绝','http',{'http_status':400,'images':2})
        if group[0]['id']==units[1]['id'] and fail_second:
            raise ProviderError('超时','timeout',{})
        return analysis(group), {}
    monkeypatch.setattr(GLMClient,'call',call)
    doc = process(store,doc_id)
    assert doc['status']=='failed' and doc['coverage']!='complete' and not doc['topics']
    fail_second = False
    with store.connect() as db:
        db.execute("UPDATE tasks SET status='queued' WHERE document_id=?",(doc_id,))
    doc = process(store,doc_id)
    assert doc['coverage']=='complete' and doc['status']=='insufficient', doc['error']
    assert set(doc['topics'][0]['unit_ids'])=={u['id'] for u in units}
    assert seen==[[u['id'] for u in units],[units[0]['id']],[units[1]['id']],[units[1]['id']]]
    assert len(events(store,'request_split'))==1


@pytest.mark.parametrize('code,stats,expected_calls',[
    ('http',{'http_status':400,'images':1},2),
    ('http',{'http_status':401,'images':1},1),
    ('http',{'http_status':400,'error_category':'quota','images':1},1),
    ('http',{'http_status':400,'error_category':'output_limit','images':1},1),
    ('timeout',{'images':1},1),
])
def test_single_image_retries_are_finite_and_auth_quota_timeout_stop(tmp_path, monkeypatch, code, stats, expected_calls):
    store,doc_id,_ = setup_document(tmp_path,images=1,count=1)
    calls = []
    def call(*args,**kwargs):
        calls.append(1)
        raise ProviderError('已脱敏错误',code,stats)
    monkeypatch.setattr(GLMClient,'call',call)
    monkeypatch.setattr('qaproject.pipeline.time.sleep',lambda _:None)
    doc = process(store,doc_id)
    assert doc['status']=='failed' and len(calls)==expected_calls
    assert len(events(store,'model_error'))==expected_calls


@pytest.mark.parametrize('code',['length','json','context'])
def test_analysis_splits_complete_units_and_exhausted_leaf_never_publishes(tmp_path,monkeypatch,code):
    store,doc_id,units = setup_document(tmp_path)
    cfg = dict(settings(),analysis_max_tokens=8192,max_output_tokens_ceiling=8192)
    monkeypatch.setattr('qaproject.pipeline.settings',lambda:cfg)
    seen = []
    def call(self,instruction,payload,group=(),**kw):
        if 'existing_topics' not in payload:
            return {'pairs':[],'skipped':[]},{}
        seen.append(group)
        if len(group)>1:
            raise ProviderError('测试恢复',code,{})
        return analysis(group),{}
    monkeypatch.setattr(GLMClient,'call',call)
    doc=process(store,doc_id)
    assert doc['coverage']=='complete' and len(seen)==3
    assert seen[0]==units and seen[1]==units[:1] and seen[2]==units[1:]


def test_unrecoverable_length_is_bounded_and_never_saved_as_complete(tmp_path,monkeypatch):
    store,doc_id,_=setup_document(tmp_path,count=1)
    calls=[]
    def call(self,instruction,payload,group=(),**kw):
        calls.append(kw['max_tokens'])
        raise ProviderError('截断','length',{'finish_reason':'length'})
    monkeypatch.setattr(GLMClient,'call',call)
    doc=process(store,doc_id)
    assert calls==[8192,16384,32768,65536,131072]
    assert doc['status']=='failed' and doc['coverage']!='complete' and not doc['topics']
    assert not list((store.root/'runs').glob('*.json'))


def test_legacy_complete_analysis_cache_survives_budget_change(tmp_path,monkeypatch):
    store,doc_id,units=setup_document(tmp_path)
    cfg=settings()
    from qaproject.glm import ANALYZE
    from qaproject.pipeline import compact_units
    request={'kind':'analysis','prompt':ANALYZE,'payload':{'units':compact_units(units),
             'existing_topics':[{k:t[k] for k in ('name','definition','aliases')} for t in store.topics()]},
             'units':units,'config':{k:cfg.get(k) for k in ('base_url','text_model','vision_model','reasoning_effort','json_mode')},
             'max_tokens':2048,'review_attempt':None}
    (store.root/'runs'/(digest(request)+'.json')).write_text(encode({'result':analysis(units),'stats':{}}))
    def call(self,instruction,payload,group=(),**kw):
        assert 'existing_topics' not in payload, 'Complete legacy analysis should be reused'
        return {'pairs':[],'skipped':[]},{}
    monkeypatch.setattr(GLMClient,'call',call)
    assert process(store,doc_id)['coverage']=='complete'
    assert events(store,'cache_hit')[0]['stage']=='analysis'


@pytest.mark.parametrize('failing_stage',['generation','review'])
def test_visual_generation_and_review_split_keep_bilingual_binding(tmp_path,monkeypatch,failing_stage):
    store,doc_id,units=setup_document(tmp_path,images=2)
    # Input has enough text that analysis uses text; generation still tests true image units.
    cfg=dict(settings(),target_per_document=2,min_per_document=2)
    monkeypatch.setattr('qaproject.pipeline.settings',lambda:cfg)
    seen=[]
    def call(self,instruction,payload,group=(),**kw):
        stage='analysis' if 'existing_topics' in payload else ('review' if 'candidates' in payload else 'generation')
        seen.append((stage,len(group)))
        if stage=='analysis':
            return analysis(group),{}
        if stage==failing_stage and len(group)>1:
            raise ProviderError('多图拒绝','http',{'http_status':400,'images':2})
        if stage=='review':
            assert all(c['unit_id'] in {u['id'] for u in group} for c in payload['candidates'])
            if failing_stage=='review' and len(group)==1:
                assert len(payload['compare_questions'])==2  # Other child's bilingual point still checked for duplicates.
            return {'checks':[dict(candidate_id=c['candidate_id'],note='范围、单位和双语均与证据一致。',
                                  **{k:True for k in CHECKS+BILINGUAL_CHECKS}) for c in payload['candidates']]},{}
        pairs=[]
        for u in group[:payload['target']]:
            pairs.append(dict(unit_id=u['id'],topic='海洋观测',
                zh=dict(input=f'隔离海域{u["id"]}的温度是多少？',thinking='证据为15摄氏度。',output='15摄氏度。'),
                en=dict(input=f'What is the temperature in synthetic area {u["id"]}?',thinking='Evidence gives 15 degrees Celsius.',output='15 degrees Celsius.')))
        return {'pairs':pairs,'skipped':[]},{}
    monkeypatch.setattr(GLMClient,'call',call)
    doc=process(store,doc_id)
    assert doc['status']=='waiting_review' and doc['bilingual_count']==2,doc['error']
    assert len(doc['candidates'])==4 and doc['qa_count']==0
    assert all(c['status']=='pending' and c['review']['binding']['evidence_hash']==c['evidence_hash'] for c in doc['candidates'])
    assert (failing_stage,2) in seen
    assert len(events(store,'request_split'))==1


def test_http_error_category_never_saves_echoed_body(tmp_path,monkeypatch):
    real_client=httpx.Client
    def respond(request):
        return httpx.Response(400,json={'error':{'message':'image count rejected; private echoed key and input'}})
    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:real_client(transport=httpx.MockTransport(respond)))
    with pytest.raises(ProviderError) as caught:
        GLMClient(settings(),key='fixture-key').call('test',{})
    assert caught.value.stats['error_category']=='image_input'
    assert 'private echoed' not in str(caught.value)+encode(caught.value.stats)


def test_http_400_fallback_transmits_each_original_image_and_restores_ids(tmp_path,monkeypatch):
    import base64
    store,doc_id,units=setup_document(tmp_path,images=2)
    seen=[]
    real_client=httpx.Client
    def respond(request):
        body=json.loads(request.content)
        content=body['messages'][1]['content']
        payload=json.loads(content[0]['text'])
        images=[base64.b64decode(c['image_url']['url'].split(',',1)[1]) for c in content if c['type']=='image_url']
        seen.append(images)
        if len(images)==2:
            return httpx.Response(400,json={'error':{'message':'images: 2 rejected; echoed private input'}})
        if 'existing_topics' in payload:
            result=analysis(payload['units'])
        else:
            result={'pairs':[],'skipped':[]}
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':encode(result)}}],
                                        'usage':{'total_tokens':30}})
    monkeypatch.setattr('qaproject.glm.api_key',lambda:'test-placeholder')
    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:real_client(transport=httpx.MockTransport(respond)))
    doc=process(store,doc_id)
    originals=[Path(u['image']).read_bytes() for u in units]
    assert doc['coverage']=='complete' and doc['status']=='insufficient',doc['error']
    assert seen[:3]==[originals,originals[:1],originals[1:]]
    assert set(doc['topics'][0]['unit_ids'])=={u['id'] for u in units}
    assert events(store,'model_error')[0]['error_category']=='image_input'
    assert 'echoed private' not in encode(events(store,'model_error'))


def test_single_image_transient_400_recovers_without_losing_page(tmp_path,monkeypatch):
    store,doc_id,units=setup_document(tmp_path,images=1,count=1)
    calls=[]
    def call(self,instruction,payload,group=(),**kw):
        if 'existing_topics' not in payload:
            return {'pairs':[],'skipped':[]},{}
        calls.append(group)
        if len(calls)==1:
            raise ProviderError('单图偶发拒绝','http',{'http_status':400,'images':1})
        return analysis(group),{}
    monkeypatch.setattr(GLMClient,'call',call)
    monkeypatch.setattr('qaproject.pipeline.time.sleep',lambda _:None)
    doc=process(store,doc_id)
    assert doc['coverage']=='complete' and calls==[units,units]
    assert len(events(store,'image_request_retry'))==1
