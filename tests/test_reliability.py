import json

from fastapi.testclient import TestClient
import httpx
import pytest

from qaproject.app import create_app
from qaproject.glm import GLMClient, ProviderError
from qaproject.pipeline import Worker, CHECKS, BILINGUAL_CHECKS, review_note_flags
from qaproject.settings import settings
from qaproject.parsers import text_alternatives
from qaproject.store import Store, digest


def test_provider_error_keeps_safe_diagnostics_and_low_effort(monkeypatch):
    requests = []

    def response(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices':[{'finish_reason':'length', 'message':{'content':'private raw answer'}}],
                                         'usage':{'completion_tokens':2048}})

    real_client = httpx.Client
    monkeypatch.setattr('qaproject.glm.httpx.Client', lambda **kw:real_client(transport=httpx.MockTransport(response)))
    with pytest.raises(ProviderError) as caught:
        GLMClient(settings(), key='secret-fixture').call('Return JSON', {'test':True}, max_tokens=2048)
    error = caught.value
    assert error.code=='length' and error.stats['finish_reason']=='length'
    assert 'seconds' in error.stats
    assert requests[0]['reasoning_effort']=='low'
    assert requests[0]['response_format']=={'type':'json_object'}
    assert requests[0]['max_tokens']==2048
    assert 'secret-fixture' not in str(error)+json.dumps(error.stats)
    assert 'private raw answer' not in str(error)+json.dumps(error.stats)


def test_contradictory_review_note_is_not_accepted():
    assert review_note_flags('该公式超出本unit文本。')
    assert review_note_flags('地点未在本unit提供。')
    assert not review_note_flags('数值、对象及范围均与原文一致。')


def test_json_error_keeps_position_without_exposing_raw_response(monkeypatch):
    raw='{"pairs":[private raw answer}'
    real_client=httpx.Client
    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:real_client(transport=httpx.MockTransport(
        lambda request:httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':raw}}]}))))
    with pytest.raises(ProviderError) as caught:
        GLMClient(settings(),key='test-placeholder').call('Return JSON',{})
    error=caught.value
    assert error.code=='json' and error.stats['json_error']['offset']>0
    assert error.stats['json_error']['line']==1
    assert raw not in str(error)+json.dumps(error.stats)


def test_text_view_preserves_frozen_image_and_true_source():
    unit = {'id':'p1','text':'这是具有明确观测对象和单位的真实文字证据。'*10,
            'image':'fixture.png','asset_id':'image1','locator':'page-0001-seg-0001','language':'zh'}
    unit['hash'] = digest(unit)
    views = text_alternatives([unit])
    assert views[0]==unit and views[0]['hash']==unit['hash']
    assert len(views)==2 and views[1]['image']==''
    assert views[1]['text']==unit['text'] and views[1]['locator']==unit['locator']
    assert views[1]['companion_of']==unit['id']
    assert text_alternatives(views)==views
    scan = dict(unit,id='scan',text='Scan page: use actual visual evidence.')
    assert text_alternatives([scan])==[scan]


def test_expired_task_reopens_with_matching_visible_status(tmp_path):
    store = Store(tmp_path/'recovery')
    doc_id,_ = store.import_file('a.txt',b'evidence','live')
    task = store.claim('stopped-worker',document_id=doc_id)
    store.update_document(doc_id,status='analyzing',summary='Retained summary')
    with store.connect() as db:
        db.execute('UPDATE tasks SET lease_until=0 WHERE id=?',(task['id'],))
    reopened = Store(store.root)
    doc = reopened.document(doc_id)
    assert doc['status']=='queued' and doc['summary']=='Retained summary'
    assert reopened.claim('new-worker',document_id=doc_id)['id']==task['id']


def test_failed_review_resumes_at_five_without_repeating_generation(tmp_path, monkeypatch):
    app = create_app(tmp_path/'app', workers=False)
    store = app.state.stores['live']
    doc_id, _ = store.import_file('sample.txt', '演示海域 A 的 2025 年 1 月温度为 15 摄氏度。'.encode(), 'live')
    calls = {'analysis':0, 'generation':0, 'review':0}

    def fake_call(self, instruction, payload, units=(), **kwargs):
        if 'existing_topics' in payload:
            calls['analysis'] += 1
            result = {'summary':'有明确范围的温度观测示例。','topics':[{'name':'海洋观测','keywords':['温度'],
                     'unit_ids':[u['id'] for u in units]}]}
        elif 'candidates' in payload:
            calls['review'] += 1
            if calls['review']==1:
                raise ProviderError('接口超时，候选保留。','timeout',{'seconds':0.01,'finish_reason':'unknown'})
            result = {'checks':[dict(candidate_id=c['candidate_id'],note='对照证据与范围及双语。',**{k:True for k in CHECKS+BILINGUAL_CHECKS})
                                for c in payload['candidates']]}
        else:
            calls['generation'] += 1
            assert payload['target']<=3
            offset = len(payload['existing_questions'])//2
            result = {'pairs':[{'unit_id':units[0]['id'],'topic':'海洋观测',
                             'zh':{'input':f'演示海域 A 在 2025 年 1 月的温度是多少（测试考点 {offset+n}）？',
                                   'thinking':'示例记录温度为15摄氏度。','output':'15摄氏度。'},
                             'en':{'input':f'What was the temperature in synthetic sea area A in January 2025 (test point {offset+n})?',
                                   'thinking':'The synthetic record gives 15 degrees Celsius.','output':'15 degrees Celsius.'}}
                             for n in range(payload['target'])], 'skipped':[]}
        return result, {'seconds':0.01, 'model':'mock'}

    monkeypatch.setattr(GLMClient,'call',fake_call)
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    failed = store.document(doc_id)
    assert failed['status']=='failed' and len(failed['candidates'])==6 and failed['bilingual_count']==3
    assert all(not c['review'] and c['status']=='needs_review' for c in failed['candidates'])
    assert 'reviewing' in failed['timings']

    client = TestClient(app)
    token = client.get('/api/bootstrap').json()['session']
    assert client.post('/api/documents/'+doc_id+'/retry',headers={'X-QA-Session':token}).status_code==200
    with store.connect() as db:
        previous = json.loads(db.execute("SELECT detail FROM events WHERE kind='retry_requested'").fetchone()[0])
    assert previous['previous_error']==failed['error']
    worker.process(store.claim(worker.name))
    completed = store.document(doc_id)
    assert completed['status']=='waiting_review'
    assert len(completed['candidates'])==10 and completed['qa_count']==0 and completed['bilingual_count']==5
    assert all(c['review'] and c['status']=='pending' for c in completed['candidates'])
    assert calls=={'analysis':1,'generation':2,'review':3}
    assert set(completed['timings'])=={'parsing','analyzing','generating','reviewing'}
