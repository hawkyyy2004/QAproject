"""Synthetic fixtures exercise paired generation, provenance, review and separate storage."""
import csv
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

from qaproject.app import create_app
from qaproject.glm import GLMClient, ProviderError
from qaproject.pipeline import Worker, CHECKS, BILINGUAL_CHECKS, approve, reject
from qaproject.settings import FIELDS, settings
from qaproject.store import Store, digest, encode


def pair(unit_id, n=0):
    return {'unit_id':unit_id,'topic':'海洋观测',
            'zh':{'input':f'合成观测 {n} 中温度是多少？','thinking':'合成记录给出温度15摄氏度。','output':'15摄氏度。'},
            'en':{'input':f'What was the temperature in synthetic observation {n}?',
                  'thinking':'The synthetic record gives a temperature of 15 degrees Celsius.','output':'15 degrees Celsius.'}}


@pytest.fixture
def setup_pipeline(tmp_path, monkeypatch):
    cfg = dict(settings(), target_per_document=5, min_per_document=5)
    monkeypatch.setattr('qaproject.pipeline.settings',lambda:cfg)
    app = create_app(tmp_path/'app',workers=False)
    store = app.state.stores['live']
    doc_id,_ = store.import_file('synthetic.txt','合成观测温度为15摄氏度，用于隔离流程测试。'.encode(),'live')
    calls = []
    behavior = {}

    def fake_call(self,instruction,payload,units=(),**kwargs):
        kind='analysis' if 'existing_topics' in payload else 'review' if 'candidates' in payload else 'generation'
        calls.append((kind,payload,units))
        if kind=='analysis':
            result={'summary':'隔离的合成观测。','topics':[{'name':'海洋观测','keywords':['温度'],'unit_ids':[u['id'] for u in units]}]}
        elif kind=='generation':
            start=len(payload['existing_questions'])//2
            result={'pairs':[pair(units[0]['id'],start+n) for n in range(payload['target'])],'skipped':[]}
        else:
            result={'checks':[dict(candidate_id=c['candidate_id'],note='中英含义、数值和范围均符合该证据。',
                                   **{k:True for k in CHECKS+BILINGUAL_CHECKS}) for c in payload['candidates']]}
        if behavior.get(kind):
            result=behavior[kind](result,payload,units)
        return result,{'model':'mock','seconds':0.001}

    monkeypatch.setattr(GLMClient,'call',fake_call)
    worker=Worker(store)
    return app,store,doc_id,worker,calls,behavior,cfg


def run(setup):
    _,store,doc_id,worker,*_=setup
    worker.process(store.claim(worker.name,document_id=doc_id))
    return store.document(doc_id)


def test_five_knowledge_points_are_ten_rows_with_same_number_of_calls(setup_pipeline):
    app,store,doc_id,_,calls,_,_=setup_pipeline
    doc=run(setup_pipeline)
    assert doc['status']=='waiting_review' and doc['bilingual_count']==5
    assert len(doc['candidates'])==10 and doc['qa_count']==doc['qa_pair_count']==0
    assert [k for k,_,_ in calls]==['analysis','generation','review','generation','review']
    assert [p['target'] for k,p,_ in calls if k=='generation']==[3,2]
    for pair_id in {c['pair_id'] for c in doc['candidates']}:
        versions=[c for c in doc['candidates'] if c['pair_id']==pair_id]
        assert {c['row']['language'] for c in versions}=={'zh','en'}
        assert len({(c['unit_id'],c['evidence_hash'],c['row']['source'],c['row']['image']) for c in versions})==1
        assert all(set(c['row'])==set(FIELDS) for c in versions)
        assert all(len(c['review']['binding']['candidate_hashes'])==2 for c in versions)
    for c in doc['candidates']:
        approve(store,c['id'],True,c['row_hash'])
    assert store.document(doc_id)['status']=='completed'
    assert store.document(doc_id)['qa_count']==10 and store.document(doc_id)['qa_pair_count']==5
    client=TestClient(app)
    all_rows=list(csv.DictReader(io.StringIO(client.get('/api/export').content.decode('utf-8-sig'))))
    assert len(all_rows)==10 and list(all_rows[0])==FIELDS
    for language in ('zh','en'):
        response=client.get('/api/export',params={'language':language,'document_id':doc_id,'topic_id':doc['topics'][0]['id']})
        rows=list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
        assert len(rows)==5 and {r['language'] for r in rows}=={language}
        assert 'QA-'+language+'.csv' in response.headers['content-disposition']
    assert client.get('/api/export?language=zh-en').status_code==400
    summary=client.get('/api/documents?summary_only=true').json()[0]
    assert summary['qa_count']==10 and summary['qa_pair_count']==5 and summary['bilingual_count']==5


@pytest.mark.parametrize('mutation',['missing_en','blank_answer','same_text','wrong_topic'])
def test_invalid_pair_never_leaves_one_language_in_database(setup_pipeline,mutation):
    _,store,_,_,_,behavior,_=setup_pipeline
    def invalid(result,payload,units):
        q=result['pairs'][0]
        if mutation=='missing_en': q.pop('en')
        if mutation=='blank_answer': q['en']['output']=''
        if mutation=='same_text': q['en']=dict(q['zh'])
        if mutation=='wrong_topic': q['topic']='不存在的主题'
        return dict(result,pairs=[q])
    behavior['generation']=invalid
    doc=run(setup_pipeline)
    assert doc['status']=='insufficient' and doc['bilingual_count']==0 and not doc['candidates']
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM qa_pairs').fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM events WHERE kind='candidate_invalid'").fetchone()[0]>0


@pytest.mark.parametrize('flag',['translation_consistent','language_correct','supported'])
def test_failed_translation_or_fact_review_blocks_both_versions(setup_pipeline,flag):
    _,store,_,_,_,behavior,_=setup_pipeline
    def invalid_review(result,*args):
        for check in result['checks']:check[flag]=False
        return result
    behavior['review']=invalid_review
    doc=run(setup_pipeline)
    assert all(c['status']=='needs_review' for c in doc['candidates'])
    for c in doc['candidates'][:2]:
        with pytest.raises(ValueError,match='审核'):approve(store,c['id'],True,c['row_hash'])
    assert store.document(doc['id'])['qa_count']==0


def test_changing_translation_invalidates_original_review_even_with_new_hash(setup_pipeline):
    _,store,_,_,_,_,_=setup_pipeline
    doc=run(setup_pipeline)
    zh,en=doc['candidates'][:2]
    changed=dict(en['row'],output='16 degrees Celsius.')
    with store.connect() as db:
        db.execute('UPDATE candidates SET row_json=?,row_hash=? WHERE id=?',(encode(changed),digest(changed),en['id']))
    with pytest.raises(ValueError,match='审核'):approve(store,zh['id'],True,zh['row_hash'])
    with pytest.raises(ValueError,match='审核'):approve(store,en['id'],True,digest(changed))


def test_partial_approval_counts_rows_and_complete_pairs_separately(setup_pipeline):
    _,store,_,_,_,_,_=setup_pipeline
    doc=run(setup_pipeline)
    first=doc['candidates'][0]
    approve(store,first['id'],True,first['row_hash'])
    approve(store,first['id'],True,first['row_hash'])
    partial=store.document(doc['id'])
    assert partial['qa_count']==1 and partial['qa_pair_count']==0
    for c in doc['candidates'][1:]:reject(store,c['id'],c['row_hash'])
    finished=store.document(doc['id'])
    assert finished['status']=='completed_under_min' and finished['qa_pair_count']==0
    assert '完整双语 0 组' in finished['error']


def test_pair_rereview_uses_only_its_evidence_and_never_regenerates(setup_pipeline):
    app,store,doc_id,worker,calls,_,_=setup_pipeline
    doc=run(setup_pipeline)
    client=TestClient(app)
    headers={'X-QA-Session':client.get('/api/bootstrap').json()['session']}
    target=doc['candidates'][0]
    url='/api/candidates/'+target['id']+'/rereview'
    body={'reviewed':False,'expected_hash':target['row_hash']}
    assert client.post(url,headers=headers,json=dict(body,expected_hash='stale')).status_code==400
    assert client.post(url,headers=headers,json=body).status_code==200
    assert client.post(url,headers=headers,json=body).status_code==400
    calls.clear()
    worker.process(store.claim(worker.name,document_id=doc_id))
    assert [k for k,_,_ in calls]==['review']
    assert len(calls[0][1]['candidates'])==1 and len(calls[0][1]['candidates'][0]['versions'])==2
    assert len(calls[0][2])==1
    assert len(store.document(doc_id)['candidates'])==10


def test_either_language_duplicate_rolls_back_whole_new_pair(setup_pipeline):
    _,store,_,_,calls,behavior,_=setup_pipeline
    def repeat_one_side(result,payload,units):
        for q in result['pairs']:q['en']=pair(units[0]['id'],0)['en']
        return result
    behavior['generation']=repeat_one_side
    doc=run(setup_pipeline)
    assert len(doc['candidates'])==2 and doc['bilingual_count']==1
    assert len({c['pair_id'] for c in doc['candidates']})==1


def test_legacy_migration_keeps_existing_rows_and_pairs_stable(setup_pipeline):
    _,store,doc_id,_,_,_,_=setup_pipeline
    doc=run(setup_pipeline)
    before=[(c['id'],c['pair_id'],c['row'],c['review']) for c in doc['candidates']]
    reopened=Store(store.root)
    assert before==[(c['id'],c['pair_id'],c['row'],c['review']) for c in reopened.document(doc_id)['candidates']]
    with reopened.connect() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0]==1


def test_visual_pair_reuses_actual_image_in_generation_and_review(tmp_path,monkeypatch):
    """Inspect the actual HTTP image bytes, rather than accepting a path-only mock."""
    import pymupdf
    from test_prototype import fake_provider
    app=create_app(tmp_path/'image',workers=False)
    store=app.state.stores['live']
    pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,40,40),False)
    pix.clear_with(255)
    doc_id,_=store.import_file('synthetic.png',pix.tobytes('png'),'live')
    seen=[]
    def provider(request):
        body=json.loads(request.content)
        content=body['messages'][1]['content']
        payload=json.loads(content[0]['text'])
        actual=[part['image_url']['url'] for part in content if part['type']=='image_url']
        assert actual and all(x.startswith('data:image/png;base64,') for x in actual)
        seen.append(('review' if 'candidates' in payload else 'analysis' if 'existing_topics' in payload else 'generation',actual))
        # Reuse the synthetic response generator with a text-only request wrapper.
        body['messages'][1]['content']=encode(payload)
        return fake_provider(httpx.Request('POST','https://fixture.invalid',content=json.dumps(body)))
    real_client=httpx.Client
    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:real_client(transport=httpx.MockTransport(provider)))
    monkeypatch.setattr('qaproject.glm.api_key',lambda:'test-placeholder')
    worker=Worker(store)
    worker.process(store.claim(worker.name))
    doc=store.document(doc_id)
    assert len(doc['candidates'])==2,doc['error']
    assert {c['row']['language'] for c in doc['candidates']}=={'zh','en'}
    assert len({c['row']['image'] for c in doc['candidates']})==1
    assert all(Path(c['row']['image']).is_file() for c in doc['candidates'])
    generated=next(images for kind,images in seen if kind=='generation')
    assert next(images for kind,images in seen if kind=='review')==generated


def test_english_evidence_can_produce_chinese_and_english(setup_pipeline):
    _,store,doc_id,_,_,_,_=setup_pipeline
    path=Path(store.document(doc_id)['path'])
    text=b'Synthetic temperature observation: the water temperature was 15 degrees Celsius.'
    # This is an isolated not-yet-prepared test source, with its identity updated before freezing.
    import hashlib
    path.write_bytes(text)
    with store.connect() as db:db.execute('UPDATE documents SET content_hash=? WHERE id=?',(hashlib.sha256(text).hexdigest(),doc_id))
    doc=run(setup_pipeline)
    assert doc['units'][0]['language']=='en' and doc['bilingual_count']==5
    assert {c['row']['language'] for c in doc['candidates']}=={'zh','en'}


def test_pre_bilingual_database_migrates_without_rewriting_legacy_records(tmp_path):
    from qaproject.demo import seed_demo
    store=Store(tmp_path/'legacy')
    seed_demo(store)
    before=[(d['id'],[(c['id'],c['row'],c['review']) for c in d['candidates']]) for d in store.documents()]
    with store.connect() as db:
        db.execute('DROP TABLE candidate_pairs')
        db.execute('DROP TABLE qa_pairs')
        db.execute('ALTER TABLE tasks DROP COLUMN review_candidate_id')
        db.execute('PRAGMA user_version=0')
    reopened=Store(store.root)
    after=[(d['id'],[(c['id'],c['row'],c['review']) for c in d['candidates']]) for d in reopened.documents()]
    assert before==after
    assert all(d['bilingual_count']==0 and d['legacy_count']==3 for d in reopened.documents(summary_only=True))


def test_missing_review_cannot_be_bypassed_by_manual_approval(setup_pipeline):
    _,store,_,_,_,_,_=setup_pipeline
    doc=run(setup_pipeline)
    target=doc['candidates'][0]
    with store.connect() as db:db.execute("UPDATE candidates SET review='{}' WHERE id=?",(target['id'],))
    with pytest.raises(ValueError,match='审核'):approve(store,target['id'],True,target['row_hash'])


def test_expired_worker_cannot_store_generated_pair_or_overwrite_new_status(setup_pipeline):
    _,store,doc_id,_,_,behavior,_=setup_pipeline
    def expire(result,*args):
        with store.connect() as db:
            db.execute("UPDATE tasks SET worker='new-worker',lease_until=lease_until+3600 WHERE document_id=?",(doc_id,))
            db.execute("UPDATE documents SET status='analyzing',progress='新worker处理中' WHERE id=?",(doc_id,))
        return result
    behavior['generation']=expire
    doc=run(setup_pipeline)
    assert not doc['candidates'] and doc['status']=='analyzing' and doc['progress']=='新worker处理中'


def test_explicit_rereview_gets_fresh_decision_and_keeps_previous_review(setup_pipeline):
    app,store,doc_id,worker,calls,behavior,cfg=setup_pipeline
    cfg['target_per_document']=cfg['min_per_document']=1
    def fail_translation(result,*args):
        result['checks'][0]['translation_consistent']=False
        return result
    behavior['review']=fail_translation
    doc=run(setup_pipeline)
    assert doc['candidates'][0]['status']=='needs_review'
    behavior.clear()
    client=TestClient(app)
    headers={'X-QA-Session':client.get('/api/bootstrap').json()['session']}
    target=doc['candidates'][0]
    for _ in range(2):
        calls.clear()
        assert client.post('/api/candidates/'+target['id']+'/rereview',headers=headers,
                           json={'reviewed':False,'expected_hash':target['row_hash']}).status_code==200
        worker.process(store.claim(worker.name,document_id=doc_id))
        assert [k for k,_,_ in calls]==['review']
    latest=store.document(doc_id)
    assert all(c['status']=='pending' for c in latest['candidates'])
    with store.connect() as db:
        history=[json.loads(r[0]) for r in db.execute("SELECT detail FROM events WHERE kind='review_replaced'")]
    assert any(not e['previous_review']['translation_consistent'] for e in history)
