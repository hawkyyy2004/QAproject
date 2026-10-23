import csv
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

from qaproject.app import create_app
from qaproject.pipeline import Worker, approve, validate_row, reject, quality_flags
from qaproject.settings import FIELDS, source_ref
from qaproject.store import Store, encode, digest, normalized, uid, CORE_TOPICS


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path/'app',seed=True,workers=False)


def test_workspace_isolation_and_small_term_search(app):
    client=TestClient(app)
    demo=client.get('/api/documents?mode=demo').json()
    assert len(demo)==4
    assert sum(d['qa_count'] for d in demo)==12
    assert client.get('/api/documents?mode=live').json()==[]
    math=client.get('/api/search',params={'mode':'demo','q':'数学'}).json()
    assert len(math)==1 and len(math[0]['documents'])==2
    narrow=client.get('/api/search',params={'mode':'demo','q':'圆锥曲线'}).json()
    assert len(narrow)==1 and len(narrow[0]['documents'])==1
    assert '椭圆' in narrow[0]['documents'][0]['name']
    assert len(app.state.stores['demo'].topics())==len(CORE_TOPICS)  # fine terms did not create topics


def test_default_application_has_no_synthetic_records(tmp_path):
    app=create_app(tmp_path/'empty',workers=False)
    client=TestClient(app)
    assert client.get('/api/documents').json()==[]
    assert client.get('/api/documents?mode=demo').status_code==400
    assert client.get('/api/search?q=数学').json()==[]
    assert sum(t['documents'] for t in client.get('/api/topics').json())==0


def test_export_is_exact_six_fields_and_saved_images(app):
    client=TestClient(app)
    response=client.get('/api/export?mode=demo')
    rows=list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
    assert len(rows)==12
    for row in rows:
        assert list(row)==FIELDS
        assert len(row['source'].split('|'))==3
        assert 'page-01-seg-01' in row['source']
        assert Path(row['image']).is_file()
    d=client.get('/api/documents?mode=demo').json()[0]
    image=client.get('/api/assets/'+d['assets'][0]['id']+'?mode=demo')
    assert image.status_code==200 and image.content.startswith(b'\x89PNG')


def test_upload_session_duplicate_and_name_safety(app):
    client=TestClient(app)
    assert client.post('/api/imports?mode=live',files={'files':('a.txt',b'test')}).status_code==403
    session=client.get('/api/bootstrap').json()['session']
    headers={'X-QA-Session':session}
    first=client.post('/api/imports?mode=live',headers=headers,files={'files':('../../../../a.txt','真实测试证据'.encode())}).json()[0]
    second=client.post('/api/imports?mode=live',headers=headers,files={'files':('renamed.txt','真实测试证据'.encode())}).json()[0]
    assert first['created'] and not second['created'] and first['id']==second['id']
    doc=app.state.stores['live'].document(first['id'])
    assert Path(doc['path']).is_relative_to(app.state.stores['live'].root/'originals')
    assert Path(doc['path']).name=='a.txt'
    assert 'API_KEY' not in json.dumps(client.get('/api/bootstrap').json())


def fake_provider(request):
    body=json.loads(request.content)
    instruction=body['messages'][0]['content']
    payload=json.loads(body['messages'][1]['content'])
    if 'existing_topics' in payload:
        result={'summary':'变量温度的测量示例，明确数值与单位。','topics':[{'name':'海洋观测','keywords':['温度'],'unit_ids':[u['id'] for u in payload['units']]}]}
    elif 'candidates' in payload:
        result={'checks':[{'candidate_id':c['candidate_id'],'supported':True,'independent':True,'scope_correct':True,
            'knowledge_value':True,'distinct':True,'topic_correct':True,'language_correct':True,
            'translation_consistent':True,'note':'逐项对照变量、时期与单位及双语等义。'} for c in payload['candidates']]}
    else:
        unit=payload['units'][0]
        result={'pairs':[{'unit_id':unit['id'],'topic':'海洋观测',
            'zh':{'input':'在演示海域 A 的 2025 年 1 月观测示例中，温度是多少？','thinking':'示例给出温度为 15 摄氏度。','output':'15 摄氏度。'},
            'en':{'input':'What was the temperature in synthetic sea area A in January 2025?',
                  'thinking':'The synthetic record gives a temperature of 15 degrees Celsius.','output':'15 degrees Celsius.'}}],
            'skipped':['仅一个可靠考点，不补齐数量。']}
    return httpx.Response(200,json={'id':'mock-request','choices':[{'finish_reason':'stop','message':{'content':encode(result)}}],'usage':{'total_tokens':100}})


def test_pipeline_review_and_idempotent_commit(tmp_path,monkeypatch):
    import qaproject.glm as glm
    monkeypatch.setattr(glm,'api_key',lambda:'test-placeholder')
    original_client=httpx.Client
    monkeypatch.setattr(glm.httpx,'Client',lambda **kw:original_client(transport=httpx.MockTransport(fake_provider)))
    store=Store(tmp_path/'live')
    doc_id,_=store.import_file('sample.txt','演示海域 A 在 2025 年 1 月的温度为 15 摄氏度。'.encode(),'live')
    worker=Worker(store)
    task=store.claim(worker.name)
    worker.process(task)
    doc=store.document(doc_id)
    assert doc['status']=='waiting_review',doc['error']
    assert len(doc['candidates'])==2 and doc['bilingual_count']==1 and doc['qa_count']==0
    c=doc['candidates'][0]
    with pytest.raises(ValueError):approve(store,c['id'],False,c['row_hash'])
    with pytest.raises(ValueError):approve(store,c['id'],True,'wrong-hash')
    approve(store,c['id'],True,c['row_hash'])
    approve(store,c['id'],True,c['row_hash'])
    assert store.document(doc_id)['qa_count']==1
    with store.connect() as db:assert db.execute('SELECT status FROM tasks').fetchone()[0]=='succeeded'


def test_frozen_image_hash_rejects_modified_asset(app):
    store=app.state.stores['demo']
    d=store.documents()[0]
    c=d['candidates'][0]
    Path(c['row']['image']).write_bytes(b'modified')
    with pytest.raises(ValueError,match='图片内容'):
        approve(store,c['id'],True,c['row_hash'])


def test_unknown_topic_cannot_be_created_without_scope(tmp_path):
    store=Store(tmp_path/'data')
    with pytest.raises(ValueError):store.topic('三角函数')
    assert store.topic('数学研究')['name']=='数学'
    assert len(store.topics())==len(CORE_TOPICS)


def test_stale_worker_cannot_write_task(tmp_path):
    store=Store(tmp_path/'data')
    doc,_=store.import_file('a.txt',b'evidence','live')
    task=store.claim('worker-a')
    with store.connect() as db:db.execute('UPDATE tasks SET lease_until=0 WHERE id=?',(task['id'],))
    second=store.claim('worker-b')
    assert second['id']==task['id']
    with pytest.raises(ValueError):store.task_update(task['id'],'worker-a','done','succeeded')


def test_illegal_source_rejected(app):
    d=app.state.stores['demo'].documents()[0]
    c=d['candidates'][0]
    bad=dict(c['row'],source='/other.pdf|page-01-seg-01|数学')
    with pytest.raises(ValueError,match='来源'):
        validate_row(bad,d['units'][0],d['topics'][0],d['path'])


def test_low_value_flags_and_rejection(app):
    assert quality_flags({'input':'根据图1的柱状图，其纵轴的单位是什么？'})
    store=app.state.stores['demo']
    d=store.documents()[0]
    c=d['candidates'][0]
    with pytest.raises(ValueError,match='已入库'):reject(store,c['id'],c['row_hash'])
