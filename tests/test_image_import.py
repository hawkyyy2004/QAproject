"""Standalone images retain their bytes across import, model calls and preview."""
import base64
import csv
import io
import json
from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from qaproject.app import create_app
from qaproject.parsers import extract
from qaproject.pipeline import Worker
from qaproject.settings import FIELDS, settings
from qaproject.store import Store, encode
from test_bilingual import pair
from test_prototype import fake_provider


def image_bytes(extension, alpha=False):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200), alpha)
    pix.clear_with(128)
    # Non-default DPI caught the old image-to-page rescaling path.
    pix.set_dpi(144, 144)
    return pix.tobytes(extension)


@pytest.mark.parametrize('name,extension,alpha', [('image.PNG','png',False), ('transparent.png','png',True),
                                                  ('image.JPG','jpeg',False), ('image.jpeg','jpeg',False)])
def test_import_classify_generate_review_commit_preview_export(tmp_path, monkeypatch, name, extension, alpha):
    raw = image_bytes(extension, alpha)
    mime = 'image/png' if extension=='png' else 'image/jpeg'
    app = create_app(tmp_path/'app', workers=False)
    store = app.state.stores['live']
    seen = []
    client = TestClient(app)
    headers = {'x-qa-session':app.state.session}
    response = client.post('/api/imports', headers=headers, files={'files':(name,raw,mime)})
    doc_id = response.json()[0]['id']
    assert response.json()[0]['created'] is True
    assert client.post('/api/imports',headers=headers,files={'files':(name,raw,mime)}).json()[0] == {
        'id':doc_id,'name':name,'created':False}

    def forbid_render(*args, **kwargs):
        raise AssertionError('PNG/JPEG import must never render or transcode')
    monkeypatch.setattr(pymupdf.Page,'get_pixmap',forbid_render)

    def provider(request):
        body = json.loads(request.content)
        content = body['messages'][1]['content']
        payload = json.loads(content[0]['text'])
        actual = [part['image_url']['url'] for part in content if part['type']=='image_url']
        assert len(actual)==1 and actual[0].startswith('data:'+mime+';base64,')
        assert base64.b64decode(actual[0].split(',',1)[1])==raw
        kind = 'analysis' if 'existing_topics' in payload else 'review' if 'candidates' in payload else 'generation'
        seen.append(kind)
        if kind=='generation':
            assert payload['evidence_mode']=='visual' and payload['target']==2
            result = {'pairs':[pair(payload['units'][0]['id'],n) for n in range(2)],'skipped':[]}
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':encode(result)}}]})
        body['messages'][1]['content'] = encode(payload)
        return fake_provider(httpx.Request('POST','https://fixture.invalid',content=json.dumps(body)))

    original_client = httpx.Client
    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:original_client(transport=httpx.MockTransport(provider)))
    monkeypatch.setattr('qaproject.glm.api_key',lambda:'test-placeholder')
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    doc = store.document(doc_id)
    assert seen==['analysis','generation','review'],doc['error']
    assert doc['status']=='waiting_review' and doc['bilingual_count']==2
    assert doc['pages']==0 and len(doc['units'])==len(doc['assets'])==1
    unit = doc['units'][0]
    assert unit['locator']=='image-0001' and unit['evidence_kind']=='standalone_image'
    assert (unit['width'],unit['height'])==(300,200)
    assert Path(unit['image']).read_bytes()==Path(doc['path']).read_bytes()==raw
    assert '目标 5 组' in doc['error']  # Do not hide the normal document quantity shortfall.
    preview = client.get('/api/assets/'+unit['asset_id'])
    assert preview.content==raw and preview.headers['content-type']==mime
    result = client.post('/api/documents/'+doc_id+'/approve',headers=headers)
    assert result.status_code==200,result.text
    assert store.document(doc_id)['qa_count']==4
    assert seen==['analysis','generation','review']  # Commit adds no model request.
    for c in doc['candidates']:
        assert c['row']['source']==doc['path']+'|image-0001|海洋观测'
        assert c['row']['image']==unit['image'] and set(c['row'])==set(FIELDS)
    # Re-entering the existing task after completion reuses evidence and all responses.
    with store.connect() as db:
        db.execute("UPDATE tasks SET status='queued' WHERE document_id=?",(doc_id,))
    worker.process(store.claim(worker.name))
    assert seen==['analysis','generation','review']
    exported = client.get('/api/export?document_id='+doc_id)
    assert exported.status_code==200
    rows = list(csv.DictReader(io.StringIO(exported.content.decode('utf-8-sig'))))
    assert len(rows)==4 and list(rows[0])==FIELDS
    assert {r['language'] for r in rows}=={'zh','en'}


@pytest.mark.parametrize('name,raw', [('bad.png',b'not an image'), ('fake.png',b'%PDF-1.7'), ('truncated.png',b'\x89PNG\r\n\x1a\n'),
                                     ('wrong.png',image_bytes('jpeg'))])
def test_invalid_or_mislabeled_images_never_register_evidence(tmp_path, name, raw):
    store = Store(tmp_path/'data')
    doc_id,_ = store.import_file(name,raw,'live')
    with pytest.raises(ValueError,match='图片'):
        extract(store,store.document(doc_id),settings())
    assert not store.document(doc_id)['assets']


def test_pixel_limit_uses_actual_dimensions_and_blocks_before_saving(tmp_path,monkeypatch):
    store = Store(tmp_path/'data')
    doc_id,_ = store.import_file('large.png',image_bytes('png'),'live')
    monkeypatch.setattr(pymupdf.Page,'get_image_info',lambda page:[dict(width=8000,height=6000)])
    with pytest.raises(ValueError,match='像素过大'):
        extract(store,store.document(doc_id),settings())
    assert not store.document(doc_id)['assets']


def test_image_size_limit_does_not_silently_compress(tmp_path):
    store = Store(tmp_path/'data')
    raw = image_bytes('png') + b'\0'*(5*1024*1024)
    doc_id,_ = store.import_file('large.png',raw,'live')
    with pytest.raises(ValueError,match='5MB'):
        extract(store,store.document(doc_id),settings())
    assert not store.document(doc_id)['assets']


def test_failed_image_save_stops_before_any_model_call(tmp_path,monkeypatch):
    store = Store(tmp_path/'data')
    doc_id,_ = store.import_file('image.png',image_bytes('png'),'live')
    original = Path.write_bytes
    def fail_asset(path, data):
        if 'assets' in path.parts:
            raise OSError('synthetic disk failure')
        return original(path,data)
    monkeypatch.setattr(Path,'write_bytes',fail_asset)
    calls = []
    def forbidden(*args,**kwargs):
        calls.append(args)
        raise AssertionError('Unsaved images cannot enter model generation')
    monkeypatch.setattr('qaproject.pipeline.GLMClient.call',forbidden)
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    doc = store.document(doc_id)
    assert doc['status']=='failed' and not doc['assets'] and not doc['candidates'] and not calls
