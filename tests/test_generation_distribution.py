"""Text cannot consume visual coverage; real image bytes still reach both calls."""
import base64
import json

import httpx
import pymupdf
import pytest

from qaproject.generation_policy import generation_limits, modality_counts
from qaproject.parsers import text_alternatives
from qaproject.pipeline import Worker, CHECKS, BILINGUAL_CHECKS, approve
from qaproject.settings import settings
from qaproject.store import Store, digest, uid
from test_bilingual import pair


@pytest.fixture
def mixed(tmp_path, monkeypatch):
    store = Store(tmp_path/'mixed')
    doc_id, _ = store.import_file('mixed.txt', b'isolated evidence fixture', 'live')
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=200, height=200)
        page.draw_rect((30, 30, 80, 150), fill=(.1, .4, .8))
        raw = page.get_pixmap().tobytes('png')
    units = []
    for n in range(22):
        image, asset = '', ''
        if n < 13:
            path = store.root/'assets/images'/f'page-{n+1}.png'
            path.write_bytes(raw)
            image = str(path)
            asset = store.add_asset(doc_id, path, n+1, 'Complete synthetic page')
        text = (f'Figure {n+1}. Synthetic temperature and salinity observations, with calibrated sensors. '
                'Only the specified location and period are covered. ')*2
        u = dict(id=f'{doc_id}:p{n+1}:s1',text=text,image=image,asset_id=asset,
                 page=n+1,seg=1,locator=f'page-{n+1:04d}-seg-0001',language='en')
        units.append(dict(u,hash=digest(u)))
    units = text_alternatives(units)
    assert len(units)==35
    store.update_document(doc_id,units=units,pages=22)
    calls, behavior = [], {}
    monkeypatch.setattr('qaproject.glm.api_key',lambda:'isolated-test-placeholder')
    original = httpx.Client

    def provider(request):
        body = json.loads(request.content)
        content = body['messages'][1]['content']
        payload = json.loads(content[0]['text'] if isinstance(content,list) else content)
        images = [p['image_url']['url'] for p in content if p['type']=='image_url'] if isinstance(content,list) else []
        kind = 'analysis' if 'existing_topics' in payload else 'review' if 'candidates' in payload else 'generation'
        if images:
            assert all(base64.b64decode(x.split(',',1)[1])==raw for x in images)
            assert body['response_format']=={'type':'json_object'}
        calls.append((kind,payload,images))
        if kind=='analysis':
            result=dict(summary='隔离海洋观测。',topics=[dict(name='海洋观测',keywords=['温度'],
                        unit_ids=[u['id'] for u in payload['units']])])
        elif kind=='generation':
            if payload['evidence_mode']=='visual':
                assert images and len(images)==1
                assert '纯正文扫描' in body['messages'][0]['content']
                if behavior.get('fail_visual_once'):
                    behavior.pop('fail_visual_once')
                    raise httpx.ReadTimeout('isolated interruption')
                if behavior.get('invalid_visual'):
                    return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{invalid'}}]})
            else:
                assert not images
            number = 0 if images and behavior.get('no_figures') else payload['target']
            start = len(payload['existing_questions'])//2
            result=dict(pairs=[pair(payload['units'][0]['id'],start+n) for n in range(number)],
                        skipped=['No usable figures on these pages.'] if not number else [])
        else:
            if any(v['qa']['image'] for c in payload['candidates'] for v in c['versions']):
                assert images
            result=dict(checks=[dict(candidate_id=c['candidate_id'],note='中英事实及范围符合证据。',
                        **{k:True for k in CHECKS+BILINGUAL_CHECKS}) for c in payload['candidates']])
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(result)}}]})

    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:original(transport=httpx.MockTransport(provider)))
    return store, doc_id, units, calls, behavior


def process(store, doc_id):
    worker = Worker(store)
    worker.process(store.claim(worker.name,document_id=doc_id))
    doc = store.document(doc_id)
    assert doc['status']=='waiting_review',doc['error']
    return doc


def enqueue(store, doc_id):
    with store.connect() as db:
        db.execute('INSERT INTO tasks(id,document_id,status) VALUES(?,?,?)',(uid(),doc_id,'queued'))


def test_35_units_produce_three_text_two_visual_in_same_five_calls(mixed):
    store, doc_id, units, calls, _ = mixed
    before = digest(units)
    doc = process(store,doc_id)
    assert modality_counts(doc['candidates'])==dict(text=3,visual=2)
    assert [k for k,_,_ in calls]==['analysis','generation','review','generation','review']
    assert [(p['evidence_mode'],p['target']) for k,p,_ in calls if k=='generation']==[('text',3),('visual',2)]
    assert digest(doc['units'])==before
    assert doc['bilingual_count']==5 and len(doc['candidates'])==10
    for c in doc['candidates']:
        approve(store,c['id'],True,c['row_hash'])
    assert store.document(doc_id)['qa_pair_count']==5


def test_unusable_images_are_bounded_and_text_fills_five(mixed):
    store, doc_id, _, calls, behavior = mixed
    behavior['no_figures']=True
    doc = process(store,doc_id)
    assert modality_counts(doc['candidates'])==dict(text=5,visual=0)
    generations = [p for k,p,_ in calls if k=='generation']
    assert [p['evidence_mode'] for p in generations]==['text','visual','visual','text']
    assert [p['target'] for p in generations]==[3,2,2,2]
    assert not doc['error']


def test_existing_five_text_pairs_leave_room_for_two_visual_without_rewriting(mixed,monkeypatch):
    store, doc_id, _, calls, _ = mixed
    with monkeypatch.context() as patch:
        patch.setattr('qaproject.pipeline.generation_phases',lambda text,visual,limits:[('text',5,False,text)])
        old = process(store,doc_id)
    before = {c['id']:(c['row_hash'],c['evidence_hash'],c['review']) for c in old['candidates']}
    calls.clear()
    enqueue(store,doc_id)
    doc = process(store,doc_id)
    assert modality_counts(doc['candidates'])==dict(text=5,visual=2)
    assert [k for k,_,_ in calls]==['generation','review']
    assert all(before[c['id']]==(c['row_hash'],c['evidence_hash'],c['review']) for c in doc['candidates'] if c['id'] in before)
    enqueue(store,doc_id)
    calls.clear()
    assert process(store,doc_id)['bilingual_count']==7
    assert not calls


def test_resume_after_image_timeout_keeps_three_text_pairs_and_sends_images(mixed):
    store, doc_id, _, calls, behavior = mixed
    behavior['fail_visual_once']=True
    worker=Worker(store)
    worker.process(store.claim(worker.name))
    failed=store.document(doc_id)
    assert failed['status']=='failed' and modality_counts(failed['candidates'])==dict(text=3,visual=0)
    old={c['id']:c['row_hash'] for c in failed['candidates']}
    enqueue(store,doc_id)
    calls.clear()
    doc=process(store,doc_id)
    assert modality_counts(doc['candidates'])==dict(text=3,visual=2)
    assert all(c['row_hash']==old[c['id']] for c in doc['candidates'] if c['id'] in old)
    assert [k for k,_,_ in calls]==['generation','review']


def test_default_limits_and_explicit_small_validation_target():
    assert generation_limits(settings())==dict(target=5,maximum=7,text=5,visual=2)
    assert generation_limits(dict(settings(),target_per_document=1))==dict(target=1,maximum=1,text=1,visual=1)


def test_visual_format_failure_is_bounded_and_text_still_delivers_five(mixed):
    store, doc_id, _, calls, behavior = mixed
    behavior['invalid_visual']=True
    doc=process(store,doc_id)
    assert modality_counts(doc['candidates'])==dict(text=5,visual=0)
    assert len([p for k,p,_ in calls if k=='generation' and p['evidence_mode']=='visual'])==2
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM events WHERE kind='model_error'").fetchone()[0]==2
        assert db.execute("SELECT count(*) FROM events WHERE kind='optional_visual_unavailable'").fetchone()[0]==1
