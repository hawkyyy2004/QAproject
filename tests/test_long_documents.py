from pathlib import Path

from fastapi.testclient import TestClient
import pymupdf

from qaproject.app import create_app
from qaproject.parsers import extract
from qaproject.pipeline import batches
from qaproject.settings import settings
from qaproject.store import Store


def test_long_pdf_keeps_last_page_and_visual_evidence(tmp_path):
    store = Store(tmp_path/'long')
    with pymupdf.open() as pdf:
        for n in range(120):
            page = pdf.new_page()
            page.insert_textbox(pymupdf.Rect(40,40,550,90), f'Observation {n+1}: water temperature was 15 degrees Celsius, '
                                'measured at the same depth and time with calibrated instruments.')
            if n >= 118:
                for x in (40, 90, 140):
                    page.draw_rect(pymupdf.Rect(x, 100, x+30, 200))
        raw = pdf.tobytes()
    doc_id, _ = store.import_file('long.pdf', raw, 'live')
    progress = []
    cfg = settings()
    assert cfg['max_pages'] is None
    units, pages = extract(store, store.document(doc_id), cfg,
                           progress=lambda done,total: progress.append((done,total)))
    assert pages == 120
    last_visual = next(u for u in units if u['page']==120 and u['image'])
    assert last_visual['locator'] == 'page-120-seg-01'
    assert 'Observation 120' in last_visual['text']
    assert len([u for u in units if u['image']]) == 2
    assert Path(last_visual['image']).is_file()
    text_view = next(u for u in units if u.get('companion_of')==last_visual['id'])
    assert text_view['locator']==last_visual['locator'] and not text_view['image']
    assert progress[0] == (0, 120) and progress[-1] == (120, 120)
    store.update_document(doc_id, units=units, pages=pages)
    groups = list(batches(units))
    assert [u['id'] for g in groups for u in g] == [u['id'] for u in units]
    assert all(len(g) <= 32 and sum(bool(u['image']) for u in g) <= 2 for g in groups)


def test_short_units_cannot_create_unbounded_model_batch():
    units = [{'id':str(n), 'text':'x', 'image':''} for n in range(2000)]
    groups = list(batches(units))
    assert max(map(len, groups)) == 32
    assert sum(map(len, groups)) == 2000


def test_list_and_search_omit_evidence_and_keep_counts(tmp_path, monkeypatch):
    app = create_app(tmp_path/'app', workers=False)
    store = app.state.stores['live']
    doc_id, _ = store.import_file('large.txt', b'private evidence', 'live')
    units = [{'id':f'u{n}', 'text':'Evidence '*2000} for n in range(100)]
    store.update_document(doc_id, units=units, pages=100, summary='Summary '*200)
    topic = store.topic('海洋观测')
    store.link_topic(doc_id, topic['id'], ['温度'], ['u99'])

    def forbid_detail(*args):
        raise AssertionError('List/search must not load full document detail')

    monkeypatch.setattr(store, 'document', forbid_detail)
    client = TestClient(app)
    assert client.get('/api/bootstrap').json()['limits']['max_pages'] is None
    response = client.get('/api/documents?summary_only=true')
    row = response.json()[0]
    assert row['unit_count'] == 100 and row['candidate_count'] == row['qa_count'] == 0
    assert not {'units', 'candidates', 'assets'} & row.keys()
    assert 'unit_ids' not in row['topics'][0]
    assert len(response.content) < 3000
    search = client.get('/api/search?q=温度').json()
    assert search[0]['documents'][0]['id'] == doc_id
    assert not {'units', 'candidates', 'assets'} & search[0]['documents'][0].keys()
