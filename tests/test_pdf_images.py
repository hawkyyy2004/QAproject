"""PDF text is evidence, never an exported text file or a page screenshot."""
from pathlib import Path
import pymupdf
import pytest

from qaproject.parsers import extract
from qaproject.app import create_app
from qaproject.pipeline import Worker, approve
from qaproject.settings import settings
from qaproject.store import Store
from test_prototype import fake_provider
import httpx
from fastapi.testclient import TestClient


def parse(store, pdf):
    doc_id, _ = store.import_file('sample.pdf', pdf.tobytes(), 'live')
    units, pages = extract(store, store.document(doc_id), settings())
    return doc_id, units, pages


def raster(white=False):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=300, height=400)
        if not white:
            page.insert_text((30, 60), 'Scan temperature: 15 C', fontsize=16)
            page.draw_rect((40, 100, 220, 220), fill=(.2, .6, .8))
        return page.get_pixmap().tobytes('png')


def test_text_short_text_and_blank_pages_never_render_or_export_text(tmp_path, monkeypatch):
    store = Store(tmp_path/'text')
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40, 60), 'Temperature: 15 C')
        pdf.new_page()  # blank page
        page = pdf.new_page()
        page.insert_text((40, 60), 'Last page contains actual evidence. '*5)
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1), overlay=False)
        for y in (30, 90, 760):
            page.draw_line((35, y), (550, y))

        def forbid_render(*args, **kwargs):
            raise AssertionError('Text and blank pages must not render')

        monkeypatch.setattr(pymupdf.Page, 'get_pixmap', forbid_render)
        doc_id, units, pages = parse(store, pdf)
    assert pages == 3 and {u['page'] for u in units} == {1, 3}
    assert all(not u['image'] and not u['asset_id'] for u in units)
    assert 'Temperature: 15 C' in units[0]['text']
    assert not store.document(doc_id)['assets']
    assert not list((store.root/'assets/images').rglob('*.png'))
    assert not list(store.root.rglob('*.txt')) and not list(store.root.rglob('*.md'))


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_graphic_keeps_complete_page_and_true_locator(tmp_path, rotation):
    store = Store(tmp_path/'full-page')
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.insert_text((40, 60), 'Unrelated page heading for the synthetic observation with calibrated instruments.')
        page.insert_image((150, 190, 330, 430), stream=raster())
        page.insert_text((145, 452), 'Figure 1. Synthetic observation, degrees C.')
        page.insert_text((40, 750), 'Unrelated page footer')
        page.set_rotation(rotation)
        expected = page.get_pixmap(dpi=144, alpha=False)
        doc_id, units, pages = parse(store, pdf)
    assert pages == 1
    visual = next(u for u in units if u['image'])
    assert len(store.document(doc_id)['assets']) == 1
    assert 'graphic_bbox' not in visual
    assert visual['evidence_kind'] == 'pdf_page'
    pix = pymupdf.Pixmap(visual['image'])
    assert (pix.width, pix.height) == (expected.width, expected.height)
    assert pix.samples == expected.samples  # includes the header, chart, caption and footer
    assert b'\x33\x99\xcc' in pix.samples  # actual plot fill survives rotation
    text = next(u for u in units if u.get('companion_of') == visual['id'])
    assert text['image'] == '' and text['locator'] == visual['locator'] == 'page-01-seg-01'
    assert 'Unrelated page heading' in text['text'] and 'degrees C' in text['text']


def test_preview_serves_full_asset_without_cropping(tmp_path):
    app = create_app(runtime=tmp_path/'preview', workers=False)
    store = app.state.stores['live']
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.insert_text((40, 60), 'Page heading')
        page.insert_image((150, 190, 330, 430), stream=raster())
        page.insert_text((40, 750), 'Page footer')
        doc_id, units, _ = parse(store, pdf)
    visual = next(u for u in units if u['image'])
    image = Path(visual['image'])
    with TestClient(app) as client:
        response = client.get('/api/assets/'+visual['asset_id'])
        assert response.status_code == 200 and response.content == image.read_bytes()
        assert not any('/figures' in route.path for route in app.routes)
        assert client.get(f'/api/assets/{visual["asset_id"]}/figures').status_code == 404
    assert not (store.root/'previews').exists()
    assert len(store.document(doc_id)['assets']) == 1


def test_single_vector_figure_with_short_text_is_not_lost(tmp_path):
    store = Store(tmp_path/'vector')
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.draw_circle((200, 200), 60, fill=(.2, .6, .8))
        page.insert_text((140, 290), 'Radius: 60 mm')
        _, units, _ = parse(store, pdf)
    assert len(units) == 1 and units[0]['image']
    assert 'Radius: 60 mm' in units[0]['text']


def test_unused_image_resource_does_not_turn_text_into_a_picture(tmp_path):
    store = Store(tmp_path/'unused')
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_image(page.rect, stream=raster())
        # Keep the resource dictionary, remove the draw command, then add real text.
        for xref in page.get_contents():
            pdf.update_stream(xref, b'')
        page.insert_text((40, 60), 'Only visible text is evidence.')
        _, units, _ = parse(store, pdf)
    assert len(units) == 1 and not units[0]['image']
    assert units[0]['text'] == 'Only visible text is evidence.'


def test_sparse_scanned_content_is_not_discarded_as_blank(tmp_path):
    store = Store(tmp_path/'sparse')
    with pymupdf.open() as source:
        page = source.new_page(width=300, height=400)
        page.draw_rect((150, 200, 151, 201), fill=(0, 0, 0))
        raw = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes('png')
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=300, height=400)
        page.insert_image(page.rect, stream=raw)
        _, units, _ = parse(store, pdf)
    assert len(units) == 1 and units[0]['image']


def test_blank_raster_skipped_and_scan_kept_without_text_layer(tmp_path):
    store = Store(tmp_path/'scan')
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=300, height=400)
        page.insert_image(page.rect, stream=raster(white=True))
        page = pdf.new_page(width=300, height=400)
        page.insert_image(page.rect, stream=raster())
        doc_id, units, pages = parse(store, pdf)
    assert pages == 2 and len(units) == 1 and units[0]['page'] == 2
    assert units[0]['image'] and units[0]['text'] == 'Scan page: use actual visual evidence.'
    assert len(store.document(doc_id)['assets']) == 1


def test_empty_pdf_is_insufficient_without_saved_images(tmp_path):
    store = Store(tmp_path/'blank')
    with pymupdf.open() as pdf:
        pdf.new_page()
        with pytest.raises(ValueError, match='没有可读取证据'):
            parse(store, pdf)
    assert not list((store.root/'assets/images').rglob('*.png'))


def test_pdf_text_reaches_bilingual_generation_review_and_commit_without_images(tmp_path, monkeypatch):
    import qaproject.glm as glm
    monkeypatch.setattr(glm, 'api_key', lambda: 'test-placeholder')
    original_client = httpx.Client
    monkeypatch.setattr(glm.httpx, 'Client', lambda **kw: original_client(transport=httpx.MockTransport(fake_provider)))
    store = Store(tmp_path/'pipeline')
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40, 60), 'Synthetic sea area A in January 2025: temperature was 15 degrees Celsius.')
        doc_id, _ = store.import_file('observation.pdf', pdf.tobytes(), 'live')
    worker = Worker(store)
    worker.process(store.claim(worker.name))
    doc = store.document(doc_id)
    assert doc['status'] == 'waiting_review', doc['error']
    assert len(doc['candidates']) == 2 and not doc['assets']
    for c in doc['candidates']:
        assert c['row']['image'] == '' and 'page-01-seg-01' in c['row']['source']
        approve(store, c['id'], True, c['row_hash'])
    assert store.document(doc_id)['qa_count'] == 2
    assert not list((store.root/'assets/images').rglob('*.png'))
