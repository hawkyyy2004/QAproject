"""A frozen MinerU manifest decides what is text, figure, table or formula."""
import importlib.util
import json
from pathlib import Path

import pymupdf

from qaproject.parsers import extract
from qaproject.settings import settings
from qaproject.store import Store
from qaproject import layout as layout_cache

ROOT = Path(__file__).resolve().parents[1]


def loader():
    spec = importlib.util.spec_from_file_location('mineru_layout', ROOT/'scripts/mineru_layout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crop_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=120, height=90)
        page.insert_text((12, 45), 'NO3 12.4 umol/L', fontsize=11)
        page.get_pixmap().save(path)
    return path


def manifest_for(store, doc_id, pages):
    doc = store.document(doc_id)
    target = store.root/'layout'/doc['content_hash']
    crops = target/'crops'
    for block in (b for page in pages for b in page['blocks']):
        if block['kind'] != 'text':
            crop_png(crops/Path(block['path']).name)
    target.mkdir(parents=True, exist_ok=True)
    (target/'manifest.json').write_text(json.dumps(
        {'schema': 1, 'engine': 'mineru', 'tier': 'basic', 'content_hash': doc['content_hash'],
         'pdf': {'name': doc['name'], 'pages': 1}, 'pages': pages}, ensure_ascii=False), encoding='utf-8')
    return target


def graphics_pdf():
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=600)
        page.insert_text((40, 60), 'Surface chlorophyll a reached 2.1 mg m-3 during the spring bloom. '*2)
        page.draw_rect((40, 120, 360, 300), fill=(.2, .5, .7))
        return pdf.tobytes()


def test_manifest_splits_text_and_figure_without_rendering_the_page(tmp_path, monkeypatch):
    store = Store(tmp_path/'split')
    doc_id, _ = store.import_file('plot.pdf', graphics_pdf(), 'live')
    manifest_for(store, doc_id, [{'page': 1, 'blocks': [
        {'kind': 'text', 'text': 'Surface chlorophyll a reached 2.1 mg m-3 during the spring bloom.'},
        {'kind': 'figure', 'bbox': [0.1, 0.2, 0.9, 0.5],
         'path': 'crops/page-01-fig-01.png', 'caption': '图 1 春季表层叶绿素 a 的空间分布'}]}])

    def forbid(*args, **kwargs):
        raise AssertionError('A frozen layout must not render whole pages')

    monkeypatch.setattr(pymupdf.Page, 'get_pixmap', forbid)
    units, pages = extract(store, store.document(doc_id), settings())
    assert pages == 1
    figures = [u for u in units if u['image']]
    assert len(figures) == 1
    assert figures[0]['locator'] == 'page-01-fig-01'
    assert figures[0]['evidence_kind'] == 'pdf_figure'
    assert '叶绿素' in figures[0]['text'] and figures[0]['asset_id']
    assert Path(figures[0]['image']).parent == store.root/'assets/images'/doc_id
    assert not list((store.root/'assets/images'/doc_id).glob('page-0001.png'))
    body = [u for u in units if not u['image']]
    assert any('2.1 mg m-3' in u['text'] for u in body)


def test_table_keeps_caption_and_html_in_one_visual_unit(tmp_path):
    store = Store(tmp_path/'table')
    doc_id, _ = store.import_file('table.pdf', graphics_pdf(), 'live')
    manifest_for(store, doc_id, [{'page': 1, 'blocks': [
        {'kind': 'table', 'bbox': [0.1, 0.2, 0.9, 0.6], 'path': 'crops/page-01-tbl-01.png',
         'caption': '表 1 各站位硝酸盐浓度', 'html': '<table><tr><td>0.4</td></tr></table>'}]}])
    units, _ = extract(store, store.document(doc_id), settings())
    tables = [u for u in units if u.get('evidence_kind') == 'pdf_table']
    assert len(tables) == 1
    assert tables[0]['locator'] == 'page-01-tbl-01'
    assert '表 1' in tables[0]['text'] and '<table>' in tables[0]['text']
    assert tables[0]['image']


def test_layout_can_be_disabled_or_ignored_when_stale(tmp_path):
    store = Store(tmp_path/'off')
    doc_id, _ = store.import_file('plot.pdf', graphics_pdf(), 'live')
    target = manifest_for(store, doc_id, [{'page': 1, 'blocks': [
        {'kind': 'figure', 'bbox': [0.1, 0.2, 0.9, 0.5], 'path': 'crops/page-01-fig-01.png',
         'caption': '图 1 分布'}]}])
    off = dict(settings(), pdf_layout='off')
    units, _ = extract(store, store.document(doc_id), off)
    assert all(u['evidence_kind'] != 'pdf_figure' for u in units)
    assert any(u.get('evidence_kind') == 'pdf_page' for u in units)

    (target/'manifest.json').write_text(json.dumps(
        {'schema': 1, 'content_hash': 'not-this-file', 'pages': [{'page': 1, 'blocks': [
            {'kind': 'figure', 'path': 'crops/page-01-fig-01.png'}]}]}), encoding='utf-8')
    store2 = Store(tmp_path/'stale')
    doc2, _ = store2.import_file('plot.pdf', graphics_pdf(), 'live')
    target2 = store2.root/'layout'/store2.document(doc2)['content_hash']
    target2.mkdir(parents=True)
    (target2/'manifest.json').write_text((target/'manifest.json').read_text(encoding='utf-8'), encoding='utf-8')
    units2, _ = extract(store2, store2.document(doc2), dict(settings(), pdf_layout='manifest'))
    assert all(u['evidence_kind'] != 'pdf_figure' for u in units2)


def test_auto_builds_the_layout_when_no_manifest_exists(tmp_path, monkeypatch):
    store = Store(tmp_path/'auto')
    doc_id, _ = store.import_file('plot.pdf', graphics_pdf(), 'live')
    doc = store.document(doc_id)
    pages = [{'page': 1, 'blocks': [
        {'kind': 'text', 'text': '观测期间表层叶绿素 a 平均 2.1 mg/m3。'},
        {'kind': 'formula', 'bbox': [0.2, 0.3, 0.5, 0.32],
         'path': 'crops/page-01-eq-01.png', 'latex': r'\Delta A = A(t_i)-A(t_0)'}]}]

    def prepare(store, doc, config):
        manifest_for(store, doc['id'], pages)
        return layout_cache.load(store, doc['content_hash']), ''

    monkeypatch.setattr('qaproject.parsers.layout_cache.prepare', prepare)
    units, _ = extract(store, store.document(doc_id), settings())
    assert [u['evidence_kind'] for u in units if u['image']] == ['pdf_formula']
    assert units[0]['text'].startswith('观测期间')


def test_auto_falls_back_when_mineru_is_unavailable(tmp_path, monkeypatch):
    store = Store(tmp_path/'fallback')
    doc_id, _ = store.import_file('plot.pdf', graphics_pdf(), 'live')
    monkeypatch.setattr('qaproject.parsers.layout_cache.prepare', lambda *args: ({}, '未找到可用的 MinerU 解释器，保留整页'))
    units, _ = extract(store, store.document(doc_id), settings())
    assert any(u.get('evidence_kind') == 'pdf_page' for u in units)


def test_cropped_figures_join_the_topic_that_owns_their_page():
    from qaproject.taxonomy import visual_topic_support
    units = [{'id': 't1', 'image': '', 'page': 1}, {'id': 't2', 'image': '', 'page': 2},
             {'id': 't3', 'image': '', 'page': 2}, {'id': 'f1', 'image': 'a.png', 'page': 1},
             {'id': 'f2', 'image': 'b.png', 'page': 2}, {'id': 'f3', 'image': 'c.png', 'page': 9}]
    topics = [{'id': 'A', 'unit_ids': ['t1']}, {'id': 'B', 'unit_ids': ['t2', 't3']}]
    claims = visual_topic_support(topics, units)
    assert claims == {'A': {'f1'}, 'B': {'f2', 'f3'}}


def test_text_only_documents_claim_no_visual_units():
    from qaproject.taxonomy import visual_topic_support
    units = [{'id': 't1', 'image': '', 'page': 1}]
    assert visual_topic_support([{'id': 'A', 'unit_ids': ['t1']}], units) == {}


def rebuild_script():
    spec = importlib.util.spec_from_file_location('rebuild_evidence', ROOT/'scripts/rebuild_evidence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rebuild_refuses_documents_with_committed_answers(tmp_path):
    script = rebuild_script()
    store = Store(tmp_path/'rebuild')
    doc_id, _ = store.import_file('plot.pdf', graphics_pdf(), 'live')
    with store.connect() as db:
        db.execute("INSERT INTO candidates(id,document_id,unit_id,topic_id,row_json,row_hash,evidence_hash,"
                   "fingerprint,status,review,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   ('c1', doc_id, 'u', None, '{}', 'h', 'e', 'f', 'approved', '{}', 0))
    result = script.rebuild(store, doc_id, apply=True)
    assert 'skipped' in result and '入库问答' in result['skipped']
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0] == 1


def test_rebuild_clears_draft_evidence_and_requeues(tmp_path):
    script = rebuild_script()
    store = Store(tmp_path/'reset')
    doc_id, _ = store.import_file('plot.pdf', graphics_pdf(), 'live')
    with store.connect() as db:
        db.execute("UPDATE documents SET units=?, unit_count=1, status='waiting_review' WHERE id=?",
                   (json.dumps([{'id': 'u1'}]), doc_id))
        db.execute("UPDATE tasks SET status='waiting_review' WHERE document_id=?", (doc_id,))
        db.execute("INSERT INTO candidates(id,document_id,unit_id,topic_id,row_json,row_hash,evidence_hash,"
                   "fingerprint,status,review,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   ('c1', doc_id, 'u1', None, '{}', 'h', 'e', 'f', 'pending', '{}', 0))
    assert script.rebuild(store, doc_id)['would_reset']
    assert script.rebuild(store, doc_id, apply=True)['requeued']
    doc = store.document(doc_id)
    assert doc['units'] == [] and doc['status'] == 'queued'
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM tasks WHERE document_id=?', (doc_id,)).fetchone()[0] == 2


def test_panels_and_inner_labels_group_into_single_figure():
    module = loader()
    blocks = [
        {'type': 'image', 'bbox': [0.10, 0.10, 0.50, 0.30],
         'content': [{'type': 'image_body', 'content': ''}]},
        {'type': 'image', 'bbox': [0.10, 0.31, 0.50, 0.51],
         'content': [{'type': 'image_body', 'content': ''}]},
        {'type': 'equation', 'bbox': [0.30, 0.33, 0.45, 0.345], 'content': r'\theta_0=40^\circ'},
        {'type': 'text', 'bbox': [0.10, 0.52, 0.50, 0.55],
         'content': [{'type': 'text', 'content': '图 2 2018 年航次表层冰雪采样站位'}]},
        {'type': 'table', 'bbox': [0.10, 0.60, 0.90, 0.80],
         'content': [{'type': 'table_caption', 'content': [{'type': 'text', 'content': '表 1 活度汇编'}]},
                     {'type': 'table_body', 'content': '<table></table>'}]},
        {'type': 'text', 'bbox': [0.10, 0.05, 0.90, 0.08],
         'content': [{'type': 'text', 'content': '春季融雪后表层水体开始分层。'}]},
    ]
    items = module.plan_page(blocks)
    kinds = [item['kind'] for item in items]
    assert kinds.count('figure') == 1 and kinds.count('formula') == 0
    figure = next(item for item in items if item['kind'] == 'figure')
    assert figure['bbox'][3] >= 0.51 and figure['caption'].startswith('图 2')
    table = next(item for item in items if item['kind'] == 'table')
    assert table['caption'] == '表 1 活度汇编' and table['html'] == '<table></table>'
    assert [item['text'] for item in items if item['kind'] == 'text'] == ['春季融雪后表层水体开始分层。']


def test_side_by_side_panels_merge_but_two_captioned_figures_do_not():
    module = loader()
    blocks = [
        {'type': 'image', 'bbox': [0.10, 0.10, 0.44, 0.30], 'content': []},
        {'type': 'image', 'bbox': [0.46, 0.10, 0.80, 0.30], 'content': []},
        {'type': 'text', 'bbox': [0.10, 0.31, 0.80, 0.33],
         'content': [{'type': 'text', 'content': '图 3 北极表层气溶胶观测站位与活度'}]},
        {'type': 'image', 'bbox': [0.10, 0.40, 0.44, 0.60],
         'content': [{'type': 'image_caption', 'content': [{'type': 'text', 'content': '图 4 冰雪年龄分布'}]}]},
        {'type': 'image', 'bbox': [0.46, 0.40, 0.80, 0.60],
         'content': [{'type': 'image_caption', 'content': [{'type': 'text', 'content': '图 5 大气沉降通量'}]}]},
    ]
    items = module.plan_page(blocks)
    figures = [item for item in items if item['kind'] == 'figure']
    assert len(figures) == 3
    grid = figures[0]
    assert grid['bbox'][0] <= 0.11 and grid['bbox'][2] >= 0.79
    assert grid['caption'].startswith('图 3')
    assert [figure['caption'] for figure in figures[1:]] == ['图 4 冰雪年龄分布', '图 5 大气沉降通量']
