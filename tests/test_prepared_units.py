"""QA must not inherit whole-page corpus evidence when MinerU is enabled."""
from qaproject.pipeline import reusable_prepared_units


def unit(kind, image=False):
    return {'id': 'u1', 'evidence_kind': kind, 'text': 'x',
            'image': '/assets/a.png' if image else ''}


def test_non_pdf_prepared_units_are_reused():
    assert reusable_prepared_units([unit('extracted_text')], {'path': 'a.csv'}, {'pdf_layout': 'auto'})


def test_page_renders_are_rejected_when_mineru_is_enabled():
    doc = {'path': 'a.pdf'}
    assert not reusable_prepared_units([unit('pdf_page', True)], doc, {'pdf_layout': 'auto'})
    assert not reusable_prepared_units([unit('extracted_text'), unit('pdf_page', True)],
                                       doc, {'pdf_layout': 'manifest'})


def test_text_only_and_cropped_evidence_are_reused():
    doc = {'path': 'a.pdf'}
    assert reusable_prepared_units([unit('extracted_text')], doc, {'pdf_layout': 'auto'})
    assert reusable_prepared_units([unit('extracted_text'), unit('pdf_figure', True)],
                                   doc, {'pdf_layout': 'auto'})


def test_explicit_layout_off_keeps_page_renders():
    doc = {'path': 'a.pdf'}
    assert reusable_prepared_units([unit('pdf_page', True)], doc, {'pdf_layout': 'off'})


def test_actual_mineru_scan_evidence_does_not_trigger_a_second_parse():
    scan=dict(unit('pdf_page', True),layout_engine='mineru')
    assert reusable_prepared_units([scan], {'path':'scan.pdf'}, {'pdf_layout':'auto'})
