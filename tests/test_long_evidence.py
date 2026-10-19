"""Oversized evidence is split locally, losslessly, before its hashes are frozen."""
from pathlib import Path

import pymupdf
import pytest

from qaproject.glm import GLMClient
from qaproject.parsers import extract, text_alternatives, text_spans, text_view_matches, TEXT_CHUNK_SIZE
from qaproject.pipeline import Worker, batches, compact_units, approve, CHECKS, BILINGUAL_CHECKS
from qaproject.settings import settings
from qaproject.store import Store, digest
from qaproject.taxonomy import TopicSelection


@pytest.mark.parametrize('text',[
    '原始证据。'*7000,
    ('Scientific observation. Temperature: 15 C; salinity: 32 PSU.\n'*900)+'TAIL',
    'x'*50001,
    ('a\t b\r\n c  ' *2500)+'末尾',
    'short evidence',
    'x'*16000,
])
def test_spans_preserve_every_character_without_overlap(text):
    spans=list(text_spans(text))
    assert ''.join(text[a:b] for a,b in spans)==text
    assert spans[0][0]==0 and spans[-1][1]==len(text)
    assert all(left[1]==right[0] for left,right in zip(spans,spans[1:]))
    if len(text)>16000:
        assert max(b-a for a,b in spans)<=TEXT_CHUNK_SIZE
    else:
        assert spans==[(0,len(text))]


def dense_pdf(graphic=False):
    pdf=pymupdf.open()
    page=pdf.new_page(width=1000,height=1400)
    line='Station Alpha recorded hourly temperature of 15 degrees Celsius with a calibrated sensor. '*3
    page.insert_text((20,20),'\n'.join([line]*180)+ '\nTAIL: Station Beta recorded salinity of 32 PSU.',fontsize=4)
    if graphic:
        page.draw_circle((700,1100),60,fill=(.2,.6,.8))
    assert len(page.get_text().strip())>16000
    return pdf


def parse(store,name,raw):
    doc_id,_=store.import_file(name,raw,'live')
    doc=store.document(doc_id)
    units,pages=extract(store,doc,settings())
    return doc_id,units,pages


def assert_hashes(units):
    assert len({u['id'] for u in units})==len(units)
    for unit in units:
        data=dict(unit)
        assert data.pop('hash')==digest(data)


def test_dense_pdf_keeps_all_text_real_locators_and_no_text_screenshot(tmp_path,monkeypatch):
    store=Store(tmp_path/'dense')
    with dense_pdf() as pdf:
        original=pdf[0].get_text('text',sort=True).strip()
        raw=pdf.tobytes()
    monkeypatch.setattr(pymupdf.Page,'get_pixmap',lambda *a,**k:pytest.fail('Pure text must not render'))
    _,units,pages=parse(store,'dense.pdf',raw)
    assert pages==1 and len(units)>1
    assert ''.join(u['text'] for u in units)==original
    assert all(not u['image'] and len(u['text'])<=TEXT_CHUNK_SIZE for u in units)
    assert [u['locator'] for u in units]==[f'page-0001-seg-{n:04d}' for n in range(1,len(units)+1)]
    assert all(original[slice(*u['text_span'])]==u['text'] for u in units)
    assert not list(store.root.rglob('*.png')) and not list(store.root.rglob('*.txt'))
    assert_hashes(units)


@pytest.mark.parametrize('suffix',['txt','md','csv'])
def test_other_formats_do_not_fail_or_invent_pdf_pages(tmp_path,suffix):
    store=Store(tmp_path/suffix)
    long='Observation: calibrated temperature was 15 degrees Celsius. '*750
    if suffix=='csv':
        raw=('description\n"'+long+'"\nshort second record\n').encode()
    else:
        raw=(long+'\n\nA separate paragraph.').encode()
    _,units,pages=parse(store,'evidence.'+suffix,raw)
    assert pages==0 and len(units)>1
    assert all(len(u['text'])<=TEXT_CHUNK_SIZE and 'page-' not in u['locator'] for u in units)
    if suffix!='csv':
        assert ''.join(u['text'] for u in units)==long.strip()+'A separate paragraph.'
        assert [u['seg'] for u in units]==list(range(1,len(units)+1))
    else:
        # Current CSV evidence is a serialized header/record block; all values survive.
        text=''.join(u['text'] for u in units)
        assert long in text and 'short second record' in text and 'description' in text
        assert all(u['locator'].startswith('record-0002-0003-seg-') for u in units)
    assert_hashes(units)


def test_long_graphic_page_has_one_full_image_and_bounded_text_views(tmp_path):
    store=Store(tmp_path/'graphic')
    with dense_pdf(graphic=True) as pdf:
        original=pdf[0].get_text('text',sort=True).strip()
        expected=pdf[0].get_pixmap(dpi=144,alpha=False)
        doc_id,units,_=parse(store,'graphic.pdf',pdf.tobytes())
    parent=next(u for u in units if u['image'])
    before=digest(parent)
    views=[u for u in units if u.get('companion_of')==parent['id']]
    assert len(views)>1 and parent['text']==original
    assert ''.join(u['text'] for u in views)==original
    assert all(text_view_matches(u,parent) and len(u['text'])<=TEXT_CHUNK_SIZE for u in views)
    assert [u['locator'] for u in views]==[f'page-0001-seg-{n:04d}' for n in range(2,len(views)+2)]
    assert len(store.document(doc_id)['assets'])==1
    pix=pymupdf.Pixmap(parent['image'])
    assert pix.samples==expected.samples and (pix.width,pix.height)==(expected.width,expected.height)
    assert text_alternatives(units)==units and digest(parent)==before
    # One page image, not one image for every text fragment; OCR is not sent twice.
    image_payload=compact_units([parent])[0]
    assert len(image_payload['text'])<120 and image_payload['text']!=original
    assert sum(len(u['text']) for u in compact_units(views))==len(original)
    assert_hashes(units)

    selection=TopicSelection(store.topics(),units)
    selection.add({'summary':'温度观测。','topics':[{'name':'海洋观测','keywords':['温度'],
                   'unit_ids':[views[-1]['id']]}]},[views[-1]])
    selected,_=selection.select()
    assert selected[0]['unit_ids']=={parent['id'],views[-1]['id']}
    forged=dict(views[-1],text='A different unsupported observation.')
    assert not text_view_matches(forged,parent)


def test_long_pdf_completes_analysis_and_commits_last_segment_without_extra_stages(tmp_path,monkeypatch):
    store=Store(tmp_path/'workflow')
    with dense_pdf() as pdf:
        doc_id,_=store.import_file('workflow.pdf',pdf.tobytes(),'live')
    cfg=dict(settings(),target_per_document=1,min_per_document=1)
    monkeypatch.setattr('qaproject.pipeline.settings',lambda:cfg)
    seen=[]
    def call(self,instruction,payload,units=(),**kw):
        if payload.get('analysis_phase')=='facts':
            seen.append(('analysis',units))
            return {'summary':'温度与盐度记录。','facts':[{'unit_id':u['id'],'focus':'温度及末段盐度记录。','keywords':['盐度']} for u in units]},{}
        stage='topic_plan' if 'existing_topics' in payload else ('review' if 'candidates' in payload else 'generation')
        seen.append((stage,units))
        if stage=='topic_plan':
            return {'summary':'温度与盐度观测。','topics':[{'name':'海洋观测','keywords':['盐度'],
                    'unit_ids':[u['id'] for u in units]}],'pairs':[],'skipped':[]},{}
        if stage=='review':
            return {'checks':[dict(candidate_id=c['candidate_id'],note='两个版本均符合本段盐度记录。',
                    **{k:True for k in CHECKS+BILINGUAL_CHECKS}) for c in payload['candidates']]},{}
        tail=next((u for u in units if 'TAIL:' in u['text']),None)
        if not tail:
            return {'pairs':[],'skipped':['本测试仅选末段知识点。']},{}
        return {'pairs':[{'unit_id':tail['id'],'topic':'海洋观测',
                'zh':{'input':'Beta站的盐度记录为多少？','thinking':'本段记录盐度为32 PSU。','output':'32 PSU。'},
                'en':{'input':'What salinity was recorded at station Beta?','thinking':'The segment records 32 PSU.','output':'32 PSU.'}}],
                'skipped':[]},{}
    monkeypatch.setattr(GLMClient,'call',call)
    worker=Worker(store)
    worker.process(store.claim(worker.name,document_id=doc_id))
    doc=store.document(doc_id)
    assert doc['status']=='waiting_review' and doc['coverage']=='complete',doc['error']
    assert len(doc['candidates'])==2 and doc['bilingual_count']==1
    analysis_calls=[g for s,g in seen if s=='analysis']
    assert sorted(u['id'] for g in analysis_calls for u in g)==sorted(u['id'] for u in doc['units'])
    assert len(analysis_calls)==len(list(batches(doc['units'])))
    tail=next(u for u in doc['units'] if 'TAIL:' in u['text'])
    assert tail['seg']>1
    for candidate in doc['candidates']:
        assert candidate['unit_id']==tail['id'] and tail['locator'] in candidate['row']['source']
        assert candidate['row']['image']==''
        approve(store,candidate['id'],True,candidate['row_hash'])
    assert store.document(doc_id)['qa_count']==2
