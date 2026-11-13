"""Corpus improvements use isolated source files and simulated model results."""
import csv
import io
import json
import threading
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from qaproject import corpus, knowledge, corpus_processing as processing
from qaproject.app import create_app
from qaproject.pipeline import Worker
from qaproject.store import Store, digest, encode
from qaproject.execution import ExecutionLimits


def run(store,doc):
    w=Worker(store,limits=ExecutionLimits())
    try:
        assert w.run_once()
        return corpus.listing(store)['items'][0]['id']
    finally:w.limits.close()


def image_bytes():
    f=io.BytesIO();Image.new('RGB',(100,120),(20,90,150)).save(f,format='PNG');return f.getvalue()


def test_article_scope_figures_table_and_scientific_conditions():
    repeated='Ocean salinity depends on evaporation, precipitation and exchange with nearby waters. '
    parsed=processing.article('<head><title>hidden</title></head><div>menu</div><main><h1>Ocean</h1>'
        '<p>Loading...</p><p>'+repeated+'</p><p>'+repeated+'</p>'
        '<p>Under different conditions salinity does not increase.</p><table><tr><td>1</td><td>2</td></tr></table>'
        '<figure><img src="/ocean.png" alt="salinity map"><figcaption>Observed salinity map.</figcaption></figure>'
        '<img src="/logo.png"></main><footer>copyright</footer>','https://example.org/page')
    assert 'menu' not in parsed['text'] and 'hidden' not in parsed['text'] and 'Loading' not in parsed['text']
    assert parsed['text'].count(repeated.strip())==1
    assert 'does not increase' in parsed['text'] and '1 | 2' in parsed['text']
    assert parsed['images']==[dict(url='https://example.org/ocean.png',alt='salinity map',caption='Observed salinity map.')]
    assert len(parsed['removed'])>=3


def test_pdf_structure_heading_table_and_full_image_relationship(tmp_path):
    import pymupdf
    path=tmp_path/'sea.pdf'
    with pymupdf.open() as pdf:
        p=pdf.new_page();p.insert_text((40,80),'Ocean circulation',fontsize=22)
        for n in range(8):p.insert_text((40,120+n*16),'Ocean circulation depends on wind and water density.',fontsize=11)
        pdf.save(path)
    structure=processing.pdf_structure(path)
    text=processing.markdown('sea.pdf',[],structure)
    assert '## Ocean circulation' in text and 'page-01' in text
    assert structure['engine'].startswith('PyMuPDF') and structure['limitations']


def test_two_exports_use_identical_full_source_and_keep_audit_rows(tmp_path):
    store=Store(tmp_path/'live')
    doc,_=store.import_file('sea.html',b'<title>Sea</title><article><p>Salinity describes dissolved salt.</p><p>Rainfall changes salinity.</p></article>','live',task_stage='corpus')
    corpus.register_source(store,doc,'web','Sea','https://example.org/sea')
    version_id=run(store,doc);version=corpus.verify(store,version_id);unit=version['units'][0]
    row=dict(entity_name='Salinity',entity_type='Property',entity_description='Dissolved salts.',page_urls=version['source_url'],page_titles='Sea',full_text=unit['text'],full_text_length=len(unit['text']))
    review=dict(row_hash=digest(row),evidence_hash=unit['hash'],**{k:True for k in knowledge.CHECKS})
    with store.connect() as db:
        db.execute("INSERT INTO knowledge_jobs(document_id,task_id,status,units,updated) VALUES(?,?,'completed',?,0)",(doc,'k',encode(version['units'])))
        db.execute('INSERT INTO knowledge_entries VALUES(?,?,?,?,?,?,?,?,?,?,?)',('fact',doc,unit['id'],unit['hash'],encode(row),digest(row),'Ocean',unit['text'],'approved',encode(review),0))
        before=tuple(db.execute('SELECT * FROM knowledge_entries').fetchone())
    first=next(csv.DictReader(io.StringIO(knowledge.export_csv(store).decode())))
    second=next(csv.DictReader(io.StringIO(corpus.export_csv(store,version_id)[0].decode())))
    assert first==second and 'Rainfall changes salinity.' in first['full_text']
    assert int(first['full_text_length'])==len(first['full_text'])
    detail=knowledge.detail(store,'fact')
    assert detail['row']==row and detail['export_row']==first | {'full_text_length':len(first['full_text'])}
    with store.connect() as db:assert tuple(db.execute('SELECT * FROM knowledge_entries').fetchone())==before


def test_reference_import_export_matches_real_image_and_preserves_reference(tmp_path):
    app=create_app(tmp_path/'app',workers=False);client=TestClient(app);headers={'x-qa-session':app.state.session};store=app.state.stores['live']
    doc,_=store.import_file('sea.png',image_bytes(),'live',task_stage='corpus');version=run(store,doc)
    endpoint=f'/api/corpus/{version}/annotations/reference'
    result=client.post(endpoint,headers=headers,json=dict(records=[dict(image_path='data/sea.png',object='square box',bnd_box=[.1,.2,.8,.9],input='Original task',output='Original annotation')],source_note='Original verified annotation',verified=True,class_map={'square box':'cube'}))
    assert result.status_code==200,result.text
    assert result.json()['imported']==1
    package=corpus.export_official_bundle(store,version)
    with zipfile.ZipFile(io.BytesIO(package)) as z:
        record=json.loads(z.read('Open-Dataset/OceanSona/sona_train_data.json'))[0]
        assert record['object']=='cube' and record['bnd_box']==[.1,.2,.8,.9]
        assert record['input']=='Original task' and z.read(record['image_path'])==image_bytes()
        assert '原类别' in json.loads(z.read('Open-Dataset/annotations.json'))[0]['source_note']
    bad=client.post(endpoint,headers=headers,json=dict(records=[dict(image_path='../../etc/passwd',category='fish')],source_note='test')).json()
    assert bad['failed']==1
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM qas').fetchone()[0]==0
        assert db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]==0


def test_reprocess_creates_version_without_rewriting_existing_qa_evidence(tmp_path):
    store=Store(tmp_path/'live');doc,_=store.import_file('sea.txt',b'Ocean evidence.','live',task_stage='corpus')
    first=run(store,doc);frozen=corpus.verify(store,first);before=store.document(doc)
    corpus.enqueue(store,doc,reprocess=True);second=run(store,doc)
    assert first!=second and corpus.verify(store,first)==frozen
    assert store.document(doc)==before
    assert corpus.listing(store)['total']==2


def test_web_image_review_uses_actual_bytes_and_shared_adapter(tmp_path,monkeypatch):
    from qaproject import web_sources,glm
    store=Store(tmp_path/'live');doc,_=store.import_file('sea.html',b'<article><p>Ocean salinity distribution.</p><figure><img src="https://example.org/map.png"><figcaption>Salinity map</figcaption></figure></article>','live',task_stage='corpus')
    corpus.register_source(store,doc,'web','Salinity','https://example.org/sea')
    monkeypatch.setattr(web_sources,'capture_image',lambda url:dict(raw=image_bytes(),url=url,width=100,height=120,suffix='.png'))
    seen=[]
    def call(self,instruction,payload,units=(),**kwargs):
        seen.append(units);assert Path(units[0]['image']).read_bytes()==image_bytes()
        return dict(relevant=True,usable=True,note='Actual scientific map.'),dict(images=1,finish_reason='stop')
    monkeypatch.setattr(glm.GLMClient,'call',call)
    version=corpus.verify(store,run(store,doc))
    assert version['manifest']['image_count']==1 and len(seen)==1
    assert version['manifest']['processing']['images'][0]['status']=='accepted'
    assert any(u.get('origin_url')=='https://example.org/map.png' for u in version['units'])


def test_semantic_review_does_not_merge_changed_numbers(tmp_path,monkeypatch):
    from qaproject import corpus_quality,glm
    store=Store(tmp_path/'live');doc,_=store.import_file('sea.txt',b'Ocean','live',task_stage='corpus')
    text=('Salinity changes with ocean evaporation and exchange. '*8)
    units=[dict(id='a',text=text+'At 20 degrees.',image=''),dict(id='b',text=text+'At 25 degrees.',image='')]
    monkeypatch.setattr(glm.GLMClient,'call',lambda *a,**kw:pytest.fail('Changed numbers must remain distinct without model merge'))
    w=Worker(store,limits=ExecutionLimits())
    try:
        result,report=corpus_quality.refine(w,store.document(doc),units,lambda:None)
        assert result==units and report==[]
    finally:w.limits.close()


def test_mineru_explicit_reuses_raw_result_for_structure_and_original_figure_pipeline(tmp_path,monkeypatch):
    import pymupdf
    from qaproject import layout
    store=Store(tmp_path/'live')
    with pymupdf.open() as pdf:
        p=pdf.new_page();p.insert_text((40,80),'Native ocean evidence.')
        raw=pdf.tobytes()
    doc,_=store.import_file('paper.pdf',raw,'live',task_stage='corpus')
    first=run(store,doc);before=store.document(doc)
    cached=store.root/'corpus/mineru'/store.document(doc)['content_hash']/'result.json'
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps({'pages':[{'page_idx':0,'blocks':[
        {'type':'doc_title','bbox':[.1,.1,.8,.15],'content':'Ocean salinity'},
        {'type':'equation','bbox':[.1,.2,.8,.3],'content':'S = m/V'},
        {'type':'table','bbox':[.1,.4,.8,.6],'content':[{'type':'table_body','content':'<table><tr><td>salinity</td><td>35</td></tr></table>'}]}]}]}))
    corpus.enqueue(store,doc,reprocess=True,pdf_engine='mineru')
    second=corpus.verify(store,run(store,doc))
    assert second['id']!=first and '## Ocean salinity' in second['markdown']
    assert '$$\nS = m/V\n$$' in second['markdown'] and '<table>' in second['markdown']
    assert any('S = m/V' in u['text'] for u in second['units'])
    assert {u['evidence_kind'] for u in second['units'] if u['image']}=={'pdf_formula','pdf_table'}
    assert second['manifest']['image_count']==2
    after=store.document(doc)
    assert {k:v for k,v in after.items() if k!='assets'}=={k:v for k,v in before.items() if k!='assets'}


def test_semantic_confirmation_is_cached_but_different_negation_is_never_merged(tmp_path,monkeypatch):
    from qaproject import corpus_quality,glm
    store=Store(tmp_path/'live');doc,_=store.import_file('sea.txt',b'Ocean','live',task_stage='corpus')
    base=('Ocean currents transport heat and affect regional climate. '*8)
    units=[dict(id='a',text=base+'This result applies.',image=''),dict(id='b',text=base+'This result applies!',image='')]
    calls=[]
    def call(*args,**kw):
        calls.append(1);return dict(equivalent=True,note='Same fact and same conditions.'),dict(finish_reason='stop')
    monkeypatch.setattr(glm.GLMClient,'call',call)
    w=Worker(store,limits=ExecutionLimits())
    try:
        first,report=corpus_quality.refine(w,store.document(doc),units,lambda:None)
        assert len(first)==1 and report[0]['review']['equivalent'] and len(calls)==1
        assert corpus_quality.refine(w,store.document(doc),units,lambda:None)[0]==first and len(calls)==1
        units[1]['text']=base+'This result does not apply.'
        assert corpus_quality.refine(w,store.document(doc),units,lambda:None)[0]==units and len(calls)==1
    finally:w.limits.close()


def test_unverified_reference_does_not_become_training_label(tmp_path):
    from qaproject.corpus_annotations import import_reference
    store=Store(tmp_path/'live');doc,_=store.import_file('sea.png',image_bytes(),'live',task_stage='corpus')
    version=run(store,doc)
    result=import_reference(store,version,[dict(image_path='sea.png',category='Fish')],'Unverified import')
    assert result['imported']==1
    with zipfile.ZipFile(io.BytesIO(corpus.export_bundle(store,version))) as z:
        assert json.loads(z.read('Open-Dataset/OceanVision/OceanVision_train_data.json'))==[]
        assert len(json.loads(z.read('Open-Dataset/annotations.json')))==1
        assert z.namelist().count('images/0001.png')==1


def test_http_charset_and_reference_source_records_are_preserved(tmp_path):
    assert '海水' in processing.article('<article>海水</article>'.encode('gb18030'),charset='gb18030')['text']
    store=Store(tmp_path/'live');doc,_=store.import_file('wiki.csv',b'entity table','live',task_stage='knowledge')
    units=[]
    for n,text in [(1,'Definition from source A.'),(2,'Unrelated page B.')]:
        u=dict(id=f'{doc}:record{n}:seg1',text=text,page=0,seg=1,locator=f'record-{n:04d}-seg-01',image='',asset_id='',entity_hint='A' if n==1 else 'B')
        u['hash']=digest(u);units.append(u)
    with store.connect() as db:
        db.execute('UPDATE knowledge_jobs SET units=? WHERE document_id=?',(encode(units),doc))
    row=dict(entity_name='A',entity_type='Theory',entity_description='Definition A',page_urls='https://example.org/a',page_titles='Page A',full_text=units[0]['text'],full_text_length=len(units[0]['text']))
    entry=dict(document_id=doc,row=row,evidence=[dict(unit_id=units[0]['id'],evidence_hash=units[0]['hash'])])
    exported=knowledge.export_row(store,entry)
    assert exported['full_text']=='Definition from source A.' and exported['page_urls']=='https://example.org/a'
