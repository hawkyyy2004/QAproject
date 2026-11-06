from fastapi.testclient import TestClient
import pytest

from qaproject.app import create_app
from qaproject.pipeline import Worker
from qaproject import corpus


def application(tmp_path):
    app=create_app(tmp_path/'app',workers=False)
    client=TestClient(app)
    return app,client,{'x-qa-session':app.state.session}


@pytest.mark.parametrize('selected',[
    ('qa',),('knowledge',),('corpus',),('qa','knowledge'),
    ('qa','corpus'),('knowledge','corpus'),('qa','knowledge','corpus'),
])
@pytest.mark.parametrize('entrypoint',['upload','library'])
def test_independent_processing_selections_queue_only_requested_tasks(tmp_path,selected,entrypoint):
    root=tmp_path/'sources';root.mkdir()
    original=root/'ocean.txt';original.write_text('海水盐度反映溶解盐含量。')
    app=create_app(tmp_path/'app',workers=False,library_root=root)
    client=TestClient(app);headers={'x-qa-session':app.state.session}
    params=[('processing',value) for value in selected]
    if entrypoint=='upload':
        def submit():
            return client.post('/api/imports',params=params,headers=headers,
                files={'files':(original.name,original.read_bytes())})
    else:
        app.state.library.scan()
        row=client.get('/api/library').json()['items'][0]
        def submit():
            return client.post('/api/library/imports',params=params,headers=headers,
                json={'files':[{'id':row['id'],'version':row['version']}]})
    response=submit()
    assert response.status_code==200,response.text
    doc_id=response.json()[0]['id'];store=app.state.stores['live']
    # Re-selecting the same material does not duplicate any active task.
    assert submit().json()[0]['id']==doc_id
    with store.connect() as db:
        stages=[row[0] or 'queued' for row in db.execute(
            'SELECT initial_stage FROM tasks WHERE document_id=? ORDER BY rowid',(doc_id,))]
        assert sorted(stages)==sorted('queued' if x=='qa' else x for x in selected)
        if 'corpus' in selected:
            assert stages[0]=='corpus'
            assert db.execute('SELECT generate_after FROM corpus_jobs WHERE document_id=?',(doc_id,)).fetchone()[0]==0
    if entrypoint=='library':
        assert not list((store.root/'originals').iterdir())
        assert store.document(doc_id)['path']==str(original)


def test_old_corpus_qa_alias_keeps_instruction_chain(tmp_path):
    app,client,headers=application(tmp_path)
    response=client.post('/api/imports?processing=corpus_qa',headers=headers,
        files={'files':('sea.txt',b'Ocean evidence.')})
    assert response.status_code==200,response.text
    with app.state.stores['live'].connect() as db:
        assert db.execute('SELECT generate_after FROM corpus_jobs').fetchone()[0]==1
        assert [r[0] for r in db.execute('SELECT initial_stage FROM tasks')]==['corpus']


def test_invalid_processing_selection_does_not_import(tmp_path):
    app,client,headers=application(tmp_path)
    response=client.post('/api/imports?processing=',headers=headers,
        files={'files':('sea.txt',b'Ocean evidence.')})
    assert response.status_code==422
    assert app.state.stores['live'].documents(summary_only=True)==[]


def test_corpus_only_upload_can_later_enter_original_qa(tmp_path):
    app,client,headers=application(tmp_path)
    result=client.post('/api/imports?processing=corpus',headers=headers,
                       files={'file':('sea.txt','海水盐度反映溶解盐含量。'.encode())})
    assert result.status_code==200,result.text
    doc_id=result.json()[0]['id'];store=app.state.stores['live']
    with store.connect() as db:
        tasks=db.execute('SELECT initial_stage FROM tasks WHERE document_id=?',(doc_id,)).fetchall()
    assert [x[0] for x in tasks]==['corpus']
    assert not next(d for d in store.documents(summary_only=True) if d['id']==doc_id)['has_qa_tasks']
    assert Worker(store).run_once()
    assert client.get('/api/corpus').json()['total']==1
    queued=client.post(f'/api/documents/{doc_id}/qa',headers=headers)
    assert queued.status_code==200,queued.text
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM tasks WHERE document_id=? AND initial_stage='queued'",(doc_id,)).fetchone()[0]==1


def test_web_capture_preserves_snapshot_and_source(tmp_path,monkeypatch):
    app,client,headers=application(tmp_path)
    raw='<html><title>海洋知识</title><p>盐度是海水的重要性质。</p></html>'.encode()
    monkeypatch.setattr('qaproject.web_sources.capture',lambda url:dict(raw=raw,url=url,
        requested_url=url,suffix='.html',title='海洋知识',text='盐度是海水的重要性质。'))
    response=client.post('/api/corpus/web',headers=headers,json={
        'url':'https://example.org/ocean','extract_knowledge':False,'source_label':'团队网页资料'})
    assert response.status_code==200,response.text
    store=app.state.stores['live'];doc=store.document(response.json()['document_id'])
    assert store.original(doc['id'])[0].read_bytes()==raw
    assert Worker(store).run_once()
    version=corpus.listing(store)['items'][0]
    detail=client.get('/api/corpus/'+version['id']).json()
    assert detail['source_url']=='https://example.org/ocean'
    assert '盐度是海水的重要性质。' in str(detail)
    download=client.get('/api/corpus/export/'+version['id'])
    assert download.status_code==200 and download.content.startswith(b'PK')


def test_new_mutations_require_session_and_no_arbitrary_paths(tmp_path):
    app,client,headers=application(tmp_path)
    assert client.post('/api/corpus/web',json={'url':'https://example.org'}).status_code==403
    assert client.post('/api/instructions',json={}).status_code==403
    result=client.post('/api/documents/not-a-document/corpus',headers=headers,json={'path':'/etc/passwd'})
    assert result.status_code==422
    assert client.get('/api/corpus/does-not-exist').status_code in {400,404}


def test_bad_import_does_not_queue_hidden_qa(tmp_path):
    app,client,headers=application(tmp_path)
    response=client.post('/api/imports?processing=corpus',headers=headers,
                          files={'file':('bad.exe',b'not a document')})
    assert response.json()[0]['error']
    assert app.state.stores['live'].documents(summary_only=True)==[]


def test_relocated_historical_image_is_reused_without_rewriting_old_evidence(tmp_path):
    from qaproject.store import digest
    app,client,headers=application(tmp_path);store=app.state.stores['live']
    doc_id,_=store.import_file('sea.txt',b'Ocean evidence.','live',task_stage='corpus')
    image=store.root/'assets/images'/doc_id/'page-0001.png'
    image.parent.mkdir(parents=True);image.write_bytes(b'identical full image bytes')
    asset_id=store.add_asset(doc_id,image,1,'old evidence')
    old=tmp_path/'previous-location/assets/images'/doc_id/image.name
    with store.connect() as db:db.execute('UPDATE assets SET path=? WHERE id=?',(str(old),asset_id))
    unit=dict(id=doc_id+':p1:s1',text='Ocean evidence.',page=1,seg=1,image=str(old),asset_id=asset_id,locator='page-01-seg-01')
    unit['hash']=digest(unit);store.update_document(doc_id,units=[unit],pages=1)
    corpus.enqueue(store,doc_id)
    assert Worker(store).run_once()
    version=corpus.listing(store)['items'][0]
    assert corpus.verify(store,version['id'])['units'][0]['image']!=str(old)
    assert store.document(doc_id)['units']==[unit]
    with store.connect() as db:
        assert db.execute('SELECT path FROM assets WHERE id=?',(asset_id,)).fetchone()[0]==str(old)


def test_multi_evidence_policy_is_explicit_and_does_not_relax_original_qa(monkeypatch):
    import json
    import httpx
    from qaproject import glm
    from qaproject.settings import settings
    sent=[]
    def provider(request):
        sent.append(json.loads(request.content)['messages'][0]['content'])
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
    original=httpx.Client
    monkeypatch.setattr(glm.httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(provider)))
    glm.GLMClient(settings(),key='test-only-placeholder').call('测试',{})
    glm.GLMClient(dict(settings(),evidence_policy='registered_bundle'),key='test-only-placeholder').call('测试',{})
    assert '每条问答仅依据一个给定 unit' in sent[0]
    assert '完整列出实际使用的evidence_ids' in sent[1]
    assert '每条问答仅依据一个给定 unit' not in sent[1]


def test_official_paper_csv_and_bundle_resolve_same_paths(tmp_path):
    import io,csv,zipfile
    from reportlab.pdfgen import canvas
    app,client,headers=application(tmp_path);store=app.state.stores['live']
    raw=io.BytesIO();c=canvas.Canvas(raw);c.drawString(30,700,'Ocean salinity varies with evaporation.');c.save()
    doc_id,_=store.import_file('ocean.pdf',raw.getvalue(),'live',task_stage='corpus')
    corpus.enqueue(store,doc_id);assert Worker(store).run_once()
    version=corpus.listing(store)['items'][0]['id']
    response=client.get('/api/corpus/csv/'+version)
    assert response.status_code==200 and response.content.startswith(b'\xef\xbb\xbf')
    row=next(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
    assert list(row)==['paper_id','title','pdf_path','md_path','images_dir','image_count']
    assert row['pdf_path']=='Paper/Pdf/ocean.pdf'
    with zipfile.ZipFile(io.BytesIO(client.get('/api/corpus/export/'+version).content)) as z:
        assert z.read('Paper/CleanedData/papers.csv')==response.content
        assert z.read(row['pdf_path'])==raw.getvalue()
        assert z.read(row['md_path'])
        assert row['images_dir']=='' and row['image_count']=='0'
