import contextlib
import csv
import io
import json
from pathlib import Path
import secrets
import time

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .demo import seed_demo
from .pipeline import Worker, approve, approve_document, reject
from .settings import ROOT, FIELDS, settings, api_key, image_media_type, MIN_DOCUMENT_TOPICS, MAX_DOCUMENT_TOPICS, MAX_DOCUMENT_SEARCH_TERMS
from .store import Store, uid
from .semantic import SemanticSearch, SearchCoordinator


class CommitBody(BaseModel):
    reviewed: bool
    expected_hash: str


class EditVersion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str
    expected_hash: str
    input: str = Field(min_length=1, max_length=20000)
    thinking: str = Field(min_length=1, max_length=20000)
    output: str = Field(min_length=1, max_length=20000)


class EditBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    versions: list[EditVersion] = Field(min_length=1, max_length=2)


def create_app(runtime=None, seed=False, workers=True):
    runtime = Path(runtime or ROOT/'runtime')
    stores = {'live':Store(runtime/'live')}
    if seed:
        stores['demo']=Store(runtime/'demo')
        seed_demo(stores['demo'])
    searches = {mode: SemanticSearch(store) for mode, store in stores.items()}
    search_runner = SearchCoordinator(searches.values())
    runners = [Worker(store, before_process=search_runner.interrupt) for store in stores.values()] if workers else []
    session = secrets.token_urlsafe(24)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        for worker in runners: worker.start()
        if workers: search_runner.start()
        try:
            yield
        finally:
            for worker in runners: worker.stop.set()
            search_runner.close()

    app=FastAPI(title='QAproject',lifespan=lifespan)
    app.state.stores=stores
    app.state.session=session
    app.state.searches=searches
    app.state.search_runner=search_runner

    @app.middleware('http')
    async def local_access(request, call_next):
        host=request.headers.get('host','').split(':')[0]
        if host not in {'127.0.0.1','localhost','testserver'}:
            return JSONResponse({'detail':'仅允许本机访问。'},status_code=403)
        if request.method not in {'GET','HEAD','OPTIONS'}:
            if request.headers.get('x-qa-session') != session:
                return JSONResponse({'detail':'会话已更新，请刷新页面。'},status_code=403)
            origin=request.headers.get('origin')
            if origin and origin not in {f'http://{request.headers.get("host")}'}:
                return JSONResponse({'detail':'来源校验失败。'},status_code=403)
        return await call_next(request)

    def workspace(mode):
        if mode not in stores: raise HTTPException(400,'工作区无效')
        return stores[mode]

    @app.exception_handler(ValueError)
    async def validation_error(request,error):
        return JSONResponse({'detail':str(error)},status_code=400)

    @app.get('/api/bootstrap')
    def bootstrap():
        cfg=settings()
        return {'session':session,'key_configured':bool(api_key()),'base_url':cfg['base_url'],
                'text_model':cfg['text_model'],'vision_model':cfg.get('vision_model'),
                'limits':{'max_file_mb':cfg['max_file_mb'],'max_pages':cfg['max_pages'],
                          'min_topics_per_document':MIN_DOCUMENT_TOPICS,'max_topics_per_document':MAX_DOCUMENT_TOPICS,
                          'max_search_terms_per_document':MAX_DOCUMENT_SEARCH_TERMS}}

    @app.get('/api/documents')
    def documents(mode='live', summary_only:bool=False):
        return workspace(mode).documents(summary_only=summary_only)

    @app.get('/api/documents/{doc_id}')
    def document(doc_id,mode='live'):
        return workspace(mode).document(doc_id)

    @app.get('/api/topics')
    def topics(mode='live'):
        return workspace(mode).topics()

    @app.get('/api/search')
    def search(q='',mode='live'):
        if len(q)>200: raise HTTPException(400,'查询过长')
        return workspace(mode).search(q)

    @app.get('/api/search/semantic')
    def semantic_search(q='',mode='live'):
        if len(q)>200: raise HTTPException(400,'查询过长')
        workspace(mode)
        return searches[mode].recommend(q)

    @app.post('/api/imports')
    async def imports(request:Request,mode='live'):
        store=workspace(mode)
        # Multipart data is streamed by Starlette into a temporary spool, then bounded per file.
        form=await request.form(max_files=20,max_fields=5)
        files=[value for _,value in form.multi_items() if hasattr(value,'read') and hasattr(value,'filename')]
        if not files: raise HTTPException(400,'请选择文件')
        limit=settings()['max_file_mb']*1024*1024
        results=[]
        for file in files:
            if Path(file.filename).suffix.lower() not in {'.pdf','.png','.jpg','.jpeg','.webp','.txt','.md','.csv'}:
                results.append({'name':file.filename,'error':'该格式尚未支持'});continue
            data=await file.read(limit+1)
            if not data or len(data)>limit:
                results.append({'name':file.filename,'error':'文件为空或超出大小限制'});continue
            doc_id,created=store.import_file(file.filename,data,mode)
            results.append({'id':doc_id,'name':file.filename,'created':created})
        return results

    @app.get('/api/assets/{asset_id}')
    def asset(asset_id,mode='live'):
        store=workspace(mode)
        with store.connect() as db:
            row=db.execute('SELECT path,hash FROM assets WHERE id=?',(asset_id,)).fetchone()
        if not row: raise HTTPException(404,'图片不存在')
        path=Path(row['path']).resolve()
        if not path.is_relative_to(store.root/'assets/images') or not path.is_file():
            raise HTTPException(404,'图片存储无效')
        return FileResponse(path,media_type=image_media_type(path))

    @app.get('/api/originals/{doc_id}')
    def original(doc_id,mode='live'):
        store=workspace(mode)
        doc=store.document(doc_id)
        path=Path(doc['path']).resolve()
        if not path.is_relative_to(store.root/'originals') or not path.is_file():
            raise HTTPException(404,'文件不存在')
        return FileResponse(path,filename=doc['name'])

    @app.post('/api/candidates/{candidate_id}/approve')
    def commit(candidate_id,body:CommitBody,mode='live'):
        if mode=='demo': raise HTTPException(400,'演示记录不执行正式提交')
        return approve(workspace(mode),candidate_id,body.reviewed,body.expected_hash)

    @app.post('/api/documents/{doc_id}/approve')
    def commit_document(doc_id,mode='live'):
        if mode!='live': raise HTTPException(400,'演示记录不执行正式提交')
        return approve_document(workspace(mode),doc_id)

    @app.post('/api/documents/{doc_id}/approve_all')
    def commit_all(doc_id,mode='live'):
        result = commit_document(doc_id,mode)
        return {'approved':result['committed_count'], 'failed':[]}

    @app.post('/api/candidates/{candidate_id}/reject')
    def reject_candidate(candidate_id,body:CommitBody,mode='live'):
        if mode=='demo': raise HTTPException(400,'演示记录不执行正式审核')
        return reject(workspace(mode),candidate_id,body.expected_hash)

    @app.post('/api/candidates/{candidate_id}/rereview')
    def rereview(candidate_id,body:CommitBody,mode='live'):
        if mode!='live': raise HTTPException(400,'测试样例不执行正式审核')
        store=workspace(mode)
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            c=db.execute('SELECT * FROM candidates WHERE id=?',(candidate_id,)).fetchone()
            if not c or c['row_hash']!=body.expected_hash: raise ValueError('候选版本已变更，请刷新。')
            if c['status'] in ('approved','rejected'): raise ValueError('该候选已完成处理，无法重新核验。')
            status=db.execute('SELECT status FROM documents WHERE id=?',(c['document_id'],)).fetchone()[0]
            if status not in ('waiting_review','completed_under_min','completed'):
                raise ValueError('请先等待文件处理完成；失败文件请使用重试恢复原任务。')
            if db.execute("SELECT 1 FROM tasks WHERE document_id=? AND status IN ('queued','running')",(c['document_id'],)).fetchone():
                raise ValueError('该文件正在处理，请等待当前任务完成。')
            db.execute('INSERT INTO tasks(id,document_id,status,review_candidate_id) VALUES(?,?,?,?)',
                       (uid(),c['document_id'],'queued',candidate_id))
            db.execute("UPDATE documents SET status='queued',progress='等待重新核验本组' WHERE id=?",(c['document_id'],))
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(uid(),c['document_id'],'rereview_requested',
                       json.dumps({'candidate_id':candidate_id,'row_hash':body.expected_hash}),time.time()))
        return {'queued':True}

    @app.post('/api/documents/{doc_id}/retry')
    def retry(doc_id,mode='live'):
        store=workspace(mode)
        doc=store.document(doc_id)
        if doc['status']!='failed': raise HTTPException(400,'仅失败任务可重试')
        store.event(doc_id, 'retry_requested', {'previous_error':doc['error'], 'previous_seconds':doc['seconds'], 'timings':doc['timings']})
        with store.connect() as db:
            db.execute("UPDATE tasks SET status='queued',worker=NULL,lease_until=0 WHERE document_id=? AND status='failed'",(doc_id,))
        store.update_document(doc_id,status='queued',error='')
        return {'queued':True}

    @app.post('/api/documents/{doc_id}/optimize')
    def optimize(doc_id,mode='live'):
        if mode!='live': raise HTTPException(400,'测试样例不执行正式优化')
        store=workspace(mode)
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            document=db.execute('SELECT status FROM documents WHERE id=?',(doc_id,)).fetchone()
            if not document: raise ValueError('找不到文件')
            if document['status'] not in ('waiting_review','insufficient','completed_under_min'):
                raise ValueError('请先等待当前任务完成；失败文件请使用重试恢复。')
            if db.execute("SELECT 1 FROM tasks WHERE document_id=? AND status IN ('queued','running')",(doc_id,)).fetchone():
                raise ValueError('该文件正在处理，请等待当前任务完成。')
            db.execute('INSERT INTO tasks(id,document_id,status,stage) VALUES(?,?,?,?)',(uid(),doc_id,'queued','quality_queued'))
            db.execute("UPDATE documents SET status='queued',progress='等待修订未通过的问答并重新审核' WHERE id=?",(doc_id,))
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(uid(),doc_id,'optimization_requested','{}',time.time()))
        return {'queued':True}

    @app.get('/api/qas')
    def qa_list(mode='live',page:int=Query(1,ge=1),page_size:int=Query(20,ge=1,le=100)):
        from .qa_summary import list_qas
        return list_qas(workspace(mode),page,page_size)

    @app.get('/api/qas/{qa_id}')
    def qa_detail(qa_id,mode='live'):
        from .qa_summary import edit_detail
        return edit_detail(workspace(mode),qa_id)

    @app.post('/api/qas/{qa_id}/revisions')
    def qa_edit(qa_id,body:EditBody,mode='live'):
        from .qa_summary import queue_revision
        if mode!='live': raise HTTPException(400,'演示记录不执行正式修改')
        return queue_revision(workspace(mode),qa_id,[v.model_dump() for v in body.versions])

    @app.post('/api/qas/{qa_id}/retry')
    def qa_edit_retry(qa_id,mode='live'):
        from .qa_summary import retry_revision
        if mode!='live': raise HTTPException(400,'演示记录不执行正式修改')
        return retry_revision(workspace(mode),qa_id)

    @app.get('/api/export')
    def export(mode='live',topic_id=None,document_id=None):
        store=workspace(mode)
        with store.connect() as db:
            query='''SELECT q.row_json FROM qas q JOIN candidates c ON c.id=q.id
                LEFT JOIN document_topic_redirects m ON m.document_id=c.document_id AND m.old_topic_id=c.topic_id WHERE 1=1'''
            args=[]
            if topic_id: query+=' AND COALESCE(m.topic_id,c.topic_id)=?';args.append(topic_id)
            if document_id: query+=' AND c.document_id=?';args.append(document_id)
            rows=[json.loads(r[0]) for r in db.execute(query+' ORDER BY q.created',args)]
        content=io.StringIO(newline='')
        writer=csv.DictWriter(content,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
        name=('demo-QA' if mode=='demo' else 'QA')+'.csv'
        return Response(content.getvalue().encode('utf-8-sig'),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="{name}"'})

    app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
    return app
