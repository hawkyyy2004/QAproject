"""HTTP contracts for versioned corpus and instruction production."""
from pathlib import Path
import io
import zipfile
from typing import Literal, Annotated
import math

from fastapi import Query, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import corpus, instructions
from . import web_sources
from .store import digest


class StrictBody(BaseModel):
    model_config = ConfigDict(extra='forbid',str_strip_whitespace=True)


RecordID = Annotated[str,Field(min_length=1,max_length=250)]


class CorpusBody(StrictBody):
    kind: Literal['paper','web','image','data'] | None = None
    title: str = Field(default='',max_length=500)
    source_label: str = Field(default='',max_length=200)
    generate_after: bool = False


class WebBody(StrictBody):
    url: str = Field(min_length=8,max_length=4000)
    title: str = Field(default='',max_length=500)
    source_label: str = Field(default='',max_length=200)
    extract_knowledge: bool = True


class InstructionBody(StrictBody):
    version_id: str = Field(min_length=1,max_length=128)
    subset: Literal['Science','MultiModel-Science','Sonar-Open','Bio']
    target: int = Field(default=5,ge=1,le=12,strict=True)
    source_label: str = Field(default='',max_length=200)
    node_ids: list[RecordID] = Field(default_factory=list,max_length=20)
    supplement_version_ids: list[RecordID] = Field(default_factory=list,max_length=5)


class AnnotationBody(StrictBody):
    unit_id: str = Field(min_length=1,max_length=250)
    kind: Literal['label','bbox','caption']
    value: dict
    verified: bool = False
    source_note: str = Field(min_length=1,max_length=1000)

    @model_validator(mode='after')
    def validate_value(self):
        if self.kind in {'label','caption'}:
            value=self.value.get(self.kind)
            if not isinstance(value,str) or not value.strip() or len(value)>5000:
                raise ValueError('类别或图注必须是非空文字，最多5000字符。')
            if set(self.value)-{self.kind}: raise ValueError('标注包含未知字段。')
        else:
            box=self.value.get('box')
            if (not isinstance(box,list) or len(box)!=4 or any(type(v) not in {int,float} or not math.isfinite(v) for v in box)
                or box[0]<0 or box[1]<0 or box[2]<=box[0] or box[3]<=box[1]):
                raise ValueError('检测框须为有限的 [x1,y1,x2,y2]，右下坐标必须大于左上坐标。')
            coordinates=self.value.get('coordinates')
            if coordinates not in {'normalized','pixel'}: raise ValueError('请明确检测框坐标格式。')
            if coordinates=='normalized' and any(v>1 for v in box): raise ValueError('归一化坐标必须在0到1之间。')
            if set(self.value)-{'box','coordinates','label'}: raise ValueError('检测框包含未知字段。')
            if 'label' in self.value and (not isinstance(self.value['label'],str) or len(self.value['label'])>500):
                raise ValueError('检测框类别无效。')
        return self


class NodeBody(StrictBody):
    name: str = Field(min_length=1,max_length=100)
    definition: str = Field(min_length=1,max_length=1000)
    parent_id: RecordID | None = None
    source_version_id: str = Field(min_length=1,max_length=128)


class AnnotationImport(StrictBody):
    records: list[AnnotationBody] = Field(min_length=1,max_length=100)


class DecisionBody(StrictBody):
    expected_hash: str = Field(min_length=64,max_length=64)


class InstructionEdit(DecisionBody):
    input: str = Field(min_length=1,max_length=8000)
    output: str = Field(min_length=1,max_length=8000)
    thinking: str = Field(default='',max_length=2000)
    language: Literal['zh','en']
    task_type: Literal['science','description','classification','localization']


def archive_response(result, filename):
    if isinstance(result,(str,Path)):
        return FileResponse(result,media_type='application/zip',filename=filename)
    return Response(result,media_type='application/zip',headers={
        'Content-Disposition':f'attachment; filename="{filename}"'})


def reconcile_chains(store, document_id=None):
    """Recover a completed Corpus's requested downstream work after any restart."""
    with store.connect() as db:
        where="status='completed' AND generate_after=1 AND version_id IS NOT NULL"
        args=[]
        if document_id:
            where+=' AND document_id=?';args.append(document_id)
        rows=[dict(r) for r in db.execute('SELECT document_id,version_id FROM corpus_jobs WHERE '+where,args)]
    results=[]
    for job in rows:
        try:
            version=corpus.verify(store,job['version_id'])
            has_text=any(not u.get('image') and u.get('text','').strip() for u in version['units'])
            subset='Science' if has_text else 'MultiModel-Science'
            result=instructions.enqueue(store,version['id'],subset,source_label=version.get('source_label',''))
            with store.connect() as db:
                db.execute('UPDATE corpus_jobs SET generate_after=0 WHERE document_id=? AND version_id=?',
                           (job['document_id'],job['version_id']))
            results.append(result)
        except (ValueError,OSError) as error:
            # Leave the persisted intention intact, and explain blocked work.
            with store.connect() as db:
                db.execute('UPDATE corpus_jobs SET error=? WHERE document_id=?',
                           ('下游指令尚未启动：'+str(error),job['document_id']))
    return results


def register_routes(app, workspace):
    @app.get('/api/corpus')
    def corpus_list(q:str=Query('',max_length=200),kind='',page:int=Query(1,ge=1),
                    page_size:int=Query(20,ge=1,le=100),mode='live'):
        return corpus.listing(workspace(mode),q=q,kind=kind,page=page,page_size=page_size)

    @app.get('/api/corpus/jobs')
    def corpus_jobs(mode='live'):
        return corpus.jobs(workspace(mode))

    @app.get('/api/corpus/nodes')
    def nodes(mode='live'):
        return instructions.list_nodes(workspace(mode))

    @app.post('/api/corpus/nodes')
    def add_node(body:NodeBody,mode='live'):
        return instructions.add_node(workspace(mode),**body.model_dump())

    @app.post('/api/corpus/web')
    def capture_web(body:WebBody,mode='live'):
        if mode!='live': raise HTTPException(400,'仅在真实工作区接入资料。')
        store=workspace(mode)
        page=web_sources.capture(body.url)
        name='web-'+digest(page['url'])[:12]+page['suffix']
        document_id,created=store.import_file(name,page['raw'],'live',task_stage='corpus')
        corpus.register_source(store,document_id,'web',title=body.title or page['title'],
                               source_url=page['url'],source_label=body.source_label)
        result=corpus.enqueue(store,document_id,kind='web')
        if body.extract_knowledge:
            from .knowledge import enqueue
            enqueue(store,document_id)
        return dict(result,id=document_id,document_id=document_id,created=created,
                    source_url=page['url'],title=body.title or page['title'])

    @app.post('/api/documents/{document_id}/corpus')
    def prepare_corpus(document_id:str,body:CorpusBody,mode='live'):
        if mode!='live': raise HTTPException(400,'仅在真实工作区整理资料。')
        store=workspace(mode)
        result=corpus.enqueue(store,document_id,**body.model_dump(exclude_none=True))
        reconcile_chains(store,document_id)
        return result

    @app.get('/api/corpus/export/{version_id}')
    def export_corpus(version_id:str,mode='live'):
        return archive_response(corpus.export_official_bundle(workspace(mode),version_id),'OceanCorpus.zip')

    @app.get('/api/corpus/csv/{version_id}')
    def export_corpus_csv(version_id:str,mode='live'):
        content,name=corpus.export_csv(workspace(mode),version_id)
        return Response(content,media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="{name}"'})

    @app.get('/api/corpus/{version_id}/annotations')
    def annotations(version_id:str,mode='live'):
        return instructions.list_annotations(workspace(mode),version_id)

    @app.post('/api/corpus/{version_id}/annotations')
    def add_annotation(version_id:str,body:AnnotationBody,mode='live'):
        return instructions.add_annotation(workspace(mode),version_id,**body.model_dump())

    @app.post('/api/corpus/{version_id}/annotations/import')
    def import_annotations(version_id:str,body:AnnotationImport,mode='live'):
        store=workspace(mode)
        corpus.verify(store,version_id)
        results=[]
        for index,item in enumerate(body.records):
            try:
                results.append(dict(index=index,result=instructions.add_annotation(
                    store,version_id,**item.model_dump())))
            except ValueError as error:
                results.append(dict(index=index,error=str(error)))
        return dict(items=results,imported=sum('error' not in r for r in results),
                    failed=sum('error' in r for r in results))

    @app.get('/api/corpus/{version_id}')
    def corpus_detail(version_id:str,mode='live'):
        return corpus.detail(workspace(mode),version_id)

    @app.get('/api/instructions')
    def instruction_list(q:str=Query('',max_length=200),subset='',status='',
                         page:int=Query(1,ge=1),page_size:int=Query(20,ge=1,le=100),mode='live'):
        return instructions.listing(workspace(mode),q=q,subset=subset,status=status,page=page,page_size=page_size)

    @app.get('/api/instructions/jobs')
    def instruction_jobs(mode='live'):
        return instructions.jobs(workspace(mode))

    @app.post('/api/instructions/jobs/{job_id}/retry')
    def retry_instruction_job(job_id:str,mode='live'):
        return instructions.retry_job(workspace(mode),job_id)

    @app.get('/api/instructions/export')
    def instruction_export(subset:Literal['Science','MultiModel-Science','Sonar-Open','Bio'],mode='live'):
        return archive_response(instructions.export_bundle(workspace(mode),subset),'OceanInstruction-'+subset+'.zip')

    @app.get('/api/instructions/csv')
    def instruction_csv(subset:Literal['Science','MultiModel-Science','Sonar-Open','Bio'],mode='live'):
        with zipfile.ZipFile(io.BytesIO(instructions.export_bundle(workspace(mode),subset))) as z:
            content=z.read(subset+'/data.csv')
        return Response(content,media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="{subset}-data.csv"'})

    @app.post('/api/instructions')
    def generate_instructions(body:InstructionBody,mode='live'):
        if mode!='live': raise HTTPException(400,'仅在真实工作区生成指令。')
        return instructions.enqueue(workspace(mode),**body.model_dump())

    @app.post('/api/instructions/{candidate_id}/commit')
    def commit_instruction(candidate_id:str,body:DecisionBody,mode='live'):
        return instructions.commit(workspace(mode),candidate_id,body.expected_hash)

    @app.post('/api/instructions/{candidate_id}/reject')
    def reject_instruction(candidate_id:str,body:DecisionBody,mode='live'):
        return instructions.reject(workspace(mode),candidate_id,body.expected_hash)

    @app.post('/api/instructions/{candidate_id}/edit')
    def edit_instruction(candidate_id:str,body:InstructionEdit,mode='live'):
        store=workspace(mode)
        result=instructions.edit(store,candidate_id,body.model_dump(exclude={'expected_hash'}),body.expected_hash)
        return dict(result,task=instructions.review_enqueue(store,candidate_id,result['hash']))

    @app.post('/api/instructions/{candidate_id}/review')
    def review_instruction(candidate_id:str,body:DecisionBody,mode='live'):
        return instructions.review_enqueue(workspace(mode),candidate_id,body.expected_hash)

    @app.get('/api/instructions/{candidate_id}')
    def instruction_detail(candidate_id:str,mode='live'):
        return instructions.detail(workspace(mode),candidate_id)
