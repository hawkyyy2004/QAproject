"""Offline instruction contracts; no simulated result is a real GLM acceptance."""
import csv
import io
import json
import threading
import zipfile
from contextlib import contextmanager

from PIL import Image
import pytest

from qaproject import corpus, instructions as ins
from qaproject.execution import ExecutionLimits
from qaproject.glm import ProviderError
from qaproject.pipeline import Worker
from qaproject.store import Store, digest, encode

REAL_GLM_CALL = ins.GLMClient.call


@pytest.fixture
def env(tmp_path, monkeypatch):
    store=Store(tmp_path/'data'); versions={};calls=[]
    def add(name='sea.txt', text='海水温度影响密度。', image=False, count=1, color='blue'):
        raw=io.BytesIO()
        if image:
            Image.new('RGB',(30,20),color).save(raw,format='PNG'); content=raw.getvalue();name=name+'.png'
        else: content=(name+text).encode()
        doc_id,_=store.import_file(name,content,'live')
        with store.connect() as db: db.execute("UPDATE tasks SET status='succeeded' WHERE document_id=?",(doc_id,))
        units=[]
        for i in range(count):
            unit=dict(id=doc_id+':'+str(i),text=text+str(i),image=store.document(doc_id)['path'] if image else '',
                      locator='image-0001' if image else 'seg-'+str(i),page=0,seg=i)
            unit['hash']=digest(unit);units.append(unit)
        version_id='v'+doc_id
        versions[version_id]=dict(document_id=doc_id,units=units,version_hash=digest(units))
        return version_id
    monkeypatch.setattr(corpus,'verify',lambda store,version_id:json.loads(encode(versions[version_id])))
    def call(self,prompt,payload,units=(),**kwargs):
        calls.append((prompt,payload,units))
        if 'checks' in prompt and 'records' in payload:
            return dict(checks=[dict(id=r['id'],note='模拟审核',**{k:True for k in ins.CHECKS}) for r in payload['records']]),dict(finish_reason='stop')
        image=next((u for u in units if u.get('image')),None)
        subset=payload['subset']
        record=dict(input='这份实际证据说明什么？'+str(len(calls)),output='温度影响密度。',language='zh',thinking='',
                    evidence_ids=[u['id'] for u in units],image_unit_id=image['id'] if image else '',task_type='description' if image else 'science')
        if subset=='Bio': record.update(output='海豚',task_type='classification')
        return dict(records=[record]),dict(finish_reason='stop')
    monkeypatch.setattr(ins.GLMClient,'call',call)
    return store,add,versions,calls,call


def run(store):
    worker=Worker(store)
    task=worker.limits.claim(store,worker.name)
    assert task and task['stage']=='instruction'
    try: ins.process(worker,task)
    finally: worker.limits.release(store,task)
    return task


def first(store): return ins.listing(store)['items'][0]


def test_independent_commit_headers_idempotence_and_qa_unchanged(env):
    store,add,versions,calls,_=env; version=add()
    job=ins.enqueue(store,version,'Science',target=1)
    assert ins.enqueue(store,version,'Science',target=1)==job
    run(store)
    value=first(store)
    assert value['status']=='approved' and len(calls)==2
    assert ins.export_bundle(store,'Science')
    with pytest.raises(ins.InstructionError):ins.commit(store,value['id'],'wrong')
    ins.commit(store,value['id'],value['hash'])
    data=ins.detail(store,value['id'])
    assert data['source_valid'] and data['review_history'] and data['source_kind']=='system_lineage'
    with zipfile.ZipFile(io.BytesIO(ins.export_bundle(store,'Science'))) as z:
        rows=list(csv.DictReader(io.StringIO(z.read('Science/data.csv').decode('utf-8-sig'))))
    assert len(rows)==1 and list(rows[0])==ins.FIELDS['Science'] and rows[0]['thinking']==''
    assert rows[0]['source'].startswith('document:')
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM qas').fetchone()[0]==0
        assert db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]==0
    assert ins.enqueue(store,version,'Science',target=1)['job_id']==job['job_id']
    assert ins.enqueue(store,version,'Science',target=2)['job_id']!=job['job_id']


@pytest.mark.parametrize('subset',['MultiModel-Science','Sonar-Open','Bio'])
def test_true_images_review_zip_and_api_annotation_shape(env,subset):
    store,add,versions,calls,_=env; version=add(image=True); unit=versions[version]['units'][0]
    if subset=='Bio':
        a=ins.add_annotation(store,version,unit['id'],'label',{'label':'海豚'},True,'人工核对来源')
        assert a['value']=='海豚' and ins.list_annotations(store,version)[0]['verified']
    ins.enqueue(store,version,subset,target=1,source_label='团队原始标注集')
    run(store);value=first(store)
    assert value['status']=='approved'
    assert all(any(u['image'] for u in units) for _,_,units in calls)
    assert PathBytes(calls[0][2][0]['image'])
    ins.commit(store,value['id'],value['hash'])
    with zipfile.ZipFile(io.BytesIO(ins.export_bundle(store,subset))) as z:
        raw_csv=z.read(subset+'/data.csv')
        assert not raw_csv.startswith(b'\xef\xbb\xbf')
        rows=list(csv.DictReader(io.StringIO(raw_csv.decode('utf-8'))))
        assert list(rows[0])==ins.FIELDS[subset]
        if 'language' in rows[0]:assert rows[0]['language']=='Chinese'
        assert rows[0]['image_path'].startswith({'MultiModel-Science':'Visual Data/image0/',
            'Sonar-Open':'Task-specific Data-Sonar/image0/','Bio':'Task-specific Data-RGB/image/'}[subset])
        assert z.read(subset+'/'+rows[0]['image_path'])==PathBytes(unit['image'])
        assert '..' not in rows[0]['image_path']


def PathBytes(path):
    from pathlib import Path
    return Path(path).read_bytes()


def test_missing_original_keeps_history_but_blocks_commit_export(env):
    from pathlib import Path
    store,add,versions,_,_=env;v=add();ins.enqueue(store,v,'Science',target=1);run(store)
    value=first(store);ins.commit(store,value['id'],value['hash'])
    Path(store.document(versions[v]['document_id'])['path']).unlink()
    historical=ins.detail(store,value['id'])
    assert historical['evidence'] and historical['source_error'] and not historical['source_valid']
    with pytest.raises(ins.InstructionError):ins.commit(store,value['id'],value['hash'])
    with pytest.raises(ins.InstructionError):ins.export_bundle(store,'Science')


@pytest.mark.parametrize('change',['supplement','node','annotation','candidate','review','asset'])
def test_binding_drift_fails_closed(env,change):
    from pathlib import Path
    store,add,versions,_,_=env
    v=add(image=change in ('annotation','asset'));supp=add('supplement.txt')
    node=ins.add_node(store,'密度','定义',v)
    subset='Sonar-Open' if change in ('annotation','asset') else 'Science'
    if change=='annotation':
        annotation=ins.add_annotation(store,v,versions[v]['units'][0]['id'],'caption','人工图注',True,'真实来源')
    ins.enqueue(store,v,subset,target=1,node_ids=[node['id']],supplement_version_ids=[supp]);run(store)
    value=first(store)
    if change=='supplement': versions[supp]['units'][0]['text']='修改'
    elif change=='node':ins.update_node(store,node['id'],{'definition':'变更'})
    elif change=='annotation':
        with store.connect() as db:
            db.execute('UPDATE corpus_annotations SET value_json=? WHERE id=?',(encode('篡改图注'),annotation['id']))
    elif change=='asset':Path(versions[v]['units'][0]['image']).write_bytes(b'changed')
    else:
        with store.connect() as db:
            field='row_json' if change=='candidate' else 'review_json'
            db.execute('UPDATE instructions_candidates SET '+field+"='{}' WHERE id=?",(value['id'],))
    assert not ins.detail(store,value['id'])['source_valid']
    with pytest.raises(ins.InstructionError):ins.commit(store,value['id'],value['hash'])


def test_annotations_validation_and_nodes_cycle(env):
    store,add,versions,_,_=env;v=add(image=True);u=versions[v]['units'][0]['id']
    for box in ([0,0,31,20],[0,0,float('nan'),1],[1,0,0,1]):
        with pytest.raises(ins.InstructionError):ins.add_annotation(store,v,u,'bbox',{'box':box,'coordinates':'pixel'},True,'来源')
    box=ins.add_annotation(store,v,u,'bbox',{'box':[0,0,30,20],'coordinates':'pixel'},True,'来源')
    assert box['value']['width']==30
    assert ins.add_annotation(store,v,u,'bbox',{'box':[0,0,30,20],'coordinates':'pixel'},True,'来源')['id']==box['id']
    with pytest.raises(ins.InstructionError):ins.add_annotation(store,v,u,'label','海豚',True,'')
    with pytest.raises(ins.InstructionError):ins.add_annotation(store,v,'unknown','label','海豚',True,'人工')
    root=ins.add_node(store,'动物','真实定义',v);child=ins.add_node(store,'海豚','真实定义',v,root['id'])
    with pytest.raises(ins.InstructionError):ins.update_node(store,root['id'],dict(parent_id=child['id']))
    with pytest.raises(ins.InstructionError):ins.delete_node(store,root['id'])
    ins.delete_node(store,child['id']);ins.delete_node(store,root['id'])


def test_subset_preconditions_and_unseen_unit(env,monkeypatch):
    store,add,versions,_,provider=env;v=add(image=True)
    with pytest.raises(ins.InstructionError):ins.enqueue(store,v,'Science')
    with pytest.raises(ins.InstructionError):ins.enqueue(store,v,'Bio')
    v=add('text.txt')
    with pytest.raises(ins.InstructionError):ins.enqueue(store,v,'MultiModel-Science')
    ins.enqueue(store,v,'Science',target=1)
    def bad(self,prompt,payload,units=(),**kw):
        result,stats=provider(self,prompt,payload,units,**kw);result['records'][0]['evidence_ids']=['unseen'];return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',bad)
    run(store)
    assert ins.jobs(store)[0]['status']=='failed' and not ins.listing(store)['items']


def test_review_failure_two_repairs_and_history(env,monkeypatch):
    store,add,versions,calls,provider=env;v=add();ins.enqueue(store,v,'Science',target=1)
    def fail(self,prompt,payload,units=(),**kw):
        result,stats=provider(self,prompt,payload,units,**kw)
        if 'checks' in result:
            result['checks'][0]['supported']=False
        return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',fail);run(store)
    value=first(store);detail=ins.detail(store,value['id'])
    assert value['status']=='failed' and len(detail['review_history'])==3
    assert len(calls)==6  # generate, three reviews, two repairs
    with pytest.raises(ins.InstructionError):ins.commit(store,value['id'],value['hash'])
    assert len(ins.listing(store,status='failed')['items'])==1


def test_retry_preserves_generation_and_edit_invalidates_review(env,monkeypatch):
    store,add,versions,calls,provider=env;v=add();job=ins.enqueue(store,v,'Science',target=1)
    def fail(self,prompt,payload,units=(),**kw):
        if 'checks' in prompt:raise ProviderError('模拟接口失败','network',{'finish_reason':'network'})
        return provider(self,prompt,payload,units,**kw)
    monkeypatch.setattr(ins.GLMClient,'call',fail);run(store)
    assert first(store)['status']=='pending' and len(calls)==1
    monkeypatch.setattr(ins.GLMClient,'call',provider)
    ins.retry_job(store,job['job_id']);run(store)
    assert len(calls)==2 and first(store)['status']=='approved'
    old=first(store);edited=ins.edit(store,old['id'],{'output':'修改后的证据答案'},old['hash'])
    assert edited['hash']!=old['hash'] and ins.detail(store,old['id'])['review']=={}
    with pytest.raises(ins.InstructionError):ins.commit(store,old['id'],old['hash'])
    ins.review_enqueue(store,old['id'],edited['hash']);run(store)
    assert first(store)['status']=='approved' and len(ins.detail(store,old['id'])['review_history'])==2


def test_batch_inputs_target_resume_and_referenced_review_only(env,monkeypatch):
    store,add,versions,calls,provider=env
    v=add(count=10,text='真实全文。'*600);supp=add('supplement.txt')
    ins.enqueue(store,v,'Science',target=5,supplement_version_ids=[supp]);run(store)
    assert ins.listing(store)['total']<=5
    for prompt,payload,units in calls:
        if prompt==ins.GENERATE:
            assert len(units)<=6 and sum(len(u.get('text','')) for u in units)<=16000
            assert all(not u['image'] for u in units)
        if prompt==ins.REVIEW:
            assert {u['id'] for u in units}=={dict((f'u{i}',u['id']) for i,u in enumerate(units,1)).get(x,x) for x in payload['records'][0]['evidence_ids']}
    assert any(any(u['id']==versions[supp]['units'][0]['id'] for u in us) for _,_,us in calls)


def test_expired_lease_writes_nothing(env):
    store,add,versions,calls,_=env;v=add();ins.enqueue(store,v,'Science',target=1)
    worker=Worker(store);task=store.claim(worker.name)
    with store.connect() as db:db.execute('UPDATE tasks SET lease_until=0 WHERE id=?',(task['id'],))
    ins.process(worker,task)
    assert not calls and not ins.listing(store)['items']


def test_target_accumulates_and_failed_batch_resumes_without_duplication(env,monkeypatch):
    store,add,versions,calls,provider=env;v=add(count=8,text='真实文字。'*500)
    job=ins.enqueue(store,v,'Science',target=4)
    generation_count=0
    def interrupted(self,prompt,payload,units=(),**kw):
        nonlocal generation_count
        if prompt==ins.GENERATE:
            generation_count+=1
            if generation_count==2: raise ProviderError('模拟第二批失败','network',{})
        return provider(self,prompt,payload,units,**kw)
    monkeypatch.setattr(ins.GLMClient,'call',interrupted);run(store)
    assert ins.listing(store)['total']==1
    old_id=first(store)['id']
    monkeypatch.setattr(ins.GLMClient,'call',provider)
    ins.retry_job(store,job['job_id']);run(store)
    assert ins.listing(store)['total']==4
    assert old_id in {c['id'] for c in ins.listing(store)['items']}
    assert len([c for c in calls if c[0]==ins.GENERATE])==4
    assert all(c['status']=='approved' for c in ins.listing(store)['items'])
    assert ins.jobs(store)[0]['target']==4 and 'evidence_json' not in ins.jobs(store)[0]


def test_registered_but_unseen_unit_is_rejected(env,monkeypatch):
    store,add,versions,_,provider=env;v=add(count=8,text='长单元内容。'*900)
    ins.enqueue(store,v,'Science',target=1)
    def bad(self,prompt,payload,units=(),**kw):
        result,stats=provider(self,prompt,payload,units,**kw)
        result['records'][0]['evidence_ids']=[versions[v]['units'][-1]['id']]
        return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',bad);run(store)
    assert ins.jobs(store)[0]['status']=='failed' and not ins.listing(store)['items']


def test_visual_batches_single_image_and_review_only_refs(env):
    store,add,versions,calls,_=env;v=add(image=True,count=3)
    ins.enqueue(store,v,'Sonar-Open',target=3);run(store)
    assert ins.listing(store)['total']==3
    for prompt,payload,units in calls:
        assert sum(bool(u['image']) for u in units)==1
        assert len(units)<=6
        if prompt==ins.REVIEW:
            assert {u['id'] for u in units}=={dict((f'u{i}',u['id']) for i,u in enumerate(units,1)).get(x,x) for x in payload['records'][0]['evidence_ids']}


def test_reject_during_inflight_review_is_not_overwritten(env,monkeypatch):
    store,add,versions,_,provider=env;v=add();ins.enqueue(store,v,'Science',target=1)
    def changed(self,prompt,payload,units=(),**kw):
        if prompt==ins.REVIEW:
            value=first(store);ins.reject(store,value['id'],value['hash'])
        return provider(self,prompt,payload,units,**kw)
    monkeypatch.setattr(ins.GLMClient,'call',changed);run(store)
    assert first(store)['status']=='rejected'


def test_annotation_schema_mutation_and_bio_wrong_label_cannot_commit(env,monkeypatch):
    store,add,versions,_,provider=env;v=add(image=True);u=versions[v]['units'][0]['id']
    ins.add_annotation(store,v,u,'label',{'label':'海豚'},False,'模型建议尚未确认')
    with pytest.raises(ins.InstructionError):ins.enqueue(store,v,'Bio',target=1)
    ins.add_annotation(store,v,u,'label',{'label':'海豚'},True,'人工确认')
    ins.enqueue(store,v,'Bio',target=1)
    def wrong(self,prompt,payload,units=(),**kw):
        result,stats=provider(self,prompt,payload,units,**kw)
        result['records'][0]['output']='鲨鱼'
        return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',wrong);run(store)
    assert not ins.listing(store)['items']


def test_real_corpus_frozen_version_worker_dispatch(tmp_path,monkeypatch):
    store=Store(tmp_path/'real-corpus')
    doc_id,_=store.import_file('sea.txt','温盐差异影响海水密度，密度差异可以驱动环流。'.encode(),'live',task_stage='corpus')
    corpus.enqueue(store,doc_id,kind='paper')
    assert Worker(store).run_once()
    version=corpus.listing(store)['items'][0]
    calls=[]
    def model(self,prompt,payload,units=(),**kw):
        calls.append(units)
        if prompt==ins.REVIEW:
            return {'checks':[dict(id=payload['records'][0]['id'],**{k:True for k in ins.CHECKS})]},{}
        return {'records':[dict(input='温盐差异如何影响环流？',output='温盐差异改变密度，密度差异驱动环流。',
                                language='zh',evidence_ids=[units[0]['id']],image_unit_id='',task_type='science')]},{}
    monkeypatch.setattr(ins.GLMClient,'call',model)
    queued=ins.enqueue(store,version['id'],'Science',target=1)
    assert Worker(store).run_once()
    value=first(store);assert value['status']=='approved' and len(calls)==2
    assert ins.detail(store,value['id'])['source_valid']
    ins.commit(store,value['id'],value['hash'])
    assert ins.enqueue(store,version['id'],'Science',target=1)['job_id']==queued['job_id']


def test_real_glm_codec_transport_true_image_and_stable_evidence_ids(env,monkeypatch):
    import base64
    import httpx
    store,add,versions,_,_=env;v=add(image=True);actual=versions[v]['units'][0]
    monkeypatch.setattr(ins.GLMClient,'call',REAL_GLM_CALL)
    monkeypatch.setattr('qaproject.glm.api_key',lambda:'offline-placeholder')
    config=dict(ins.settings(),base_url='https://glm.invalid',vision_model='offline-vision')
    monkeypatch.setattr(ins,'settings',lambda:config)
    real_client=httpx.Client;requests=[]
    def respond(request):
        body=json.loads(request.content); content=body['messages'][1]['content'];requests.append(body)
        payload=json.loads(content[0]['text'])
        image=next(c for c in content if c['type']=='image_url')
        assert base64.b64decode(image['image_url']['url'].split(',',1)[1])==PathBytes(actual['image'])
        assert payload['units'][0]['id']=='u1'
        if 'records' in payload:
            assert payload['records'][0]['evidence_ids']==['u1']
            assert payload['records'][0]['image_unit_id']=='u1'
            result=dict(checks=[dict(id=payload['records'][0]['id'],**{k:True for k in ins.CHECKS})])
        else:
            result=dict(records=[dict(input='图中能观察到什么？',output='可观察的形态。',language='en',
                                     evidence_ids=['u1'],image_unit_id='u1',task_type='description')])
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':encode(result)}}]})
    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:real_client(transport=httpx.MockTransport(respond)))
    ins.enqueue(store,v,'Sonar-Open',target=1);run(store)
    value=first(store)
    assert value['status']=='approved' and len(requests)==2
    assert value['data']['evidence_ids']==[actual['id']] and value['data']['image_unit_id']==actual['id']


def test_appended_annotations_and_unselected_nodes_preserve_committed_snapshot(env):
    store,add,versions,_,_=env;v=add(image=True);unit=versions[v]['units'][0]['id']
    label=ins.add_annotation(store,v,unit,'label','海豚',True,'人工确认')
    root=ins.add_node(store,'动物','真实定义',v)
    child=ins.add_node(store,'海豚','真实定义',v,root['id'])
    job=ins.enqueue(store,v,'Bio',target=1,node_ids=[child['id']]);run(store)
    value=first(store);ins.commit(store,value['id'],value['hash'])
    before=ins.detail(store,value['id'])
    ins.add_annotation(store,v,unit,'caption','后续补充图注',False,'新增来源')
    ins.add_node(store,'未选节点','其他定义',v)
    reopened=Store(store.root)
    after=ins.detail(reopened,value['id'])
    assert after['source_valid'] and after['hash']==before['hash']
    assert after['annotations']==before['annotations'] and after['annotations'][0]['id']==label['id']
    assert ins.export_bundle(reopened,'Bio')
    assert ins.commit(reopened,value['id'],value['hash'])['status']=='committed'
    assert ins._job(reopened,job['job_id'])['binding_hash']==ins._job(store,job['job_id'])['binding_hash']


@pytest.mark.parametrize('mutation',['delete','rehash'])
def test_frozen_annotation_members_still_required(env,mutation):
    store,add,versions,_,_=env;v=add(image=True);unit=versions[v]['units'][0]['id']
    annotation=ins.add_annotation(store,v,unit,'label','海豚',True,'人工确认')
    ins.enqueue(store,v,'Bio',target=1);run(store);value=first(store)
    ins.commit(store,value['id'],value['hash'])
    with store.connect() as db:
        if mutation=='delete':db.execute('DELETE FROM corpus_annotations WHERE id=?',(annotation['id'],))
        else:
            data={k:annotation[k] for k in ('version_id','unit_id','kind','value','verified','source_note')}
            data['value']='鲨鱼'
            db.execute('UPDATE corpus_annotations SET value_json=?,hash=? WHERE id=?',
                       (encode(data['value']),digest(data),annotation['id']))
    assert not ins.detail(store,value['id'])['source_valid']
    with pytest.raises(ins.InstructionError):ins.export_bundle(store,'Bio')


def test_reject_during_inflight_repair_preserves_content_and_history(env,monkeypatch):
    store,add,_,_,provider=env;v=add();ins.enqueue(store,v,'Science',target=1)
    rejected=[]
    def model(self,prompt,payload,units=(),**kw):
        if prompt==ins.REPAIR:
            value=first(store);rejected.append(value)
            ins.reject(store,value['id'],value['hash'])
        result,stats=provider(self,prompt,payload,units,**kw)
        if prompt==ins.REVIEW:result['checks'][0]['supported']=False
        return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',model);run(store)
    value=first(store)
    assert value['status']=='rejected' and value['hash']==rejected[0]['hash']
    assert value['data']==rejected[0]['data']
    assert len(ins.detail(store,value['id'])['review_history'])==1
    with store.connect() as db:
        assert db.execute('SELECT attempts FROM instructions_repair_state').fetchone()[0]==0
    with pytest.raises(ins.InstructionError):ins.commit(store,value['id'],value['hash'])


@pytest.mark.parametrize('boundary',['before_request','after_cache'])
def test_interrupted_repair_replays_review_once_and_preserves_attempt(env,monkeypatch,boundary):
    from qaproject.execution import WorkInterrupted
    store,add,_,calls,provider=env;v=add();job=ins.enqueue(store,v,'Science',target=1)
    def model(self,prompt,payload,units=(),**kw):
        result,stats=provider(self,prompt,payload,units,**kw)
        if prompt==ins.REVIEW and payload['records'][0]['output']=='温度影响密度。':
            result['checks'][0]['supported']=False
        if prompt==ins.REPAIR:result['records'][0]['output']='修订后的有效事实。'
        assert self.config['evidence_policy']=='registered_bundle'
        return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',model)
    remaining=[2 if boundary=='before_request' else 1]
    original_update=store.task_update;original_event=store.event
    def update(*args,**kwargs):
        if boundary=='before_request' and args[2]=='repairing' and remaining[0]:
            remaining[0]-=1;raise WorkInterrupted()
        return original_update(*args,**kwargs)
    def event(doc,kind,detail):
        if boundary=='after_cache' and kind=='instruction_model_call' and detail.get('stage')=='repair' and remaining[0]:
            remaining[0]-=1;raise WorkInterrupted()
        return original_event(doc,kind,detail)
    monkeypatch.setattr(store,'task_update',update);monkeypatch.setattr(store,'event',event)
    while remaining[0]:
        run(store)
        assert ins._job(store,job['job_id'])['status']=='queued'
        with store.connect() as db:
            assert db.execute('SELECT COUNT(*) FROM instructions_review_history').fetchone()[0]==1
            assert db.execute('SELECT attempts FROM instructions_repair_state').fetchone()[0]==0
    reopened=Store(store.root);run(reopened)
    assert first(reopened)['status']=='approved'
    with reopened.connect() as db:
        assert db.execute('SELECT attempts FROM instructions_repair_state').fetchone()[0]==1
        assert db.execute('SELECT COUNT(*) FROM instructions_review_history').fetchone()[0]==2
    assert len([c for c in calls if c[0]==ins.REPAIR])==1
    assert len([c for c in calls if c[0]==ins.REVIEW])==2


def test_unchanged_repairs_use_two_persistent_attempts_not_replayed_reviews(env,monkeypatch):
    store,add,_,calls,provider=env;v=add();job=ins.enqueue(store,v,'Science',target=1)
    repairs=[]
    def model(self,prompt,payload,units=(),**kw):
        if prompt==ins.REPAIR:
            repairs.append(payload['repair_attempt'])
            return {'records':[dict(payload['records'][0])]},{}
        result,stats=provider(self,prompt,payload,units,**kw)
        if prompt==ins.REVIEW:result['checks'][0]['supported']=False
        return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',model);run(store)
    assert repairs==[1,2]
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM instructions_review_history').fetchone()[0]==1
        assert db.execute('SELECT attempts FROM instructions_repair_state').fetchone()[0]==2
    ins.retry_job(store,job['job_id']);run(store)
    assert repairs==[1,2] and first(store)['status']=='failed'
    assert len([c for c in calls if c[0]==ins.REVIEW])==1


@pytest.mark.parametrize('review_passed',[True,False])
def test_partial_generation_is_insufficient_without_filling_or_dropping_approved(env,monkeypatch,review_passed):
    store,add,_,calls,provider=env;v=add();job=ins.enqueue(store,v,'Science',target=3)
    generated=[False]
    def model(self,prompt,payload,units=(),**kw):
        if prompt==ins.GENERATE and generated[0]:return {'records':[]},{}
        if prompt==ins.GENERATE:generated[0]=True
        result,stats=provider(self,prompt,payload,units,**kw)
        if prompt==ins.REVIEW:result['checks'][0]['supported']=review_passed
        return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',model);run(store)
    value=ins._job(store,job['job_id'])
    assert value['status']=='insufficient'
    assert '目标 3 条' in value['error'] and '实际生成 1 条' in value['error']
    assert f'审核通过 {int(review_passed)} 条' in value['error']
    assert first(store)['status']==('approved' if review_passed else 'failed')
    if review_passed:
        assert ins.commit(store,first(store)['id'],first(store)['hash'])['status']=='committed'
    else:
        with pytest.raises(ins.InstructionError):ins.commit(store,first(store)['id'],first(store)['hash'])
    with store.connect() as db:
        assert db.execute('SELECT stage FROM tasks WHERE id=?',(job['task_id'],)).fetchone()[0]=='instruction_insufficient'


def test_commit_rechecks_concurrent_rejection_and_current_review(env,monkeypatch):
    store,add,_,_,_=env;v=add();ins.enqueue(store,v,'Science',target=1);run(store)
    value=first(store);original=ins.detail
    def detail_then_reject(*args):
        result=original(*args)
        ins.reject(store,value['id'],value['hash'])
        return result
    monkeypatch.setattr(ins,'detail',detail_then_reject)
    with pytest.raises(ins.InstructionError,match='候选已变化'):ins.commit(store,value['id'],value['hash'])
    assert first(store)['status']=='rejected'


def test_provider_error_keeps_safe_detail_and_drops_private_stats(env,monkeypatch):
    store,add,_,_,_=env;v=add();job=ins.enqueue(store,v,'Science',target=1)
    def model(*args,**kwargs):
        raise ProviderError('公司接口返回 HTTP 400；API_KEY=PRIVATEKEY Authorization=Bearer PRIVATEAUTH', 'http',
            {'http_status':400,'finish_reason':'length','error_category':'context','seconds':1.25,
             'body':'PRIVATEBODY','headers':{'Authorization':'PRIVATEHEADER'},
             'usage':{'prompt_tokens':12,'echo':'PRIVATETOKENS'}})
    monkeypatch.setattr(ins.GLMClient,'call',model);run(store)
    value=ins._job(store,job['job_id'])
    assert 'HTTP 400' in value['error'] and 'context' in value['error']
    with store.connect() as db:
        event=json.loads(db.execute("SELECT detail FROM events WHERE kind='instruction_error'").fetchone()[0])
    assert event['provider_stats']=={'http_status':400,'finish_reason':'length','error_category':'context',
                                    'seconds':1.25,'usage':{'prompt_tokens':12}}
    assert 'PRIVATE' not in encode([value['error'],event])


@pytest.mark.parametrize('bad_id',['','u2','unknown'])
@pytest.mark.parametrize('corrected',[True,False])
def test_visual_wrong_image_id_one_correction_real_codec(env,monkeypatch,bad_id,corrected):
    import base64
    import httpx
    store,add,versions,_,_=env;v=add(image=True)
    image=versions[v]['units'][0];image['page']=1
    text=dict(image,id=image['id']+'-text',image='',text='同页的真实文字说明。')
    versions[v]['units'].append(text)
    monkeypatch.setattr(ins.GLMClient,'call',REAL_GLM_CALL)
    monkeypatch.setattr('qaproject.glm.api_key',lambda:'offline-placeholder')
    config=dict(ins.settings(),base_url='https://glm.invalid',vision_model='offline')
    monkeypatch.setattr(ins,'settings',lambda:config)
    real_client=httpx.Client;requests=[];generations=[]
    def respond(request):
        body=json.loads(request.content);content=body['messages'][1]['content']
        payload=json.loads(content[0]['text']);requests.append(payload)
        attachments=[c for c in content if c['type']=='image_url']
        assert len(attachments)==1
        assert base64.b64decode(attachments[0]['image_url']['url'].split(',',1)[1])==PathBytes(image['image'])
        if 'validation_correction' in payload:
            assert '唯一一次纠正' in payload['validation_correction']['instruction']
        if 'records' in payload:
            assert payload['records'][0]['image_unit_id']=='u1'
            result=dict(checks=[dict(id=payload['records'][0]['id'],**{k:True for k in ins.CHECKS})])
        else:
            generations.append(payload)
            assert payload['image_unit_ids']==['u1']
            assert [u['id'] for u in payload['units']]==['u1','u2']
            chosen='u1' if corrected and len(generations)==2 else bad_id
            result=dict(records=[dict(input='图中可见什么形态？',output='可见的蓝色形态。',language='zh',
                evidence_ids=['u1','u2'],image_unit_id=chosen,task_type='description')])
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':encode(result)}}]})
    monkeypatch.setattr('qaproject.glm.httpx.Client',lambda **kw:real_client(transport=httpx.MockTransport(respond)))
    ins.enqueue(store,v,'MultiModel-Science',target=1);run(store)
    assert len(generations)==2 and 'validation_correction' not in generations[0]
    if corrected:
        value=first(store)
        assert value['status']=='approved' and value['data']['image_unit_id']==image['id']
        assert value['data']['evidence_ids']==[image['id'],text['id']]
        assert len(requests)==3
    else:
        assert ins.jobs(store)[0]['status']=='failed' and not ins.listing(store)['items']
        assert len(requests)==2


@pytest.mark.parametrize('failure',['shape','reference','json','network','timeout','http'])
def test_generation_correction_only_for_known_validation_failures(env,monkeypatch,failure):
    store,add,_,_,provider=env;v=add();ins.enqueue(store,v,'Science',target=1)
    attempts=[]
    def model(self,prompt,payload,units=(),**kw):
        if prompt==ins.GENERATE:
            attempts.append(payload)
            if len(attempts)==1:
                if failure in ('json','network','timeout','http'):raise ProviderError('模拟失败',failure,{})
                if failure=='shape':return {'records':{}},{}
                return {'records':[dict(input='问题',output='答案',language='zh',evidence_ids=['unknown'])]},{}
        return provider(self,prompt,payload,units,**kw)
    monkeypatch.setattr(ins.GLMClient,'call',model);run(store)
    recover=failure in ('shape','reference','json')
    assert len(attempts)==(2 if recover else 1)
    assert ins.jobs(store)[0]['status']==('completed' if recover else 'failed')


def test_repair_reference_correction_preserves_review_repair_budget(env,monkeypatch):
    store,add,_,_,provider=env;v=add(image=True);ins.enqueue(store,v,'MultiModel-Science',target=1)
    repairs=[];reviews=[]
    def model(self,prompt,payload,units=(),**kw):
        result,stats=provider(self,prompt,payload,units,**kw)
        if prompt==ins.REVIEW:
            reviews.append(payload);result['checks'][0]['supported']=len(reviews)>1
        if prompt==ins.REPAIR:
            repairs.append(payload)
            if len(repairs)==1:result['records'][0]['image_unit_id']=''
        return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',model);run(store)
    assert first(store)['status']=='approved'
    assert len(repairs)==2 and [r['repair_attempt'] for r in repairs]==[1,1]
    assert 'validation_correction' in repairs[1] and len(reviews)==2
    with store.connect() as db:
        assert db.execute('SELECT attempts FROM instructions_repair_state').fetchone()[0]==1


def test_corrected_success_cache_survives_interruption_without_repeating_model(env,monkeypatch):
    from qaproject.execution import WorkInterrupted
    store,add,_,_,provider=env;v=add(image=True)
    job=ins.enqueue(store,v,'MultiModel-Science',target=1);generations=[]
    def model(self,prompt,payload,units=(),**kw):
        result,stats=provider(self,prompt,payload,units,**kw)
        if prompt==ins.GENERATE:
            generations.append(payload)
            if len(generations)==1:result['records'][0]['image_unit_id']=''
        return result,stats
    original_event=store.event;interrupted=[]
    def event(doc,kind,detail):
        if kind=='instruction_model_call' and detail['stage']=='generate' and not interrupted:
            interrupted.append(True);raise WorkInterrupted()
        return original_event(doc,kind,detail)
    monkeypatch.setattr(ins.GLMClient,'call',model);monkeypatch.setattr(store,'event',event)
    run(store)
    assert ins._job(store,job['job_id'])['status']=='queued'
    run(store)
    assert len(generations)==2 and first(store)['status']=='approved'


@pytest.mark.parametrize('image',[False,True])
def test_exact_cross_job_duplicate_commit_is_atomic_and_idempotent(env,monkeypatch,image):
    store,add,_,_,provider=env;v=add(image=image)
    subset='MultiModel-Science' if image else 'Science'
    def model(self,prompt,payload,units=(),**kw):
        result,stats=provider(self,prompt,payload,units,**kw)
        if prompt==ins.GENERATE:
            result['records'][0]['input']='  SAME   question ' if payload['existing_records']==[] else 'unused'
        return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',model)
    ins.enqueue(store,v,subset,target=1,source_label='first');run(store);a=first(store)
    ins.enqueue(store,v,subset,target=1,source_label='second');run(store);b=first(store)
    edited=ins.edit(store,b['id'],{'input':'same question'},b['hash'])
    ins.review_enqueue(store,b['id'],edited['hash']);run(store);b=ins.detail(store,b['id'])
    barrier=threading.Barrier(2);original_detail=ins.detail
    def synchronized(*args):
        value=original_detail(*args);barrier.wait(timeout=10);return value
    monkeypatch.setattr(ins,'detail',synchronized)
    outcomes=[]
    def submit(value):
        try:outcomes.append(ins.commit(store,value['id'],value['hash'])['status'])
        except ins.InstructionError as error:outcomes.append(error.code)
    threads=[threading.Thread(target=submit,args=(value,)) for value in (a,b)]
    for t in threads:t.start()
    for t in threads:t.join(timeout=15)
    assert sorted(outcomes)==['committed','duplicate']
    monkeypatch.setattr(ins,'detail',original_detail)
    rows=ins.listing(store)['items'];committed=next(r for r in rows if r['status']=='committed')
    assert ins.commit(store,committed['id'],committed['hash'])['status']=='committed'
    assert sum(r['status']=='approved' for r in rows)==1


def test_same_prompt_different_image_content_can_commit(env,monkeypatch):
    store,add,versions,_,provider=env;first_v=add(image=True);second_v=add('different',image=True,color='red')
    def model(self,prompt,payload,units=(),**kw):
        result,stats=provider(self,prompt,payload,units,**kw)
        if prompt==ins.GENERATE:result['records'][0]['input']='图中可见什么？'
        return result,stats
    monkeypatch.setattr(ins.GLMClient,'call',model)
    for v in (first_v,second_v):
        ins.enqueue(store,v,'MultiModel-Science',target=1);run(store);value=first(store)
        assert ins.commit(store,value['id'],value['hash'])['status']=='committed'
    assert len(ins.listing(store,status='committed')['items'])==2
