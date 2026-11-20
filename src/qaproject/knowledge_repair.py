"""Explicit per-rejection repairs on the existing leased knowledge queue."""
import json
import time

from .execution import WorkInterrupted
from .glm import GLMClient, ProviderError
from .persistence import file_hash
from .source_files import original_path, verify_reference
from .store import digest, encode, uid
from .settings import settings


def rejections(store, document_id):
    result = []
    with store.connect() as db:
        batches = db.execute('SELECT result FROM knowledge_batches WHERE document_id=?', (document_id,)).fetchall()
        repairs = [dict(r) for r in db.execute('SELECT * FROM knowledge_repairs WHERE document_id=? ORDER BY created,id', (document_id,))]
    for batch in batches:
        for item in json.loads(batch[0]).get('rejected', []):
            row = item.get('row') or item.get('candidate') or {}
            rejection_id = digest(item)
            history = [r for r in repairs if r['rejection_id'] == rejection_id]
            review = item.get('review', {})
            # Unsupported evidence is not repaired by inventing a new fact.
            repairable = review.get('supported') is not False and not review.get('local_validation')
            result.append(dict(id=rejection_id, expected_hash=digest(item),
                entity_name=row.get('entity_name',''), description=row.get('entity_description',''),
                reason=review.get('note','审核未通过'), repairable=repairable,
                status=history[-1]['status'] if history else 'rejected', attempts=len(history),
                history=[dict(status=r['status'],error=r['error'],review=json.loads(r['review']) if r['review'] else {}) for r in history],
                snapshot=item))
    return result


def enqueue_repair(store, document_id, rejection_id, expected_hash):
    item = next((r for r in rejections(store,document_id) if r['id']==rejection_id), None)
    if not item or item['expected_hash'] != expected_hash:
        raise ValueError('失败条目已变更，请刷新后重试。')
    if not item['repairable']:
        raise ValueError('该条目证据不足或引用无效，不能自动修补事实。')
    doc=store.document(document_id);verify_reference(doc,store.root)
    if file_hash(original_path(doc,store.root)) != doc['content_hash']:
        raise ValueError('原始文件已变更，不能修订旧条目。')
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        history=db.execute('SELECT * FROM knowledge_repairs WHERE document_id=? AND rejection_id=? ORDER BY created,id',
                           (document_id,rejection_id)).fetchall()
        if history and history[-1]['status'] in ('queued','running','approved'):
            return dict(task_id=history[-1]['task_id'],status=history[-1]['status'])
        if history and history[-1]['status']=='retryable':
            task_id=history[-1]['task_id']
            db.execute("UPDATE tasks SET status='queued',stage='knowledge',worker=NULL,lease_until=0 WHERE id=? AND status='failed'",(task_id,))
            db.execute("UPDATE knowledge_repairs SET status='queued' WHERE id=?",(history[-1]['id'],))
            return dict(task_id=task_id,status='queued')
        if len(history)>=2:raise ValueError('该条目已尝试两次修复，保留失败历史，不再自动返工。')
        task_id=uid()
        db.execute("INSERT INTO tasks(id,document_id,status,stage,initial_stage) VALUES(?,?,'queued','knowledge','knowledge')",(task_id,document_id))
        db.execute('INSERT INTO knowledge_repairs(id,rejection_id,document_id,task_id,snapshot,snapshot_hash,status,created) VALUES(?,?,?,?,?,?,?,?)',
                   (uid(),rejection_id,document_id,task_id,encode(item['snapshot']),expected_hash,'queued',time.time()))
    return dict(task_id=task_id,status='queued')


def process_repair(worker, task, repair):
    from .knowledge import CHECKS, EXTRACT, REVIEW, validate_entry, lease, passages
    from .knowledge_context import explicit_aliases, canonical_name, entity_context
    store=worker.store; doc=store.document(task['document_id']); started=time.perf_counter()
    stage='knowledge_repair'
    def checkpoint():
        worker.limits.check(worker.stop)
        store.task_update(task['id'],worker.name,stage)
    def verify():
        verify_reference(doc,store.root)
        if file_hash(original_path(doc,store.root)) != doc['content_hash']:
            raise ValueError('原件已变更，不能修复旧证据。')
    def save(**fields):
        with store.connect() as db:
            lease(db,task,worker.name)
            db.execute('UPDATE knowledge_repairs SET '+','.join(k+'=?' for k in fields)+' WHERE id=?',
                       [*fields.values(),repair['id']])
    try:
        checkpoint();verify()
        original=json.loads(repair['snapshot'])
        if digest(original)!=repair['snapshot_hash']:raise ValueError('失败记录绑定失效')
        with store.connect() as db:
            saved=db.execute('SELECT units FROM knowledge_jobs WHERE document_id=?',(doc['id'],)).fetchone()
            existing=[json.loads(r[0]) for r in db.execute("SELECT row_json FROM knowledge_entries WHERE document_id=? AND status='approved' ORDER BY created,id",(doc['id'],))]
        units=json.loads(saved[0]) if saved else []
        unit=next((u for u in units if u['id']==original['unit_id']),None)
        if not unit or unit['hash']!=original['evidence_hash'] or digest({k:v for k,v in unit.items() if k!='hash'})!=unit['hash']:
            raise ValueError('修复证据绑定失效')
        aliases=explicit_aliases([u for u in units if not u.get('image')])
        save(status='running',error='')
        config=settings()
        if repair['generated']:
            generated=json.loads(repair['generated'])
            if generated['hash']!=digest(generated['item']):raise ValueError('修复缓存绑定失效')
            item=generated['item']
        else:
            context,index=entity_context(existing,[unit],[(original.get('row') or {}).get('entity_name','')],aliases)
            payload=dict(units=[dict(id=unit['id'],passages=passages(unit))],existing_entities=context,
                         existing_entity_index=index,document_topics=[t['name'] for t in doc['topics']],
                         failed_entry=original,repair_instruction='仅修订这一个失败条目，不新增其他实体；保留原实体身份及原有事实范围。证据不足返回空entries。')
            with worker.limits.request(worker.stop,checkpoint):
                result,stats=GLMClient(config).call(EXTRACT,payload,[unit])
            store.event(doc['id'],'knowledge_repair_model_call',dict(stats,stage='repair'))
            if not isinstance(result,dict) or not isinstance(result.get('entries'),list) or len(result['entries'])!=1:
                raise ValueError('未返回唯一可修复条目，保留原失败记录。')
            item=result['entries'][0]
            if not isinstance(item,dict) or item.get('unit_id')!=unit['id']:raise ValueError('修复引用了其他证据')
            name=item.get('entity_name','')
            item=dict(item,entity_name=canonical_name(name,aliases,existing,item.get('entity_type')))
            prior=original['row']
            expected=canonical_name(prior['entity_name'],aliases,existing,prior['entity_type'])
            if item['entity_name'].lower()!=expected.lower():raise ValueError('修复不能替换实体身份')
            validate_entry(item,unit,doc,[t['name'] for t in doc['topics']])
            save(generated=encode(dict(item=item,hash=digest(item))))
        row,topic,quote=validate_entry(item,unit,doc,[t['name'] for t in doc['topics']])
        entry=dict(id=digest([doc['id'],unit['id'],row]),unit_id=unit['id'],row=row,topic=topic,quote=quote,evidence_hash=unit['hash'])
        if repair['review']:
            receipt=json.loads(repair['review'])
            if receipt.get('row_hash')!=digest(row) or receipt.get('evidence_hash')!=unit['hash']:
                raise ValueError('修复审核绑定失效')
            check=receipt['check']
            if receipt.get('check_hash')!=digest(check) or any(type(check.get(k)) is not bool for k in CHECKS):
                raise ValueError('修复审核缓存失效')
        else:
            related,_=entity_context(existing,[unit],[row['entity_name']],aliases)
            payload=dict(units=[dict(id=unit['id'],text=unit['text'])],entries=[entry],existing_entities=related,
                         original_failed_entry=original,repair_review='除常规审核外，必须确认未扩大原条目的事实范围、未替换实体。不符合则scope_correct=false。')
            with worker.limits.request(worker.stop,checkpoint):
                result,stats=GLMClient(config).call(REVIEW,payload,[unit])
            store.event(doc['id'],'knowledge_repair_model_call',dict(stats,stage='repair_review'))
            checks=result.get('checks') if isinstance(result,dict) else None
            if not isinstance(checks,list) or len(checks)!=1 or not isinstance(checks[0],dict):raise ValueError('修复审核结构无效')
            check=checks[0]
            if check.get('id')!=entry['id'] or any(type(check.get(k)) is not bool for k in CHECKS):raise ValueError('修复审核缺项')
            receipt=dict(check=check,check_hash=digest(check),row_hash=digest(row),evidence_hash=unit['hash'])
            save(review=encode(receipt))
        checkpoint();verify()
        passed=all(check.get(k) is True for k in CHECKS)
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE');lease(db,task,worker.name)
            if passed:
                binding=dict(check,row_hash=digest(row),evidence_hash=unit['hash'])
                db.execute('INSERT OR IGNORE INTO knowledge_entries VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                           (entry['id'],doc['id'],unit['id'],unit['hash'],encode(row),digest(row),topic,quote,'approved',encode(binding),time.time()))
            db.execute('UPDATE knowledge_repairs SET status=?,error=? WHERE id=?',
                       ('approved' if passed else 'failed','' if passed else '修复后审核未通过，原失败记录保留。',repair['id']))
            db.execute("UPDATE tasks SET status=?,stage='knowledge_repair_complete',lease_until=0 WHERE id=?",
                       ('succeeded' if passed else 'failed',task['id']))
        store.event(doc['id'],'knowledge_repair_complete',dict(stage=stage,seconds=round(time.perf_counter()-started,3),passed=passed))
    except Exception as error:
        interrupted=isinstance(error,WorkInterrupted)
        message=str(error) if type(error) is ValueError else '修复中断或接口失败，已保存完成步骤，可重试。'
        with store.connect() as db:
            lease(db,task,worker.name)
            db.execute('UPDATE knowledge_repairs SET status=?,error=? WHERE id=?',('queued' if interrupted else 'retryable' if isinstance(error,ProviderError) else 'failed',message,repair['id']))
            db.execute("UPDATE tasks SET status=?,stage='knowledge',lease_until=0 WHERE id=?",('queued' if interrupted else 'failed',task['id']))
        store.event(doc['id'],'knowledge_repair_failed',dict(stage=stage,error_type=type(error).__name__,error_code=getattr(error,'code','internal'),
            error_category=getattr(error,'stats',{}).get('error_category',''),
            finish_reason=getattr(error,'stats',{}).get('finish_reason','incomplete'),seconds=round(time.perf_counter()-started,3)))
