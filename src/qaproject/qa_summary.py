"""Browse committed records and review edits before atomically replacing them."""
import json
import time

from .store import digest, encode, normalized, uid
from .execution import WorkInterrupted
from .conflicts import EditConflict
from .source_files import verify_reference


def members(db, qa_id):
    target = db.execute('SELECT c.*,p.pair_id FROM candidates c JOIN qas q ON q.id=c.id '
                        'LEFT JOIN candidate_pairs p ON p.candidate_id=c.id WHERE c.id=?', (qa_id,)).fetchone()
    if not target:
        raise ValueError('找不到已入库 QA。')
    rows = db.execute('''SELECT c.*,p.pair_id FROM candidates c JOIN qas q ON q.id=c.id
        LEFT JOIN candidate_pairs p ON p.candidate_id=c.id
        WHERE p.pair_id=? ORDER BY p.language DESC''', (target['pair_id'],)).fetchall() if target['pair_id'] else [target]
    if target['pair_id'] and (len(rows) != 2 or {json.loads(r['row_json'])['language'] for r in rows} != {'zh','en'}):
        raise ValueError('本组尚未完整入库，请先在文件处理中完成入库。')
    return [dict(r) for r in rows]


def latest_revision(db, qa_id):
    row = db.execute('''SELECT r.id,r.status,r.note,r.created,r.updated FROM qa_revisions r
        JOIN json_each(r.before_json) v ON json_extract(v.value,'$.id')=?
        WHERE r.document_id=(SELECT document_id FROM candidates WHERE id=?)
        ORDER BY r.created DESC,r.id DESC LIMIT 1''', (qa_id,qa_id)).fetchone()
    return dict(row) if row else None


def list_qas(store, page=1, page_size=20):
    with store.connect() as db:
        db.execute('BEGIN')
        total = db.execute('SELECT COUNT(*) FROM qas').fetchone()[0]
        pages = max(1, (total+page_size-1)//page_size)
        page = min(page, pages)
        rows = db.execute('''SELECT q.id,q.row_json,q.created,c.document_id,d.name AS document_name
            FROM qas q JOIN candidates c ON c.id=q.id JOIN documents d ON d.id=c.document_id
            ORDER BY q.created DESC,q.id DESC LIMIT ? OFFSET ?''', (page_size,(page-1)*page_size)).fetchall()
        items = [dict(id=r['id'], row=json.loads(r['row_json']), created=r['created'],
                      document_id=r['document_id'], document_name=r['document_name'],
                      revision=latest_revision(db,r['id'])) for r in rows]
    return dict(items=items,total=total,page=page,page_size=page_size,pages=pages)


def edit_detail(store, qa_id):
    with store.connect() as db:
        db.execute('BEGIN')
        rows = members(db,qa_id)
        document = db.execute('SELECT units,name FROM documents WHERE id=?',(rows[0]['document_id'],)).fetchone()
        unit = next((u for u in json.loads(document['units']) if u['id']==rows[0]['unit_id']),None)
        if not unit:
            raise ValueError('原始证据不可用。')
        revision = latest_revision(db,qa_id)
        draft = db.execute('SELECT after_json FROM qa_revisions WHERE id=?',(revision['id'],)).fetchone() if revision else None
        return dict(draft_versions=json.loads(draft[0]) if draft and revision['status']!='applied' else [], versions=[dict(id=r['id'], row=json.loads(r['row_json']), row_hash=r['row_hash']) for r in rows],
                    document_name=document['name'], document_id=rows[0]['document_id'],
                    evidence=unit, revision=revision,edit_version=edit_version(rows,revision))


def edit_version(rows, revision):
    return digest([sorted((r['id'],r['row_hash']) for r in rows),
                   [revision['id'],revision['status']] if revision else None])


def queue_revision(store, qa_id, versions, expected_edit_version=None):
    from .pipeline import validate_row
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        rows = members(db,qa_id)
        if expected_edit_version is not None and expected_edit_version!=edit_version(rows,latest_revision(db,qa_id)):
            raise EditConflict('其他操作已更新本组问答或修改草稿。请刷新最新版本后再保存；本次输入未覆盖任何记录。')
        for r in rows:
            formal = db.execute('SELECT row_json FROM qas WHERE id=?',(r['id'],)).fetchone()
            if r['status']!='approved' or formal[0]!=r['row_json']:
                raise ValueError('正式内容与候选版本不一致，请先检查数据。')
        doc_id = rows[0]['document_id']
        if db.execute("SELECT 1 FROM tasks WHERE document_id=? AND status IN ('queued','running')",(doc_id,)).fetchone():
            raise ValueError('该文件正在处理或审核，请稍后再保存。')
        if len(versions)!=len(rows) or {v['id'] for v in versions}!={r['id'] for r in rows}:
            raise ValueError('请同时提交本组全部已入库版本。')
        doc = db.execute('SELECT * FROM documents WHERE id=?',(doc_id,)).fetchone()
        unit = next(u for u in json.loads(doc['units']) if u['id']==rows[0]['unit_id'])
        changed = []
        for r in rows:
            v = next(v for v in versions if v['id']==r['id'])
            if v['expected_hash']!=r['row_hash']:
                raise EditConflict('内容已被修改，请刷新最新版本后重新编辑；本次输入未覆盖任何记录。')
            row = json.loads(r['row_json'])
            row.update({k:v[k].strip() for k in ('input','thinking','output')})
            topic = dict(db.execute('SELECT * FROM topics WHERE id=?',(r['topic_id'],)).fetchone())
            validate_row(row,unit,topic,doc['path'])
            changed.append(dict(id=r['id'],row=row))
        if all(digest(v['row'])==r['row_hash'] for v,r in zip(changed,rows)):
            return {'unchanged':True}
        revision_id, task_id, now = uid(),uid(),time.time()
        db.execute('INSERT INTO qa_revisions VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (revision_id,doc_id,encode(rows),encode(changed),'queued','等待核验修改内容','{}',now,now,task_id))
        db.execute('INSERT INTO tasks(id,document_id,status,stage,revision_id) VALUES(?,?,?,?,?)',
                   (task_id,doc_id,'queued','qa_revision',revision_id))
        db.execute('INSERT INTO events VALUES(?,?,?,?,?)',
                   (uid(),doc_id,'qa_revision_requested',encode({'revision_id':revision_id,'qa_ids':[r['id'] for r in rows]}),now))
    return {'revision_id':revision_id,'status':'queued'}


def process_revision(store, task, worker, call):
    from .pipeline import (assert_lease, approve, review_groups, review_binding, compact_units,
                           CHECKS, BILINGUAL_CHECKS, REVIEW, quality_flags, review_note_flags, validate_row)
    started = time.perf_counter()
    revision_id = task['revision_id']
    check = {}
    try:
        with store.connect() as db:
            assert_lease(db,task['id'],worker)
            revision = dict(db.execute('SELECT * FROM qa_revisions WHERE id=?',(revision_id,)).fetchone())
            if revision['status']=='applied':
                db.execute("UPDATE tasks SET status='succeeded',stage='qa_revision' WHERE id=?",(task['id'],))
                return
            db.execute("UPDATE qa_revisions SET status='reviewing',updated=? WHERE id=?",(time.time(),revision_id))
        before = json.loads(revision['before_json'])
        changes = json.loads(revision['after_json'])
        doc = store.document(task['document_id'])
        verify_reference(doc,store.root)
        unit = next(u for u in doc['units'] if u['id']==before[0]['unit_id'])
        candidates = [dict(r,row=next(v['row'] for v in changes if v['id']==r['id'])) for r in before]
        for c in candidates:
            topic = next(t for t in [*doc['topics'],*doc['historical_topics']] if t['id']==c['topic_id'])
            validate_row(c['row'],unit,topic,doc['path'])
        group = review_groups(candidates)[0]
        selected = {r['id'] for r in before}
        result = call('review',REVIEW,{'candidates':[group],'units':compact_units([unit]),
                      'compare_questions':[c['row']['input'] for c in doc['candidates']
                                           if c['id'] not in selected and c['status'] not in ('rejected','discarded')]},[unit])
        checks = result.get('checks',[])
        if not isinstance(checks,list) or len(checks)!=1 or not isinstance(checks[0],dict) or checks[0].get('candidate_id')!=group['candidate_id']:
            raise ValueError('审核响应不完整，请重试本次修改。')
        check = {k:checks[0].get(k) for k in CHECKS+BILINGUAL_CHECKS+['note']}
        check['binding'] = review_binding(group,unit)
        check['local_flags'] = list(dict.fromkeys([f for c in candidates for f in quality_flags(c['row'])]+review_note_flags(check.get('note',''))))
        passed = (all(check.get(k) is True for k in CHECKS+BILINGUAL_CHECKS)
                  and isinstance(check['note'],str) and bool(check['note'].strip()) and not check['local_flags'])
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            assert_lease(db,task['id'],worker)
            if passed:
                for old,c in zip(before,candidates):
                    current = db.execute('SELECT * FROM candidates WHERE id=?',(old['id'],)).fetchone()
                    formal = db.execute('SELECT row_json FROM qas WHERE id=?',(old['id'],)).fetchone()
                    if not current or dict(current)!={k:v for k,v in old.items() if k!='pair_id'} or not formal or formal[0]!=old['row_json']:
                        raise ValueError('正式内容或审核版本已变化，请刷新后重新编辑。')
                    db.execute("UPDATE candidates SET row_json=?,row_hash=?,fingerprint=?,review=?,status='pending' WHERE id=?",
                               (encode(c['row']),digest(c['row']),normalized(c['row']['input']),encode(check),c['id']))
                # Validate the entire edited group before any replacement; all writes roll back together.
                for c in candidates:
                    approve(store,c['id'],True,digest(c['row']),dry_run=True,_connection=db)
                for c in candidates:
                    approve(store,c['id'],True,digest(c['row']),_connection=db)
                    db.execute('UPDATE qas SET row_json=? WHERE id=?',(encode(c['row']),c['id']))
            status = 'applied' if passed else 'rejected'
            note = '修改已通过审核并保存。' if passed else '；'.join([str(check.get('note') or '修改未通过审核。'),*check['local_flags']])
            db.execute('UPDATE qa_revisions SET status=?,note=?,review=?,updated=? WHERE id=?',
                       (status,note,encode(check),time.time(),revision_id))
            db.execute("UPDATE tasks SET status='succeeded',stage='qa_revision' WHERE id=?",(task['id'],))
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(uid(),doc['id'],'qa_revision_'+status,
                       encode({'revision_id':revision_id,'seconds':round(time.perf_counter()-started,3)}),time.time()))
    except WorkInterrupted:
        # Let the shared executor requeue this exact revision on shutdown.
        raise
    except Exception as error:
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                assert_lease(db,task['id'],worker)
            except ValueError:
                return
            # Provider errors are already sanitized by the company adapter.
            note = str(error) if isinstance(error,ValueError) else '修改审核失败，正式内容已保留，请重试。'
            db.execute("UPDATE qa_revisions SET status='failed',note=?,review=?,updated=? WHERE id=?",(note,encode(check),time.time(),revision_id))
            db.execute("UPDATE tasks SET status='failed',stage='qa_revision' WHERE id=?",(task['id'],))
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(uid(),task['document_id'],'qa_revision_failed',
                       encode({'revision_id':revision_id,'stage':'review_or_commit','error_type':type(error).__name__,
                               'seconds':round(time.perf_counter()-started,3),'note':note}),time.time()))


def retry_revision(store, qa_id):
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        rows = members(db,qa_id)
        rev = latest_revision(db,qa_id)
        if not rev or rev['status']!='failed':
            raise ValueError('仅审核异常的修改可以重试。')
        if db.execute("SELECT 1 FROM tasks WHERE document_id=? AND status IN ('queued','running')",(rows[0]['document_id'],)).fetchone():
            raise ValueError('该文件正在处理，请稍后重试。')
        task_id = uid()
        db.execute('INSERT INTO tasks(id,document_id,status,stage,revision_id) VALUES(?,?,?,?,?)',
                   (task_id,rows[0]['document_id'],'queued','qa_revision',rev['id']))
        db.execute("UPDATE qa_revisions SET status='queued',note='等待重新审核',task_id=?,updated=? WHERE id=?",
                   (task_id,time.time(),rev['id']))
    return {'queued':True}
