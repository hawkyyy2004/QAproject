import json
from pathlib import Path
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .glm import GLMClient, ProviderError, ANALYZE, GENERATE, REVIEW
from .parsers import extract, text_alternatives
from .settings import FIELDS, settings, source_ref
from .store import uid, digest, encode, normalized

CHECKS = ['supported', 'independent', 'scope_correct', 'knowledge_value', 'distinct', 'topic_correct']


def quality_flags(row):
    flags=[]
    question=row['input']
    if re.search(r'这些|该研究|该报告|本文|文中|上述|^根据图\s*\d',question):
        flags.append('问题可能依赖未定义指代，需要写明对象和范围。')
    if re.search(r'(纵轴|横轴|坐标轴).*(刻度范围|标注为|单位是什么)',question):
        flags.append('单纯读取图轴/刻度可能缺乏领域知识价值。')
    return flags


def review_note_flags(note):
    if re.search(r'超出|未在本|未在该|未提供|未给出|无法核实|不支持|不足以支持', str(note)):
        return ['审核说明提到证据不足或超出原文，需人工检查，不能默认通过。']
    return []


def finish_review(db, document_id):
    pending=db.execute("SELECT COUNT(*) FROM candidates WHERE document_id=? AND status NOT IN ('approved','rejected')",(document_id,)).fetchone()[0]
    if not pending:
        count=db.execute('SELECT COUNT(*) FROM qas q JOIN candidates c ON c.id=q.id WHERE c.document_id=?',(document_id,)).fetchone()[0]
        minimum=settings()['min_per_document']
        db.execute('UPDATE documents SET status=?,error=? WHERE id=?',
                   ('completed' if count>=minimum else 'completed_under_min',
                    '' if count>=minimum else f'实际入库 {count} 条，低于最低 {minimum} 条；质量优先，不补造。',document_id))


def reject(store, candidate_id, expected_hash):
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM candidates WHERE id=?',(candidate_id,)).fetchone()
        if not row or row['row_hash']!=expected_hash:
            raise ValueError('候选版本已变更，请刷新。')
        if row['status']=='approved':
            raise ValueError('已入库记录不能通过候选排除操作撤回。')
        db.execute("UPDATE candidates SET status='rejected' WHERE id=?",(candidate_id,))
        db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(uid(),row['document_id'],'human_rejected',encode({'candidate_id':candidate_id,'row_hash':expected_hash}),time.time()))
        finish_review(db,row['document_id'])
    return {'rejected':True}


def validate_row(row, unit, topic, path):
    if set(row) != set(FIELDS) or any(not isinstance(x, str) for x in row.values()):
        raise ValueError('候选必须严格六个字符串字段。')
    if any(not row[x].strip() for x in FIELDS if x != 'image'):
        raise ValueError('必填字段为空。')
    if row['source'] != source_ref(path, unit['locator'], topic['name']):
        raise ValueError('来源、定位或主题不匹配。')
    if row['image'] != unit['image'] or (row['image'] and not Path(row['image']).is_file()):
        raise ValueError('图片引用不匹配或未保存。')
    if row['language'] not in {'zh', 'en'}:
        raise ValueError('语言无效。')
    if unit['text'] and not unit['image'] and row['language'] != unit['language']:
        raise ValueError('语言与文字证据不匹配。')
    if re.search(r'这[篇份个].*(文档|文件|报告).*(讲|介绍|多少)|PDF.*几页|CSV.*多少行|第\s*\d+\s*行.*数值', row['input'], re.I):
        raise ValueError('问题以文件或无意义索引为考点。')


def approve(store, candidate_id, reviewed, expected_hash):
    if not reviewed:
        raise ValueError('请先对照证据确认质量，再提交。')
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        c = db.execute('SELECT * FROM candidates WHERE id=?', (candidate_id,)).fetchone()
        if not c or c['row_hash'] != expected_hash:
            raise ValueError('候选版本已变更，请刷新后核对。')
        if c['status']=='rejected':
            raise ValueError('该候选已排除，初版暂不支持恢复。')
        d = db.execute('SELECT * FROM documents WHERE id=?', (c['document_id'],)).fetchone()
        unit = next((u for u in json.loads(d['units']) if u['id'] == c['unit_id']), None)
        topic = db.execute('SELECT * FROM topics WHERE id=?', (c['topic_id'],)).fetchone()
        if not unit or unit['hash'] != c['evidence_hash'] or digest(json.loads(c['row_json'])) != c['row_hash']:
            raise ValueError('证据或候选 hash 不匹配。')
        row = json.loads(c['row_json'])
        validate_row(row, unit, dict(topic), d['path'])
        import hashlib
        if hashlib.sha256(Path(d['path']).read_bytes()).hexdigest() != d['content_hash']:
            raise ValueError('原始文档副本已变更。')
        evidence_now = dict(unit)
        evidence_now.pop('hash')
        if digest(evidence_now) != unit['hash']:
            raise ValueError('证据缓存已变更。')
        if row['image']:
            import hashlib
            asset = db.execute('SELECT * FROM assets WHERE id=?', (unit['asset_id'],)).fetchone()
            if not asset or hashlib.sha256(Path(row['image']).read_bytes()).hexdigest() != asset['hash']:
                raise ValueError('图片内容已变更。')
        if not db.execute('SELECT 1 FROM document_topics WHERE document_id=? AND topic_id=?', (d['id'], topic['id'])).fetchone():
            raise ValueError('主题关联已失效。')
        if c['status']=='approved':
            return {'approved':True,'already_committed':True}
        db.execute('INSERT OR IGNORE INTO qas VALUES(?,?,?)', (candidate_id, c['row_json'], time.time()))
        db.execute("UPDATE candidates SET status='approved' WHERE id=?", (candidate_id,))
        finish_review(db,d['id'])
        db.execute('INSERT INTO events VALUES(?,?,?,?,?)', (uid(), d['id'], 'human_review_commit', encode({'candidate_id': candidate_id, 'row_hash': c['row_hash'], 'evidence_hash': c['evidence_hash']}), time.time()))
    return {'approved': True}


def batches(units, limit=12000, max_units=32):
    group, chars, images = [], 0, 0
    for unit in units:
        extra = len(unit['text'])
        visual = int(bool(unit['image']))
        if group and (chars+extra > limit or images+visual > 2 or len(group) >= max_units):
            yield group
            group, chars, images = [], 0, 0
        group.append(unit)
        chars += extra
        images += visual
    if group:
        yield group


def compact_units(units):
    return [{k:u[k] for k in ('id','text','language')} for u in units]


def bounded_analysis(groups, run, concurrency):
    """Bound requests in flight; consume in source order for deterministic topic merges."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        iterator = iter(groups)
        pending = deque()
        for _ in range(concurrency):
            group = next(iterator, None)
            if group is not None:
                pending.append((group, pool.submit(run, group)))
        while pending:
            group, future = pending.popleft()
            yield group, future.result()
            group = next(iterator, None)
            if group is not None:
                pending.append((group, pool.submit(run, group)))


class Worker:
    def __init__(self, store):
        self.store = store
        self.name = 'local-'+uid()[:8]
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def start(self):
        self.thread.start()

    def loop(self):
        while not self.stop.wait(1):
            task = self.store.claim(self.name)
            if task:
                self.process(task)

    def process(self, task):
        store = self.store
        doc = store.document(task['document_id'])
        started = time.perf_counter()
        config = settings()

        timings = {}
        current_stage = None
        phase_start = started
        run_id = uid()

        def save_timing():
            nonlocal phase_start
            if current_stage:
                timings[current_stage] = round(timings.get(current_stage, 0)+time.perf_counter()-phase_start, 3)
                phase_start = time.perf_counter()
                store.update_document(doc['id'], timings=timings)

        def stage(name, progress=''):
            nonlocal current_stage, phase_start
            save_timing()
            current_stage, phase_start = name, time.perf_counter()
            store.task_update(task['id'], self.name, name)
            store.update_document(doc['id'], status=name, error='', progress=progress)

        def call_cached(kind, instruction, payload, group):
            store.task_update(task['id'], self.name, kind)
            token_limit = config.get(kind+'_max_tokens', config['max_output_tokens'])
            provider_config = {k:config.get(k) for k in ('base_url','text_model','vision_model','reasoning_effort','json_mode')}
            signature = digest({'kind':kind, 'prompt':instruction, 'payload':payload,
                                'units':group, 'config':provider_config, 'max_tokens':token_limit})
            file = store.root / 'runs' / (signature+'.json')
            if file.is_file():
                store.event(doc['id'], 'cache_hit', {'run_id':run_id, 'stage':kind, 'signature':signature})
                return json.loads(file.read_text())['result']
            try:
                result, stats = GLMClient(config).call(instruction, payload, group, max_tokens=token_limit)
            except ProviderError as error:
                store.event(doc['id'], 'model_error', dict(error.stats, run_id=run_id, stage=kind, code=error.code))
                raise
            temp = file.with_suffix('.tmp')
            temp.write_text(encode({'result':result, 'stats':stats}), encoding='utf-8')
            temp.replace(file)
            store.event(doc['id'], 'model_call', dict(stats, run_id=run_id, stage=kind))
            return result

        def review_candidates(candidates, all_units, compare):
            support = {c['unit_id'] for c in candidates}
            group = [u for u in all_units if u['id'] in support]
            stage('reviewing', f'核验 {len(candidates)} 条候选')
            result = call_cached('review', REVIEW, {'candidates':candidates, 'units':compact_units(group),
                                                   'compare_questions':compare}, group)
            checks = result.get('checks', [])
            if not isinstance(checks, list):
                raise ValueError('审核响应结构无效。')
            for candidate in candidates:
                check = next((c for c in checks if isinstance(c,dict) and c.get('candidate_id')==candidate['candidate_id']), {})
                passed = all(check.get(k) is True for k in CHECKS) and isinstance(check.get('note'),str) and bool(check['note'])
                flags = quality_flags(candidate['qa'])+review_note_flags(check.get('note',''))
                if flags:
                    check['local_flags'] = flags
                    check['note'] = '；'.join([str(check.get('note','')),*flags])
                    passed = False
                with store.connect() as db:
                    db.execute('UPDATE candidates SET status=?,review=? WHERE id=?',
                               ('pending' if passed else 'needs_review', encode(check), candidate['candidate_id']))

        try:
            stage('parsing', '读取原文与图片')
            if doc['units']:
                units, pages = doc['units'], doc['pages']
                expanded = text_alternatives(units)
                if len(expanded)!=len(units):
                    units = expanded
                    store.update_document(doc['id'], units=units)
            else:
                def parsing_progress(done, total):
                    store.task_update(task['id'], self.name, 'parsing')
                    store.update_document(doc['id'], pages=total, progress=f'已解析 {done}/{total} 页')

                units, pages = extract(store, doc, config, progress=parsing_progress)
                store.update_document(doc['id'], units=units, pages=pages)
            if doc['mode'] == 'demo':
                save_timing()
                store.update_document(doc['id'], status='parsed_demo', summary='文件已解析。内部测试不对投入文件生成模拟答案。', coverage='complete', progress='')
                store.task_update(task['id'], self.name, 'parsed_demo', 'succeeded')
                return
            unit_map = {u['id']:u for u in units}
            covered = {u['companion_of'] for u in units if u.get('companion_of')}
            # Text-backed pages are classified from their actual extracted text;
            # pages lacking a text layer still receive full visual analysis.
            text_units = [u for u in units if not u['image']]
            visual_units = [u for u in units if u['image']]
            groups = list(batches(text_units))+list(batches(u for u in visual_units if u['id'] not in covered))
            generation_groups = list(batches(text_units))+list(batches(visual_units))
            stage('analyzing', f'梗概与主题 0/{len(groups)} 批')
            summaries = []
            known_topics = [{k:t[k] for k in ('name','definition','aliases')} for t in store.topics()]

            def analyze(group):
                return call_cached('analysis', ANALYZE, {'units':compact_units(group), 'existing_topics':known_topics}, group)

            concurrency = max(1, min(2, int(config.get('analysis_concurrency', 1))))
            reuse_analysis = doc['coverage']=='complete' and bool(doc['topics'])
            if reuse_analysis:
                for topic in doc['topics']:
                    support = set(topic['unit_ids'])
                    for u in text_units:
                        parent = unit_map.get(u.get('companion_of'))
                        if parent and parent['id'] in support and parent['text']==u['text'] and parent['locator']==u['locator']:
                            support.add(u['id'])
                    store.link_topic(doc['id'], topic['id'], [], sorted(support))
                store.event(doc['id'],'analysis_resumed',{'run_id':run_id,'coverage':'complete'})
                store.update_document(doc['id'],progress='复用已完成的梗概与主题')
            results = () if reuse_analysis else bounded_analysis(groups, analyze, concurrency)
            for index, (group, result) in enumerate(results, 1):
                if not isinstance(result.get('summary'), str) or not isinstance(result.get('topics'), list):
                    raise ValueError('主题分析响应结构无效。')
                summaries.append(result['summary'])
                allowed = {u['id'] for u in group}
                for t in result['topics']:
                    if not isinstance(t,dict):
                        raise ValueError('主题分析响应结构无效。')
                    support = t.get('unit_ids', [])
                    if not isinstance(support,list) or not support or any(not isinstance(x,str) for x in support) or not set(support)<=allowed:
                        raise ValueError('主题缺少当前证据支持。')
                    words = t.get('keywords', [])
                    if not isinstance(words,list) or any(not isinstance(w,str) or len(w)>80 for w in words):
                        raise ValueError('检索词结构无效。')
                    topic = store.topic(t['name'], t.get('definition',''), t.get('new_reason',''))
                    # Both registered views contain exactly the same page text.
                    expanded_support = list(support)
                    for unit_id in support:
                        u = unit_map[unit_id]
                        parent = unit_map.get(u.get('companion_of'))
                        if parent and parent['text']==u['text'] and parent['locator']==u['locator']:
                            expanded_support.append(parent['id'])
                    store.link_topic(doc['id'], topic['id'], words, expanded_support)
                store.update_document(doc['id'], summary='\n'.join(summaries), progress=f'梗概与主题 {index}/{len(groups)} 批')
            store.update_document(doc['id'], coverage='complete')
            doc = store.document(doc['id'])
            if not doc['topics']:
                save_timing()
                store.update_document(doc['id'], status='insufficient', error='没有实质知识主题，不补造问答。', progress='', seconds=round(time.perf_counter()-started,2))
                store.task_update(task['id'], self.name, 'insufficient', 'succeeded')
                return
            existing = [c['row']['input'] for c in doc['candidates'] if c['status']!='rejected']
            # A failed review is resumed before any more generation, including at the target.
            unreviewed = [c for c in doc['candidates'] if c['status'] not in ('approved','rejected') and not c['review']]
            for group in generation_groups:
                support = {u['id'] for u in group}
                pending = [{'candidate_id':c['id'],'unit_id':c['unit_id'],'qa':c['row']} for c in unreviewed if c['unit_id'] in support]
                if pending:
                    questions = {c['qa']['input'] for c in pending}
                    review_candidates(pending, group, [q for q in existing if q not in questions])
            stage('generating', f'已生成 {len(existing)}/{config["target_per_document"]} 条候选')
            batch_size = max(1,min(3,int(config.get('generation_batch_size',3))))
            for group in generation_groups:
                group_ids = {u['id'] for u in group}
                active_topics = [t for t in doc['topics'] if set(t['unit_ids']) & group_ids]
                if not active_topics:
                    continue
                # Two short requests may cover this evidence group; then move to other pages.
                for _ in range(2):
                    remaining = config['target_per_document']-len(existing)
                    if remaining<=0:
                        break
                    target = min(remaining,batch_size)
                    payload = {'units':compact_units(group), 'topics':[{'name':t['name'],'unit_ids':[u for u in t['unit_ids'] if u in group_ids]} for t in active_topics],
                               'target':target, 'existing_questions':existing}
                    try:
                        result = call_cached('generation', GENERATE, payload, group)
                    except ProviderError as error:
                        if error.code not in {'length','json'} or target==1:
                            raise
                        store.event(doc['id'], 'smaller_batch_retry', {'run_id':run_id,'reason':error.code,'target':1})
                        payload['target'] = 1
                        result = call_cached('generation', GENERATE, payload, group)
                        target = 1
                    if not isinstance(result.get('qas'),list) or not isinstance(result.get('skipped'),list):
                        raise ValueError('QA 生成结构无效。')
                    generated = []
                    previous_questions = list(existing)
                    for q in result['qas'][:target]:
                        try:
                            if not isinstance(q,dict):
                                raise ValueError('候选问答结构无效。')
                            unit = next((u for u in group if u['id']==q.get('unit_id')), None)
                            topic = next((t for t in active_topics if t['name']==q.get('topic')), None)
                            if not unit or not topic or unit['id'] not in topic['unit_ids']:
                                raise ValueError('QA 引用了未知单元或未获准主题。')
                            row = {k:q.get(k,'') for k in ['input','thinking','output','language']}
                            row.update(source=source_ref(doc['path'],unit['locator'],topic['name']), image=unit['image'])
                            validate_row(row,unit,topic,doc['path'])
                        except ValueError as error:
                            store.event(doc['id'], 'candidate_invalid', {'run_id':run_id,'reason':str(error)})
                            continue
                        cid = uid()
                        with store.connect() as db:
                            inserted = db.execute('INSERT OR IGNORE INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                                (cid,doc['id'],unit['id'],topic['id'],encode(row),digest(row),unit['hash'],normalized(row['input']),'needs_review','{}',time.time()))
                        if inserted.rowcount:
                            generated.append({'candidate_id':cid,'unit_id':unit['id'],'qa':row})
                            existing.append(row['input'])
                    if not generated:
                        break
                    review_candidates(generated,group,previous_questions)
                    stage('generating', f'已生成 {len(existing)}/{config["target_per_document"]} 条候选')
                if len(existing)>=config['target_per_document']:
                    break
            count = len(existing)
            save_timing()
            seconds = round(time.perf_counter()-started,2)
            store.update_document(doc['id'], status='waiting_review' if count else 'insufficient',
                                  error='' if count>=config['min_per_document'] else f'仅生成 {count} 条候选，质量优先，不补造。',
                                  seconds=seconds, progress=f'{count} 条候选待核对' if count else '')
            store.event(doc['id'], 'run_complete', {'run_id':run_id,'seconds':seconds,'timings':timings,'candidates':count})
            store.task_update(task['id'],self.name,'waiting_review' if count else 'insufficient','succeeded')
        except Exception as error:
            save_timing()
            message = str(error) if isinstance(error,ValueError) else '内部处理失败；已保留证据及候选，可重试继续。'
            store.event(doc['id'], 'run_failed', {'run_id':run_id,'stage':current_stage,'error':message,
                                                'error_type':type(error).__name__,'seconds':round(time.perf_counter()-started,2),'timings':timings})
            store.update_document(doc['id'],status='failed',error=message,progress='',seconds=round(time.perf_counter()-started,2))
            try:
                store.task_update(task['id'],self.name,'failed','failed')
            except ValueError:
                pass
