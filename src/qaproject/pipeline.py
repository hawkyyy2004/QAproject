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
from .store import Store, uid, digest, encode, normalized

CHECKS = ['supported', 'independent', 'scope_correct', 'knowledge_value', 'distinct', 'topic_correct']
BILINGUAL_CHECKS = ['language_correct', 'translation_consistent']


def quality_flags(row):
    flags=[]
    question=row['input']
    if re.search(r'这些|该研究|该报告|本文|文中|上述|^根据图\s*\d',question):
        flags.append('问题可能依赖未定义指代，需要写明对象和范围。')
    if re.search(r'(纵轴|横轴|坐标轴).*(刻度范围|标注为|单位是什么)',question):
        flags.append('单纯读取图轴/刻度可能缺乏领域知识价值。')
    if re.search(r'\b(this study|this report|the above|these results|this figure)\b',question,re.I):
        flags.append('英文问题可能依赖未定义指代，需要写明对象和范围。')
    return flags


def review_note_flags(note):
    if re.search(r'超出|未在本|未在该|未提供|未给出|无法核实|不支持|不足以支持|\b(unsupported|not supported|cannot verify|not provided|outside the evidence)\b', str(note), re.I):
        return ['审核说明提到证据不足或超出原文，需人工检查，不能默认通过。']
    return []


def finish_review(db, document_id):
    pending=db.execute("SELECT COUNT(*) FROM candidates WHERE document_id=? AND status NOT IN ('approved','rejected')",(document_id,)).fetchone()[0]
    if not pending:
        count=db.execute('SELECT COUNT(*) FROM qas q JOIN candidates c ON c.id=q.id WHERE c.document_id=?',(document_id,)).fetchone()[0]
        counts=Store.pair_counts(db,document_id)
        minimum=settings()['min_per_document']
        actual=counts['qa_pair_count'] if counts['pair_count'] else count
        insufficient = (f'实际入库 {count} 条，完整双语 {counts["qa_pair_count"]} 组，低于目标 {minimum} 组；质量优先，不补造。'
                        if counts['pair_count'] else f'历史单语实际入库 {count} 条，低于目标 {minimum} 条；质量优先，不补造。')
        db.execute('UPDATE documents SET status=?,error=? WHERE id=?',
                   ('completed' if actual>=minimum else 'completed_under_min',
                    '' if actual>=minimum else insufficient,document_id))


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
    # Output language is independent of the evidence language; translation is reviewed against evidence.
    if re.search(r'这[篇份个].*(文档|文件|报告).*(讲|介绍|多少)|PDF.*几页|CSV.*多少行|第\s*\d+\s*行.*数值', row['input'], re.I):
        raise ValueError('问题以文件或无意义索引为考点。')


def validate_pair(q, unit, topic, path):
    rows = []
    for language in ('zh', 'en'):
        version = q.get(language)
        if not isinstance(version, dict) or set(version) != {'input','thinking','output'}:
            raise ValueError('双语组必须同时包含中英文问题、依据摘要和答案。')
        row = dict(version, language=language, source=source_ref(path,unit['locator'],topic['name']), image=unit['image'])
        validate_row(row,unit,topic,path)
        rows.append(row)
    if not re.search('[\u4e00-\u9fff]',rows[0]['input']) or not re.search('[A-Za-z]{2,}',rows[1]['input']):
        raise ValueError('双语问题缺少相应语言的正文，需重新生成。')
    if normalized(rows[0]['input']) == normalized(rows[1]['input']):
        raise ValueError('双语问题不能使用同一文本冒充两种语言。')
    return rows


def review_groups(candidates):
    groups = {}
    for c in candidates:
        key = c.get('pair_id') or c['id']
        group = groups.setdefault(key, {'candidate_id':key,'unit_id':c['unit_id'],'versions':[]})
        group['versions'].append({'candidate_id':c['id'],'qa':c['row']})
    return list(groups.values())


def review_binding(group, unit):
    return {'candidate_hashes':{v['candidate_id']:digest(v['qa']) for v in group['versions']},
            'evidence_hash':unit['hash'], 'group_id':group['candidate_id']}


def assert_lease(db, task_id, worker):
    if not db.execute("SELECT 1 FROM tasks WHERE id=? AND worker=? AND status='running' AND lease_until>?",
                      (task_id,worker,time.time())).fetchone():
        raise ValueError('任务租约已失效')


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
        membership = db.execute('SELECT pair_id FROM candidate_pairs WHERE candidate_id=?',(candidate_id,)).fetchone()
        if membership:
            siblings = [dict(x,row=json.loads(x['row_json'])) for x in db.execute('''SELECT c.*,m.language AS pair_language FROM candidates c
                JOIN candidate_pairs m ON m.candidate_id=c.id WHERE m.pair_id=?''',(membership['pair_id'],))]
            if len(siblings)!=2 or {x['row']['language'] for x in siblings}!={'zh','en'}:
                raise ValueError('双语配对不完整，禁止提交。')
            for sibling in siblings:
                if sibling['pair_language']!=sibling['row']['language']:
                    raise ValueError('双语配对的语言标签不一致。')
                if sibling['document_id']!=d['id'] or sibling['unit_id']!=c['unit_id'] or sibling['topic_id']!=c['topic_id'] or sibling['evidence_hash']!=unit['hash']:
                    raise ValueError('双语配对的来源或证据不一致。')
                validate_row(sibling['row'],unit,dict(topic),d['path'])
                if digest(sibling['row'])!=sibling['row_hash']:
                    raise ValueError('双语候选已修改，必须重新审核。')
            expected = review_binding({'candidate_id':membership['pair_id'],
                                       'versions':[{'candidate_id':x['id'],'qa':x['row']} for x in siblings]},unit)
            review = json.loads(c['review'])
            if review.get('binding')!=expected or not all(review.get(k) is True for k in CHECKS+BILINGUAL_CHECKS) or not review.get('note') or review.get('local_flags'):
                raise ValueError('双语候选尚未通过有效的证据与翻译一致性审核，必须重新审核。')
        else:
            review = json.loads(c['review'])
            expected = review_binding({'candidate_id':c['id'],'versions':[{'candidate_id':c['id'],'qa':row}]},unit)
            if review.get('binding')!=expected or not all(review.get(k) is True for k in CHECKS+BILINGUAL_CHECKS) or not review.get('note') or review.get('local_flags'):
                raise ValueError('历史单语候选需重新核验有效证据后入库。')
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
    def __init__(self, store, before_process=None):
        self.store = store
        self.before_process = before_process
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
        if self.before_process:
            self.before_process()
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
                                'units':group, 'config':provider_config, 'max_tokens':token_limit,
                                'review_attempt':task['id'] if kind=='review' and task.get('review_candidate_id') else None})
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
            stage('reviewing', f'核验 {len(candidates)} 组候选与双语一致性')
            result = call_cached('review', REVIEW, {'candidates':candidates, 'units':compact_units(group),
                                                   'compare_questions':compare}, group)
            checks = result.get('checks', [])
            if not isinstance(checks, list):
                raise ValueError('审核响应结构无效。')
            for candidate in candidates:
                check = next((c for c in checks if isinstance(c,dict) and c.get('candidate_id')==candidate['candidate_id']), {})
                passed = all(check.get(k) is True for k in CHECKS+BILINGUAL_CHECKS) and isinstance(check.get('note'),str) and bool(check['note'])
                flags = list(dict.fromkeys([flag for v in candidate['versions'] for flag in quality_flags(v['qa'])]+review_note_flags(check.get('note',''))))
                if flags:
                    check['local_flags'] = flags
                    check['note'] = '；'.join([str(check.get('note','')),*flags])
                    passed = False
                check['binding'] = review_binding(candidate, next(u for u in group if u['id']==candidate['unit_id']))
                with store.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    assert_lease(db,task['id'],self.name)
                    for version in candidate['versions']:
                        current = db.execute('SELECT status,row_hash,review FROM candidates WHERE id=?',(version['candidate_id'],)).fetchone()
                        if current and current['row_hash']==digest(version['qa']) and current['status'] in ('approved','rejected'):
                            continue
                        if current and current['review']!='{}':
                            db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(uid(),doc['id'],'review_replaced',
                                       encode({'candidate_id':version['candidate_id'],'previous_review':json.loads(current['review']),
                                               'row_hash':current['row_hash'],'run_id':run_id}),time.time()))
                        updated = db.execute("UPDATE candidates SET status=?,review=? WHERE id=? AND row_hash=? AND status NOT IN ('approved','rejected')",
                                   ('pending' if passed else 'needs_review', encode(check), version['candidate_id'],digest(version['qa'])))
                        if not updated.rowcount:
                            raise ValueError('候选版本或审核状态已变更，请重新审核。')

        try:
            if task.get('review_candidate_id'):
                target = next((c for c in doc['candidates'] if c['id']==task['review_candidate_id']),None)
                if not target or target['status'] in ('approved','rejected'):
                    raise ValueError('候选状态已变更，无法重新核验。')
                key = target.get('pair_id') or target['id']
                selected = [c for c in doc['candidates'] if (c.get('pair_id') or c['id'])==key]
                unit = next(u for u in doc['units'] if u['id']==target['unit_id'])
                topic = next(t for t in doc['topics'] if t['id']==target['topic_id'])
                for c in selected:
                    validate_row(c['row'],unit,topic,doc['path'])
                if target.get('pair_id') and (len(selected)!=2 or {c['row']['language'] for c in selected}!={'zh','en'}):
                    raise ValueError('双语配对不完整，无法重新核验。')
                review_candidates(review_groups(selected),[unit],
                                  [c['row']['input'] for c in doc['candidates'] if c not in selected and c['status']!='rejected'])
                save_timing()
                store.update_document(doc['id'],status='waiting_review',progress='本组核验完成，请对照证据核对',seconds=round(time.perf_counter()-started,2))
                store.task_update(task['id'],self.name,'waiting_review','succeeded')
                return
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
            unreviewed_keys = {c.get('pair_id') or c['id'] for c in doc['candidates']
                              if c['status'] not in ('approved','rejected') and not c['review'].get('binding')}
            unreviewed = review_groups([c for c in doc['candidates'] if (c.get('pair_id') or c['id']) in unreviewed_keys])
            for group in generation_groups:
                support = {u['id'] for u in group}
                pending = [c for c in unreviewed if c['unit_id'] in support]
                if pending:
                    questions = {v['qa']['input'] for c in pending for v in c['versions']}
                    review_candidates(pending, group, [q for q in existing if q not in questions])
            count = doc['bilingual_count']
            stage('generating', f'已生成 {count}/{config["target_per_document"]} 组双语候选')
            batch_size = max(1,min(3,int(config.get('generation_batch_size',3))))
            for group in generation_groups:
                group_ids = {u['id'] for u in group}
                active_topics = [t for t in doc['topics'] if set(t['unit_ids']) & group_ids]
                if not active_topics:
                    continue
                # Two short requests may cover this evidence group; then move to other pages.
                for _ in range(2):
                    remaining = config['target_per_document']-count
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
                    if not isinstance(result.get('pairs'),list) or not isinstance(result.get('skipped'),list):
                        raise ValueError('双语 QA 生成结构无效；必须返回完整中英配对。')
                    if result['skipped']:
                        store.event(doc['id'],'generation_skipped',{'run_id':run_id,'reasons':[str(x)[:300] for x in result['skipped'][:3]]})
                    generated = []
                    previous_questions = list(existing)
                    for q in result['pairs'][:target]:
                        try:
                            if not isinstance(q,dict):
                                raise ValueError('候选问答结构无效。')
                            unit = next((u for u in group if u['id']==q.get('unit_id')), None)
                            topic = next((t for t in active_topics if t['name']==q.get('topic')), None)
                            if not unit or not topic or unit['id'] not in topic['unit_ids']:
                                raise ValueError('QA 引用了未知单元或未获准主题。')
                            rows = validate_pair(q,unit,topic,doc['path'])
                        except ValueError as error:
                            store.event(doc['id'], 'candidate_invalid', {'run_id':run_id,'reason':str(error)})
                            continue
                        pair_id = uid()
                        versions = []
                        with store.connect() as db:
                            db.execute('BEGIN IMMEDIATE')
                            assert_lease(db,task['id'],self.name)
                            if any(db.execute('SELECT 1 FROM candidates WHERE document_id=? AND fingerprint=?',
                                              (doc['id'],normalized(row['input']))).fetchone() for row in rows):
                                continue  # Never leave one language behind when either version is a duplicate.
                            db.execute('INSERT INTO qa_pairs VALUES(?,?,?)',(pair_id,doc['id'],time.time()))
                            for row in rows:
                                cid = uid()
                                db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                                    (cid,doc['id'],unit['id'],topic['id'],encode(row),digest(row),unit['hash'],normalized(row['input']),'needs_review','{}',time.time()))
                                db.execute('INSERT INTO candidate_pairs VALUES(?,?,?)',(cid,pair_id,row['language']))
                                versions.append({'candidate_id':cid,'qa':row})
                        generated.append({'candidate_id':pair_id,'unit_id':unit['id'],'versions':versions})
                        existing.extend(row['input'] for row in rows)
                        count += 1
                    if not generated:
                        break
                    review_candidates(generated,group,previous_questions)
                    stage('generating', f'已生成 {count}/{config["target_per_document"]} 组双语候选')
                if count>=config['target_per_document']:
                    break
            save_timing()
            seconds = round(time.perf_counter()-started,2)
            has_candidates = bool(existing)
            store.update_document(doc['id'], status='waiting_review' if has_candidates else 'insufficient',
                                  error='' if count>=config['min_per_document'] else f'仅生成 {count} 组完整双语候选，目标 {config["min_per_document"]} 组；质量优先，不补造。',
                                  seconds=seconds, progress=f'{count} 组双语候选 · {len(existing)} 条记录待核对' if has_candidates else '')
            store.event(doc['id'], 'run_complete', {'run_id':run_id,'seconds':seconds,'timings':timings,'pairs':count,'candidates':len(existing)})
            store.task_update(task['id'],self.name,'waiting_review' if has_candidates else 'insufficient','succeeded')
        except Exception as error:
            try:
                with store.connect() as db:
                    assert_lease(db,task['id'],self.name)
            except ValueError:
                store.event(doc['id'],'stale_worker_result',{'run_id':run_id,'stage':current_stage,'error_type':type(error).__name__})
                return
            save_timing()
            message = str(error) if isinstance(error,ValueError) else '内部处理失败；已保留证据及候选，可重试继续。'
            store.event(doc['id'], 'run_failed', {'run_id':run_id,'stage':current_stage,'error':message,
                                                'error_type':type(error).__name__,'seconds':round(time.perf_counter()-started,2),'timings':timings})
            store.update_document(doc['id'],status='failed',error=message,progress='',seconds=round(time.perf_counter()-started,2))
            try:
                store.task_update(task['id'],self.name,'failed','failed')
            except ValueError:
                pass
