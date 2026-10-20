import contextlib
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

from .settings import MIN_DOCUMENT_TOPICS, MAX_DOCUMENT_TOPICS, MAX_DOCUMENT_SEARCH_TERMS

CORE_TOPICS = [
    ('数学', '数学概念、关系、方法与应用。', ['数学研究']),
    ('海洋生态', '海洋生物、生态过程及生态系统影响。', ['海洋生态学']),
    ('海洋能源', '海上风电、潮汐能等海洋相关能源及评价。', ['海洋能源开发']),
    ('海洋经济', '海洋产业、经济指标、价值评价和政策。', ['海洋经济研究']),
    ('海洋观测', '海洋观测、遥感、监测和数据分析方法。', ['海洋监测']),
    ('物理海洋', '海流、海浪、潮汐、海冰及海洋动力和热盐过程。', ['物理海洋学']),
    ('海洋化学', '海水成分、同位素及海洋化学过程。', ['海洋化学研究']),
    ('海洋地质', '海底地貌、沉积、地质构造与演化。', ['海洋地质学']),
    ('海洋探索', '面向未知海域或深海环境的调查与探索。', []),
    ('海洋气象', '海上天气、台风及海气相互作用。', []),
    ('海洋工程', '海洋设施、结构与工程技术。', []),
]

# Reviewed parent relationships, not synonyms. Never infer these from substring
# matches or embedding similarity. Unlisted unique topics remain independent.
TOPIC_PARENTS = {
    **dict.fromkeys(('物理海洋学', '南海环流', '海洋环流', '太平洋年代际振荡',
                     '北极海冰数值预报', '海冰同化', '海冰预报', '海冰数值预报与资料同化',
                     '海冰预报与资料同化'), '物理海洋'),
    **dict.fromkeys(('同位素海洋学', '同位素海洋化学'), '海洋化学'),
    '台风极值风速预测': '海洋气象',
}


def classification_catalog(topics):
    """Compact shared names and scopes; known fine directions are not new classes."""
    names = {t['name'] for t in topics}
    return {t['name']:t['definition'] for t in topics
            if TOPIC_PARENTS.get(t['name']) not in names}


def uid():
    return uuid.uuid4().hex


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def digest(obj):
    return hashlib.sha256(encode(obj).encode()).hexdigest()


def normalized(value):
    return re.sub(r'[\W_]+', '', value.casefold())


def search_term_key(value):
    # Symbols can distinguish concepts (C/C++, 2.5/25); never erase them to merge terms.
    return ' '.join(value.casefold().split())


def limit_search_terms(terms, limit=MAX_DOCUMENT_SEARCH_TERMS):
    """Keep ranked terms, normalize spelling duplicates and retain their evidence union."""
    result, names = {}, {}
    for word, support in terms.items():
        word = word.strip()
        key = search_term_key(word)
        if not key:
            continue
        if key in names:
            result[names[key]].update(support)
        elif limit is None or len(result) < limit:
            names[key] = word
            result[word] = set(support)
    return result


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        for name in ['originals', 'assets/images', 'runs', 'exports']:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'state.sqlite3'
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS documents(
              id TEXT PRIMARY KEY, content_hash TEXT UNIQUE, name TEXT, path TEXT,
              status TEXT, summary TEXT DEFAULT '', error TEXT DEFAULT '',
              pages INTEGER DEFAULT 0, units TEXT DEFAULT '[]', created REAL,
              seconds REAL DEFAULT 0, mode TEXT, coverage TEXT DEFAULT 'pending');
            CREATE TABLE IF NOT EXISTS topics(
              id TEXT PRIMARY KEY, name TEXT UNIQUE, definition TEXT, aliases TEXT);
            CREATE TABLE IF NOT EXISTS document_topics(
              document_id TEXT REFERENCES documents(id), topic_id TEXT REFERENCES topics(id),
              keywords TEXT, unit_ids TEXT, PRIMARY KEY(document_id,topic_id));
            CREATE TABLE IF NOT EXISTS assets(
              id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id),
              path TEXT, hash TEXT, page INTEGER, label TEXT);
            CREATE TABLE IF NOT EXISTS document_search_terms(
              document_id TEXT REFERENCES documents(id), term TEXT,
              topic_id TEXT REFERENCES topics(id), unit_ids TEXT,
              PRIMARY KEY(document_id,term));
            CREATE TABLE IF NOT EXISTS document_topic_redirects(
              document_id TEXT REFERENCES documents(id), old_topic_id TEXT REFERENCES topics(id),
              topic_id TEXT REFERENCES topics(id), unit_ids TEXT, created REAL,
              PRIMARY KEY(document_id,old_topic_id));
            CREATE TABLE IF NOT EXISTS candidates(
              id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id),
              unit_id TEXT, topic_id TEXT REFERENCES topics(id), row_json TEXT,
              row_hash TEXT, evidence_hash TEXT, fingerprint TEXT,
              status TEXT, review TEXT, created REAL,
              UNIQUE(document_id,fingerprint));
            CREATE TABLE IF NOT EXISTS qas(
              id TEXT PRIMARY KEY REFERENCES candidates(id), row_json TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS qa_pairs(
              id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id), created REAL);
            CREATE TABLE IF NOT EXISTS candidate_pairs(
              candidate_id TEXT PRIMARY KEY REFERENCES candidates(id),
              pair_id TEXT NOT NULL REFERENCES qa_pairs(id),
              language TEXT NOT NULL CHECK(language IN ('zh','en')),
              UNIQUE(pair_id,language));
            CREATE TABLE IF NOT EXISTS tasks(
              id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id),
              status TEXT, worker TEXT, lease_until REAL DEFAULT 0,
              attempts INTEGER DEFAULT 0, stage TEXT DEFAULT 'queued');
            CREATE TABLE IF NOT EXISTS events(
              id TEXT PRIMARY KEY, document_id TEXT, kind TEXT, detail TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS qa_revisions(
              id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id),
              before_json TEXT NOT NULL, after_json TEXT NOT NULL, status TEXT NOT NULL,
              note TEXT, review TEXT, created REAL, updated REAL, task_id TEXT);
            CREATE INDEX IF NOT EXISTS qa_revision_documents ON qa_revisions(document_id,created DESC,id DESC);
            CREATE INDEX IF NOT EXISTS qas_created ON qas(created DESC,id DESC);
            CREATE TABLE IF NOT EXISTS schema_migrations(name TEXT PRIMARY KEY, applied REAL);
            CREATE INDEX IF NOT EXISTS tasks_state ON tasks(status,lease_until);
            CREATE INDEX IF NOT EXISTS topic_docs ON document_topics(topic_id);
            CREATE INDEX IF NOT EXISTS candidate_docs ON candidates(document_id);
            CREATE INDEX IF NOT EXISTS pair_docs ON qa_pairs(document_id);
            ''')
            if db.execute('PRAGMA user_version').fetchone()[0] < 1:
                db.execute('PRAGMA user_version=1')
            if 'review_candidate_id' not in {r['name'] for r in db.execute('PRAGMA table_info(tasks)')}:
                db.execute('ALTER TABLE tasks ADD COLUMN review_candidate_id TEXT REFERENCES candidates(id)')
            if 'revision_id' not in {r['name'] for r in db.execute('PRAGMA table_info(tasks)')}:
                db.execute('ALTER TABLE tasks ADD COLUMN revision_id TEXT REFERENCES qa_revisions(id)')
            if 'repair_attempts' not in {r['name'] for r in db.execute('PRAGMA table_info(qa_pairs)')}:
                db.execute('ALTER TABLE qa_pairs ADD COLUMN repair_attempts INTEGER NOT NULL DEFAULT 0')
            if 'unit_count' not in {r['name'] for r in db.execute('PRAGMA table_info(documents)')}:
                db.execute('ALTER TABLE documents ADD COLUMN unit_count INTEGER DEFAULT 0')
                db.execute('UPDATE documents SET unit_count=json_array_length(units)')
            columns = {r['name'] for r in db.execute('PRAGMA table_info(documents)')}
            for name, default in [('progress', "''"), ('timings', "'{}'")]:
                if name not in columns:
                    db.execute(f'ALTER TABLE documents ADD COLUMN {name} TEXT DEFAULT {default}')
            for name, definition, aliases in CORE_TOPICS:
                db.execute('INSERT OR IGNORE INTO topics VALUES(?,?,?,?)',
                           (uid(), name, definition, encode(aliases)))
            migration = 'document_search_terms_limit_10_v1'
            if not db.execute('SELECT 1 FROM schema_migrations WHERE name=?', (migration,)).fetchone():
                for row in db.execute('SELECT DISTINCT document_id FROM document_topics').fetchall():
                    self._limit_document_terms(db, row[0])
                db.execute('INSERT INTO schema_migrations VALUES(?,?)', (migration,time.time()))
            self._reuse_shared_topics(db)
            # Leased tasks recover only after expiry, including across restarts.
            db.execute("UPDATE tasks SET status='queued' WHERE status='running' AND lease_until<?", (time.time(),))
            db.execute("""UPDATE documents SET status='queued',progress='等待恢复未完成的处理'
                WHERE id IN (SELECT document_id FROM tasks WHERE status='queued')
                AND status IN ('parsing','analyzing','generating','reviewing','repairing','repair_reviewing','preflight')""")

            # Recover old releases' excess labels through the same worker.
            for row in db.execute('''SELECT d.id FROM documents d JOIN document_topics dt ON dt.document_id=d.id
                WHERE d.mode='live' GROUP BY d.id HAVING COUNT(*)>5''').fetchall():
                if not db.execute("SELECT 1 FROM tasks WHERE document_id=? AND status IN ('queued','running')",(row[0],)).fetchone():
                    task_id = uid()
                    db.execute('INSERT INTO tasks(id,document_id,status,stage) VALUES(?,?,?,?)',
                               (task_id,row[0],'queued','topic_reclassification'))
                    db.execute("UPDATE documents SET status='queued',progress='系统自动概括整篇正式主题，无需手动筛选' WHERE id=?",(row[0],))
                    db.execute('INSERT INTO events VALUES(?,?,?,?,?)',
                               (uid(),row[0],'topic_reclassification_queued',encode({'task_id':task_id}),time.time()))

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA journal_mode=WAL')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def event(self, doc_id, kind, detail):
        with self.connect() as db:
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)', (uid(), doc_id, kind, encode(detail), time.time()))

    def _reuse_shared_topics(self, db):
        """Lift known legacy directions locally; preserve evidence and source snapshots."""
        catalog = {r['name']: dict(r) for r in db.execute('SELECT * FROM topics')}
        names_by_id = {t['id']: name for name,t in catalog.items()}
        parents = {catalog[name]['id']: catalog[parent]['id'] for name, parent in TOPIC_PARENTS.items()
                   if name in catalog and parent in catalog}
        if not parents:
            return
        marks = ','.join('?' for _ in parents)
        affected = db.execute(f'''SELECT DISTINCT document_id FROM document_topics
            WHERE topic_id IN ({marks}) AND document_id NOT IN
            (SELECT document_id FROM tasks WHERE status='running' AND lease_until>?)''',
            (*parents, time.time())).fetchall()
        for row in affected:
            doc_id = row[0]
            previous = [dict(r) for r in db.execute('SELECT * FROM document_topics WHERE document_id=?', (doc_id,))]
            previous_maps = [dict(r) for r in db.execute('SELECT * FROM document_topic_redirects WHERE document_id=?', (doc_id,))]
            merged = {}
            for old in previous:
                target = parents.get(old['topic_id'], old['topic_id'])
                item = merged.setdefault(target, dict(keywords=[], unit_ids=set()))
                item['keywords'].extend(json.loads(old['keywords']))
                item['unit_ids'].update(json.loads(old['unit_ids']))
                if target != old['topic_id']:
                    # Keep a fine direction searchable only on its own old support.
                    old_name = names_by_id[old['topic_id']]
                    db.execute('''INSERT INTO document_search_terms VALUES(?,?,?,?)
                        ON CONFLICT(document_id,term) DO NOTHING''',
                        (doc_id, old_name, target, old['unit_ids']))
                    db.execute('UPDATE document_search_terms SET topic_id=? WHERE document_id=? AND topic_id=?',
                               (target,doc_id,old['topic_id']))
                    support = {r[0] for r in db.execute('SELECT unit_id FROM candidates WHERE document_id=? AND topic_id=?',
                                                       (doc_id,old['topic_id']))}
                    item['unit_ids'].update(support)
                    db.execute('INSERT OR REPLACE INTO document_topic_redirects VALUES(?,?,?,?,?)',
                               (doc_id,old['topic_id'],target,encode(sorted(support)),time.time()))
                    db.execute('UPDATE document_topic_redirects SET topic_id=? WHERE document_id=? AND topic_id=?',
                               (target,doc_id,old['topic_id']))
            db.execute('DELETE FROM document_topics WHERE document_id=?', (doc_id,))
            db.executemany('INSERT INTO document_topics VALUES(?,?,?,?)',
                [(doc_id,topic_id,encode(list(dict.fromkeys(t['keywords']))),encode(sorted(t['unit_ids'])))
                 for topic_id,t in merged.items()])
            self._limit_document_terms(db,doc_id)
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)', (uid(),doc_id,'shared_topics_reused',
                encode({'previous':previous,'previous_mappings':previous_maps,
                        'selected_ids':list(merged),'policy':'shared_domains_v1'}),time.time()))

    def topics(self):
        with self.connect() as db:
            rows = db.execute('SELECT t.*,COUNT(dt.document_id) as documents FROM topics t LEFT JOIN document_topics dt ON dt.topic_id=t.id GROUP BY t.id ORDER BY t.name').fetchall()
        return [dict(r, aliases=json.loads(r['aliases'])) for r in rows]

    def topic(self, name, definition='', new_reason=''):
        norm = normalized(name)
        for item in self.topics():
            if any(normalized(x) == norm for x in [item['name'], *item['aliases']]):
                return item
        if not name.strip() or len(name) > 24 or not definition.strip() or not new_reason.strip():
            raise ValueError('新主题未通过名称、定义和独立分类理由校验，保留候选待归类。')
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO topics VALUES(?,?,?,?)', (uid(), name.strip(), definition, '[]'))
        return next(t for t in self.topics() if t['name'] == name.strip())

    def import_file(self, name, content, mode):
        h = hashlib.sha256(content).hexdigest()
        with self.connect() as db:
            old = db.execute('SELECT id FROM documents WHERE content_hash=?', (h,)).fetchone()
            if old:
                return old['id'], False
            doc_id = uid()
            name = Path(name.replace('\\', '/')).name
            if not name or name in {'.', '..'}:
                raise ValueError('文件名无效')
            folder = self.root / 'originals' / doc_id
            folder.mkdir()
            path = folder / name
            path.write_bytes(content)
            db.execute('INSERT INTO documents(id,content_hash,name,path,status,created,mode) VALUES(?,?,?,?,?,?,?)',
                       (doc_id, h, name, str(path), 'queued', time.time(), mode))
            db.execute('INSERT INTO tasks(id,document_id,status) VALUES(?,?,?)', (uid(), doc_id, 'queued'))
        return doc_id, True

    def document(self, doc_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
            if not row:
                raise ValueError('找不到文件')
            item = dict(row)
            item['units'] = json.loads(item['units'])
            item['timings'] = json.loads(item['timings'])
            item['topics'] = [dict(x, keywords=json.loads(x['keywords']), unit_ids=json.loads(x['unit_ids'])) for x in db.execute(
                'SELECT t.*,dt.keywords,dt.unit_ids FROM document_topics dt JOIN topics t ON t.id=dt.topic_id WHERE document_id=?', (doc_id,))]
            item['search_terms'] = [dict(x, unit_ids=json.loads(x['unit_ids'])) for x in db.execute(
                'SELECT term,topic_id,unit_ids FROM document_search_terms WHERE document_id=?', (doc_id,))]
            by_topic = {t['id']: t for t in item['topics']}
            seen = {t['id']: set(t['keywords']) for t in item['topics']}
            for term in item['search_terms']:
                if term['topic_id'] in by_topic and term['term'] not in seen[term['topic_id']]:
                    by_topic[term['topic_id']]['keywords'].append(term['term'])
                    seen[term['topic_id']].add(term['term'])
            item['candidates'] = [dict(x, row=json.loads(x['row_json']), review=json.loads(x['review'])) for x in db.execute(
                '''SELECT c.*,p.pair_id FROM candidates c LEFT JOIN candidate_pairs p ON p.candidate_id=c.id
                   WHERE c.document_id=? AND c.status!='discarded' ORDER BY c.created,c.rowid''', (doc_id,))]
            item['discarded_pair_count'] = db.execute('''SELECT COUNT(DISTINCT m.pair_id) FROM candidates c
                JOIN candidate_pairs m ON m.candidate_id=c.id WHERE c.document_id=? AND c.status='discarded' ''', (doc_id,)).fetchone()[0]
            item['historical_topics'] = [dict(r,unit_ids=json.loads(r['unit_ids'])) for r in db.execute(
                '''SELECT t.*,m.topic_id AS canonical_topic_id,m.unit_ids FROM document_topic_redirects m
                   JOIN topics t ON t.id=m.old_topic_id WHERE m.document_id=?''',(doc_id,))]
            redirects = {t['id']:t['canonical_topic_id'] for t in item['historical_topics']}
            for candidate in item['candidates']:
                candidate['canonical_topic_id'] = redirects.get(candidate['topic_id'],candidate['topic_id'])
            item['qa_count'] = db.execute('SELECT COUNT(*) FROM qas q JOIN candidates c ON q.id=c.id WHERE c.document_id=?', (doc_id,)).fetchone()[0]
            item.update(self.pair_counts(db, doc_id))
            item['assets'] = [dict(x) for x in db.execute('SELECT * FROM assets WHERE document_id=?', (doc_id,))]
        for candidate in item['candidates']:
            candidate.pop('row_json', None)
        return item

    def documents(self, summary_only=False):
        if summary_only:
            # List/search never load evidence bodies, image references or QA answers.
            with self.connect() as db:
                rows = db.execute('''SELECT d.id,d.name,d.status,substr(d.summary,1,500) AS summary,
                    d.error,d.pages,d.unit_count,d.created,d.seconds,d.mode,d.coverage,d.progress,
                    (SELECT COUNT(*) FROM candidates c WHERE c.document_id=d.id AND c.status!='discarded') AS candidate_count,
                    (SELECT COUNT(*) FROM candidates c WHERE c.document_id=d.id
                        AND c.status NOT IN ('approved','rejected','discarded')) AS pending_commit_count,
                    (SELECT COUNT(*) FROM candidates c JOIN qas q ON q.id=c.id
                        WHERE c.document_id=d.id) AS qa_count
                    FROM documents d ORDER BY d.created DESC''').fetchall()
                items = {r['id']: dict(r, topics=[]) for r in rows}
                for r in db.execute('''SELECT dt.document_id,t.id,t.name,dt.keywords
                    FROM document_topics dt JOIN topics t ON t.id=dt.topic_id'''):
                    items[r['document_id']]['topics'].append(
                        {'id':r['id'], 'name':r['name'], 'keywords':json.loads(r['keywords'])})
                by_topic = {(doc_id, t['id']): t for doc_id, d in items.items() for t in d['topics']}
                seen = {key: set(t['keywords']) for key, t in by_topic.items()}
                for r in db.execute('SELECT document_id,topic_id,term FROM document_search_terms'):
                    key = (r['document_id'], r['topic_id'])
                    topic = by_topic.get(key)
                    if topic is not None and r['term'] not in seen[key]:
                        topic['keywords'].append(r['term'])
                        seen[key].add(r['term'])
                for item in items.values():
                    item.update(self.pair_counts(db, item['id']))
            return list(items.values())
        with self.connect() as db:
            ids = [r[0] for r in db.execute('SELECT id FROM documents ORDER BY created DESC')]
        return [self.document(i) for i in ids]

    @staticmethod
    def pair_counts(db, doc_id):
        # Count metadata only: list/search must never decode answers or evidence.
        counts = db.execute('''SELECT COUNT(*) AS pair_count,
            COALESCE(SUM(active=2),0) AS bilingual_count,
            COALESCE(SUM(committed=2),0) AS qa_pair_count FROM (
              SELECT p.id,SUM(c.status NOT IN ('rejected','discarded')) AS active,COUNT(q.id) AS committed
              FROM qa_pairs p JOIN candidate_pairs m ON m.pair_id=p.id
              JOIN candidates c ON c.id=m.candidate_id LEFT JOIN qas q ON q.id=c.id
              WHERE p.document_id=? GROUP BY p.id)''', (doc_id,)).fetchone()
        legacy = db.execute('''SELECT COUNT(*) FROM candidates c LEFT JOIN candidate_pairs m ON m.candidate_id=c.id
            WHERE c.document_id=? AND m.candidate_id IS NULL''', (doc_id,)).fetchone()[0]
        return dict(counts, legacy_count=legacy)

    def update_document(self, doc_id, **values):
        allowed = {'status', 'summary', 'error', 'pages', 'units', 'seconds', 'coverage', 'progress', 'timings'}
        if not set(values) <= allowed:
            raise ValueError('不允许更新字段')
        if 'units' in values:
            values['unit_count'] = len(values['units'])
            values['units'] = encode(values['units'])
        if 'timings' in values:
            values['timings'] = encode(values['timings'])
        with self.connect() as db:
            db.execute('UPDATE documents SET '+','.join(k+'=?' for k in values)+' WHERE id=?', [*values.values(), doc_id])

    def link_topic(self, doc_id, topic_id, keywords, unit_ids):
        incoming_keywords, incoming_support = list(keywords), list(unit_ids)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT keywords,unit_ids FROM document_topics WHERE document_id=? AND topic_id=?', (doc_id, topic_id)).fetchone()
            if not previous and db.execute('SELECT COUNT(*) FROM document_topics WHERE document_id=?', (doc_id,)).fetchone()[0] >= MAX_DOCUMENT_TOPICS:
                raise ValueError('每篇文件最多5个正式主题，须先按全文筛选。')
            if previous:
                keywords = list(dict.fromkeys([*json.loads(previous['keywords']), *keywords]))
                unit_ids = list(dict.fromkeys([*json.loads(previous['unit_ids']), *unit_ids]))
            db.execute('INSERT OR REPLACE INTO document_topics VALUES(?,?,?,?)', (doc_id, topic_id, encode(keywords), encode(unit_ids)))
            existing_terms = {search_term_key(r['term']): dict(r) for r in db.execute(
                'SELECT term,topic_id,unit_ids FROM document_search_terms WHERE document_id=?', (doc_id,))}
            for word in incoming_keywords:
                key = search_term_key(word)
                if not key:
                    continue
                old = existing_terms.get(key)
                support = sorted(set(incoming_support) | set(json.loads(old['unit_ids']) if old else []))
                record = dict(term=old['term'] if old else word.strip(), topic_id=old['topic_id'] if old else topic_id,
                              unit_ids=encode(support))
                db.execute('''INSERT INTO document_search_terms VALUES(?,?,?,?)
                           ON CONFLICT(document_id,term) DO UPDATE SET unit_ids=excluded.unit_ids''',
                           (doc_id,record['term'],record['topic_id'],record['unit_ids']))
                existing_terms[key] = record
            self._limit_document_terms(db, doc_id)

    @staticmethod
    def _limit_document_terms(db, doc_id):
        """Cap both keyword stores together; never read or change evidence/QA bodies."""
        topics = [dict(r) for r in db.execute(
            'SELECT topic_id,keywords,unit_ids FROM document_topics WHERE document_id=? ORDER BY rowid', (doc_id,))]
        previous = [dict(r) for r in db.execute(
            'SELECT term,topic_id,unit_ids FROM document_search_terms WHERE document_id=? ORDER BY rowid', (doc_id,))]
        topic_ids = {t['topic_id'] for t in topics}
        pool, owners, explicit = {}, {}, set()
        # Explicit term evidence has finer scope than a legacy topic's unit list.
        for term in previous:
            if term['topic_id'] not in topic_ids:
                continue
            word, key = term['term'].strip(), search_term_key(term['term'])
            pool.setdefault(word, set()).update(json.loads(term['unit_ids']))
            owners.setdefault(key, term['topic_id'])
            explicit.add(key)
        for topic in topics:
            for word in json.loads(topic['keywords']):
                key = search_term_key(word)
                if key not in explicit:
                    pool.setdefault(word, set()).update(json.loads(topic['unit_ids']))
                    owners.setdefault(key, topic['topic_id'])
        limited = limit_search_terms(pool)
        rows = [dict(term=word, topic_id=owners[search_term_key(word)], unit_ids=encode(sorted(support)))
                for word, support in limited.items()]
        keywords = {t['topic_id']: [] for t in topics}
        for row in rows:
            keywords[row['topic_id']].append(row['term'])
        changed = rows != previous or any(json.loads(t['keywords']) != keywords[t['topic_id']] for t in topics)
        if not changed:
            return
        for topic in topics:
            db.execute('UPDATE document_topics SET keywords=? WHERE document_id=? AND topic_id=?',
                       (encode(keywords[topic['topic_id']]),doc_id,topic['topic_id']))
        db.execute('DELETE FROM document_search_terms WHERE document_id=?', (doc_id,))
        db.executemany('INSERT INTO document_search_terms VALUES(?,?,?,?)',
                       [(doc_id,r['term'],r['topic_id'],r['unit_ids']) for r in rows])
        db.execute('INSERT INTO events VALUES(?,?,?,?,?)', (uid(),doc_id,'document_search_terms_limited',
                   encode({'previous_topics':topics,'previous_terms':previous,'retained':[r['term'] for r in rows],
                           'max_per_document':MAX_DOCUMENT_SEARCH_TERMS}),time.time()))

    def reclassify_topics(self, doc_id):
        """Enqueue automatic classification through the normal worker; no user selection."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM documents WHERE id=?',(doc_id,)).fetchone():
                raise ValueError('找不到文件')
            active = db.execute("SELECT id FROM tasks WHERE document_id=? AND status IN ('queued','running')",(doc_id,)).fetchone()
            if active:
                return active['id']
            task_id = uid()
            db.execute('INSERT INTO tasks(id,document_id,status,stage) VALUES(?,?,?,?)',
                       (task_id,doc_id,'queued','topic_reclassification'))
            db.execute("UPDATE documents SET status='queued',progress='系统自动概括整篇正式主题，无需手动筛选' WHERE id=?",(doc_id,))
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(uid(),doc_id,'topic_reclassification_queued',encode({'task_id':task_id}),time.time()))
            return task_id

    def legacy_topic_context(self, doc_id):
        with self.connect() as db:
            rows = db.execute('''SELECT DISTINCT t.* FROM topics t WHERE t.id IN
                (SELECT topic_id FROM document_topics WHERE document_id=? UNION
                 SELECT topic_id FROM candidates WHERE document_id=?)''',(doc_id,doc_id)).fetchall()
            return [dict(id=t['id'],name=t['name'],definition=t['definition'],unit_ids=list(dict.fromkeys(
                [r[0] for r in db.execute('SELECT unit_id FROM candidates WHERE document_id=? AND topic_id=?',(doc_id,t['id']))]
                ))) for t in rows]

    def apply_topic_selection(self, doc_id, selected, terms, summary, task_id, worker, *, redirects=None):
        """Register only selected topics and publish classification in one leased transaction."""
        if not MIN_DOCUMENT_TOPICS <= len(selected) <= MAX_DOCUMENT_TOPICS:
            raise ValueError('每篇文件须有1至5个正式主题。')
        terms = limit_search_terms(terms)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute("SELECT 1 FROM tasks WHERE id=? AND document_id=? AND worker=? AND status='running' AND lease_until>?",
                              (task_id, doc_id, worker, time.time())).fetchone():
                raise ValueError('任务租约已失效')
            previous = [dict(r) for r in db.execute('SELECT * FROM document_topics WHERE document_id=?', (doc_id,))]
            catalog = {normalized(n): dict(t) for t in db.execute('SELECT * FROM topics')
                       for n in [t['name'], *json.loads(t['aliases'])]}
            records = []
            for t in selected:
                topic = catalog.get(normalized(t['name']))
                if not topic:
                    if not t['name'].strip() or len(t['name']) > 24 or not t['definition'].strip() or not t['new_reason'].strip():
                        raise ValueError('新主题缺少规范名称、定义或独立分类理由。')
                    topic = dict(id=uid(), name=t['name'])
                    db.execute('INSERT INTO topics VALUES(?,?,?,?)',
                               (topic['id'], t['name'], t['definition'], '[]'))
                    catalog[normalized(t['name'])] = topic
                records.append(dict(t, id=topic['id'], unit_ids=set(t['unit_ids'])))
            ids = {t['id'] for t in records}
            required = {r[0] for r in db.execute('SELECT DISTINCT topic_id FROM candidates WHERE document_id=?', (doc_id,))}
            by_name = {t['name']:t for t in records}
            redirects = redirects or {}
            if any(old not in ids and redirects.get(old) not in by_name for old in required):
                raise ValueError('整篇主题规划缺少历史候选来源映射，系统需重新规划。')
            previous_maps = [dict(r) for r in db.execute('SELECT * FROM document_topic_redirects WHERE document_id=?',(doc_id,))]
            db.execute('DELETE FROM document_topic_redirects WHERE document_id=?',(doc_id,))
            for old_id, name in redirects.items():
                if name not in by_name:
                    raise ValueError('历史来源映射指向未知正式主题。')
                topic = by_name[name]
                if old_id==topic['id']:
                    continue
                support = sorted({r[0] for r in db.execute('SELECT unit_id FROM candidates WHERE document_id=? AND topic_id=?',(doc_id,old_id))})
                if not set(support)<=topic['unit_ids']:
                    raise ValueError('历史来源映射缺少对应真实单元支持，系统需重新规划。')
                db.execute('INSERT INTO document_topic_redirects VALUES(?,?,?,?,?)',
                           (doc_id,old_id,topic['id'],encode(support),time.time()))
            db.execute('DELETE FROM document_search_terms WHERE document_id=?', (doc_id,))
            db.execute('DELETE FROM document_topics WHERE document_id=?', (doc_id,))
            # Group a file's fine-term hits under its best supported selected topic,
            # falling back to its primary topic for display, never for QA evidence.
            term_rows = []
            keywords = {t['id']: [] for t in records}
            for word, support in terms.items():
                support = set(support)
                topic = max(records, key=lambda t: len(support & t['unit_ids']))
                term_rows.append((doc_id,word,topic['id'],encode(sorted(support))))
                keywords[topic['id']].append(word)
            db.executemany('INSERT INTO document_topics VALUES(?,?,?,?)',
                [(doc_id,t['id'],encode(keywords[t['id']]),encode(sorted(t['unit_ids']))) for t in records])
            db.executemany('INSERT INTO document_search_terms VALUES(?,?,?,?)', term_rows)
            db.execute("UPDATE documents SET coverage='complete',summary=? WHERE id=?", (summary,doc_id))
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)', (uid(),doc_id,'document_topics_selected',
                encode({'previous':previous,'selected_ids':[t['id'] for t in records], 'term_count':len(terms),
                        'max_per_document':MAX_DOCUMENT_TOPICS,'previous_mappings':previous_maps,
                        'source_mappings':redirects,'policy':'whole_document_direct_v2'}),time.time()))

    def add_asset(self, doc_id, path, page, label):
        asset_id = uid()
        with self.connect() as db:
            db.execute('INSERT INTO assets VALUES(?,?,?,?,?,?)', (asset_id, doc_id, str(path), hashlib.sha256(path.read_bytes()).hexdigest(), page, label))
        return asset_id

    def claim(self, worker, document_id=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("UPDATE tasks SET status='queued' WHERE status='running' AND lease_until<?", (time.time(),))
            row = db.execute("SELECT * FROM tasks WHERE status='queued'"+
                             (' AND document_id=?' if document_id else '')+
                             " ORDER BY CASE WHEN stage='quality_queued' OR review_candidate_id IS NOT NULL OR revision_id IS NOT NULL THEN 2 WHEN stage='topic_reclassification' THEN 1 ELSE 0 END,rowid LIMIT 1",
                             (document_id,) if document_id else ()).fetchone()
            if not row:
                return None
            db.execute("UPDATE tasks SET status='running',worker=?,lease_until=?,attempts=attempts+1 WHERE id=?", (worker, time.time()+1800, row['id']))
            return dict(row)

    def task_update(self, task_id, worker, stage, status='running'):
        with self.connect() as db:
            result = db.execute('UPDATE tasks SET stage=?,status=?,lease_until=? WHERE id=? AND worker=? AND status=\'running\' AND lease_until>?',
                                (stage, status, time.time()+1800, task_id, worker, time.time()))
            if result.rowcount != 1:
                raise ValueError('任务租约已失效')

    def search(self, query):
        query = normalized(query)
        results = []
        documents = self.documents(summary_only=True)
        for topic in self.topics():
            all_docs = [d for d in documents if any(t['id'] == topic['id'] for t in d['topics'])]
            exact_topic = not query or any(query in normalized(x) for x in [topic['name'], *topic['aliases']])
            matched, terms = [], set()
            for doc in all_docs:
                terms_for_doc = next(t['keywords'] for t in doc['topics'] if t['id'] == topic['id'])
                matching = [x for x in terms_for_doc if query in normalized(x)] if query else terms_for_doc
                if exact_topic or matching:
                    matched.append(doc)
                    terms.update(terms_for_doc if exact_topic else matching)
            if matched:
                results.append({'topic': topic, 'terms': sorted(terms), 'documents': matched,
                                'reason': '主题或别名' if exact_topic else '具体检索方向'})
        if query:
            def match_rank(result):
                topic = result['topic']
                if normalized(topic['name']) == query:
                    return 0
                if query in [normalized(a) for a in topic['aliases']]:
                    return 1
                return 2 if result['reason'] == '主题或别名' else 3
            results.sort(key=lambda r: (match_rank(r), r['topic']['name']))
        return results
