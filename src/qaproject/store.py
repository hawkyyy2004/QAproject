import contextlib
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

CORE_TOPICS = [
    ('数学', '数学概念、关系、方法与应用。', ['数学研究']),
    ('海洋生态', '海洋生物、生态过程及生态系统影响。', ['海洋生态学']),
    ('海洋能源', '海上风电、潮汐能等海洋相关能源及评价。', ['海洋能源开发']),
    ('海洋经济', '海洋产业、经济指标、价值评价和政策。', ['海洋经济研究']),
    ('海洋观测', '海洋观测、遥感、监测和数据分析方法。', ['海洋监测']),
]


def uid():
    return uuid.uuid4().hex


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def digest(obj):
    return hashlib.sha256(encode(obj).encode()).hexdigest()


def normalized(value):
    return re.sub(r'[\W_]+', '', value.casefold())


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
            CREATE TABLE IF NOT EXISTS candidates(
              id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id),
              unit_id TEXT, topic_id TEXT REFERENCES topics(id), row_json TEXT,
              row_hash TEXT, evidence_hash TEXT, fingerprint TEXT,
              status TEXT, review TEXT, created REAL,
              UNIQUE(document_id,fingerprint));
            CREATE TABLE IF NOT EXISTS qas(
              id TEXT PRIMARY KEY REFERENCES candidates(id), row_json TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS tasks(
              id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id),
              status TEXT, worker TEXT, lease_until REAL DEFAULT 0,
              attempts INTEGER DEFAULT 0, stage TEXT DEFAULT 'queued');
            CREATE TABLE IF NOT EXISTS events(
              id TEXT PRIMARY KEY, document_id TEXT, kind TEXT, detail TEXT, created REAL);
            CREATE INDEX IF NOT EXISTS tasks_state ON tasks(status,lease_until);
            CREATE INDEX IF NOT EXISTS topic_docs ON document_topics(topic_id);
            CREATE INDEX IF NOT EXISTS candidate_docs ON candidates(document_id);
            ''')
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
            # Leased tasks recover only after expiry, including across restarts.
            db.execute("UPDATE tasks SET status='queued' WHERE status='running' AND lease_until<?", (time.time(),))
            db.execute("""UPDATE documents SET status='queued',progress='等待恢复未完成的处理'
                WHERE id IN (SELECT document_id FROM tasks WHERE status='queued')
                AND status IN ('parsing','analyzing','generating','reviewing')""")

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
            item['candidates'] = [dict(x, row=json.loads(x['row_json']), review=json.loads(x['review'])) for x in db.execute(
                'SELECT * FROM candidates WHERE document_id=? ORDER BY created', (doc_id,))]
            item['qa_count'] = db.execute('SELECT COUNT(*) FROM qas q JOIN candidates c ON q.id=c.id WHERE c.document_id=?', (doc_id,)).fetchone()[0]
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
                    (SELECT COUNT(*) FROM candidates c WHERE c.document_id=d.id) AS candidate_count,
                    (SELECT COUNT(*) FROM candidates c JOIN qas q ON q.id=c.id
                        WHERE c.document_id=d.id) AS qa_count
                    FROM documents d ORDER BY d.created DESC''').fetchall()
                items = {r['id']: dict(r, topics=[]) for r in rows}
                for r in db.execute('''SELECT dt.document_id,t.id,t.name,dt.keywords
                    FROM document_topics dt JOIN topics t ON t.id=dt.topic_id'''):
                    items[r['document_id']]['topics'].append(
                        {'id':r['id'], 'name':r['name'], 'keywords':json.loads(r['keywords'])})
            return list(items.values())
        with self.connect() as db:
            ids = [r[0] for r in db.execute('SELECT id FROM documents ORDER BY created DESC')]
        return [self.document(i) for i in ids]

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
        with self.connect() as db:
            previous = db.execute('SELECT keywords,unit_ids FROM document_topics WHERE document_id=? AND topic_id=?', (doc_id, topic_id)).fetchone()
            if previous:
                keywords = list(dict.fromkeys([*json.loads(previous['keywords']), *keywords]))
                unit_ids = list(dict.fromkeys([*json.loads(previous['unit_ids']), *unit_ids]))
            db.execute('INSERT OR REPLACE INTO document_topics VALUES(?,?,?,?)', (doc_id, topic_id, encode(keywords), encode(unit_ids)))

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
                             (' AND document_id=?' if document_id else '')+' ORDER BY rowid LIMIT 1',
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
        return results
