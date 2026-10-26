"""Small, disposable semantic index. No model imports in the web/PDF process."""
import contextlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import threading
import time

from .settings import ROOT, settings
from .store import digest, normalized, uid


def active_topics(store):
    return sorted((t for t in store.topics() if t['documents']), key=lambda t: t['id'])


def topic_text(topic):
    # Bounded, stable profiles; detailed keywords keep their document scope.
    return '\n'.join([topic['name'], topic['definition'][:400],
                      '；'.join(topic['aliases'])[:200]])


def catalog_version(topics, model_id):
    return digest([model_id, [(t['id'], topic_text(t)) for t in topics]])


@contextlib.contextmanager
def cache_connection(path):
    db = sqlite3.connect(path, timeout=2)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


_cache_init_lock = threading.Lock()


def init_cache(path):
    with _cache_init_lock, cache_connection(path) as db:
        if (db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='state'").fetchone()
                and db.execute("SELECT 1 FROM state WHERE key='qa_vectors_migrated'").fetchone()):
            return
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
          CREATE TABLE IF NOT EXISTS vectors(
            hash TEXT PRIMARY KEY, vector BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS qa_vectors(
            hash TEXT PRIMARY KEY, vector BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS rankings(
            key TEXT, catalog TEXT, scores BLOB NOT NULL,
            PRIMARY KEY(key,catalog));
          CREATE TABLE IF NOT EXISTS queries(
            key TEXT PRIMARY KEY, text TEXT NOT NULL, touched REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
          CREATE TABLE IF NOT EXISTS attempts(
            id INTEGER PRIMARY KEY, created REAL, seconds REAL,
            status TEXT, error_type TEXT);
        ''')
        # Preserve pre-upgrade QA caches before topic indexing can delete them.
        # Old rows have no owner metadata; copy once, without reading business data.
        db.execute('BEGIN IMMEDIATE')
        if not db.execute("SELECT 1 FROM state WHERE key='qa_vectors_migrated'").fetchone():
            db.execute('INSERT OR IGNORE INTO qa_vectors SELECT * FROM vectors')
            db.execute("INSERT INTO state VALUES('qa_vectors_migrated','1')")


def matching_topic_ids(store, topics, query):
    """Same scope as basic search, without reading documents/evidence/QA."""
    matched = {t['id'] for t in topics
               if any(query in normalized(x) for x in [t['name'], *t['aliases']])}
    with store.connect() as db:
        for row in db.execute('SELECT topic_id,keywords FROM document_topics'):
            if any(query in normalized(x) for x in json.loads(row['keywords'])):
                matched.add(row['topic_id'])
        for row in db.execute('SELECT topic_id,term FROM document_search_terms'):
            if query in normalized(row['term']):
                matched.add(row['topic_id'])
    return matched


class SemanticSearch:
    def __init__(self, store, config=None):
        self.store = store
        self.config = config if config is not None else settings().get('semantic_search', {})
        self.enabled = self.config.get('enabled', False)
        self.model_dir = (ROOT / self.config.get('model_dir', 'runtime/models/bge-small-zh-v1.5')).resolve()
        spec = json.loads((ROOT / 'config/embedding-model.json').read_text())
        self.model_id = digest([self.config.get('model_id', 'bge-small-zh-v1.5-onnx-int8-cls-v1'), spec])
        self.path = store.root / 'search.sqlite3'
        init_cache(self.path)
        self.coordinator = None

    def model_available(self):
        return all((self.model_dir / f).is_file() for f in ('model.onnx', 'tokenizer.json', 'manifest.json'))

    def recommend(self, text):
        query = normalized(text)
        topics = active_topics(self.store)
        response = {'recommendations': [], 'status': 'ready', 'message': '',
                    'indexed_topics': 0, 'total_topics': len(topics)}
        if not query or not topics:
            return response
        if not self.enabled:
            return dict(response, status='disabled', message='语义推荐未启用，当前展示基础匹配。')
        anchors = [t for t in topics if normalized(t['name']) == query]
        if not anchors:
            anchors = [t for t in topics if query in [normalized(a) for a in t['aliases']]]
        anchor = anchors[0] if len(anchors) == 1 else None
        excluded = matching_topic_ids(self.store, topics, query)
        if len(excluded) == len(topics):
            return dict(response, message='当前资料中没有其他可推荐的主题。')
        version = catalog_version(topics, self.model_id)
        key = 't:' + anchor['id'] if anchor else 'q:' + query
        with cache_connection(self.path) as db:
            response['indexed_topics'] = db.execute(
                "SELECT COUNT(*) FROM rankings WHERE catalog=? AND key LIKE 't:%'", (version,)).fetchone()[0]
            row = db.execute('SELECT scores FROM rankings WHERE key=? AND catalog=?', (key, version)).fetchone()
            error = db.execute("SELECT value FROM state WHERE key='error'").fetchone()
            if not row and not anchor:
                db.execute('INSERT INTO queries VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET touched=excluded.touched',
                           (key, query, time.time()))
                db.execute('DELETE FROM queries WHERE key NOT IN (SELECT key FROM queries ORDER BY touched DESC LIMIT 128)')
        if row and len(row['scores']) == len(topics) * 4:
            scores = struct.unpack('<' + 'f' * len(topics), row['scores'])
            ranked = sorted(zip(topics, scores), key=lambda x: (-x[1], x[0]['name'], x[0]['id']))
            threshold = self.config.get('min_similarity', 0.65)
            response['recommendations'] = [
                {'topic': t, 'reason': f'与“{anchor["name"]}”语义相近' if anchor else '与搜索内容语义相近'}
                for t, score in ranked if t['id'] not in excluded and math.isfinite(score) and score >= threshold
            ][:5]
            response['message'] = ('按语义相似程度排序；点击主题查看其关联资料。'
                                   if response['recommendations'] else '暂未找到足够相关的其他主题。')
            return response
        if not self.model_available():
            return dict(response, status='unavailable', message='本地语义模型尚未安装，当前展示基础匹配。')
        if error:
            return dict(response, status='error', message='语义推荐暂不可用，基础匹配仍可使用。')
        busy = self.coordinator.busy() if self.coordinator else False
        return dict(response, status='busy' if busy else 'pending',
                    message='正在优先处理文件，完成后自动更新语义推荐。' if busy else '正在准备语义推荐，完成后自动显示。')

    def next_job(self):
        if not self.enabled or not self.model_available():
            return None
        topics = active_topics(self.store)
        if not topics:
            return None
        version = catalog_version(topics, self.model_id)
        with cache_connection(self.path) as db:
            ready = db.execute("SELECT value FROM state WHERE key='catalog'").fetchone()
            queries = [dict(r) for r in db.execute('''SELECT q.key,q.text FROM queries q
                WHERE NOT EXISTS(SELECT 1 FROM rankings r WHERE r.key=q.key AND r.catalog=?)
                ORDER BY touched DESC LIMIT 8''', (version,))]
        if ready and ready['value'] == version and not queries:
            return None
        return {'cache': str(self.path), 'model_dir': str(self.model_dir), 'model_id': self.model_id,
                'catalog': version, 'topics': topics, 'queries': queries}


class SearchCoordinator:
    """One low-priority subprocess, canceled before any PDF worker begins."""
    def __init__(self, services):
        self.services = list(services)
        for service in self.services:
            service.coordinator = self
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.process = None
        self.thread = threading.Thread(target=self.loop, daemon=True, name='semantic-idle')
        self.retry_after = {}

    def busy(self):
        for service in self.services:
            with service.store.connect() as db:
                if db.execute("SELECT 1 FROM tasks WHERE status IN ('queued','running') LIMIT 1").fetchone():
                    return True
        return False

    def start(self):
        self.thread.start()

    def interrupt(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1)

    def close(self):
        self.stop.set()
        self.interrupt()
        if self.thread.is_alive():
            self.thread.join(timeout=3)

    def run_job(self, service, job):
        path = service.store.root / 'semantic-job.json'
        path.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        started = time.monotonic()
        canceled = False
        timed_out = False
        # Do not pass company credentials or user shell variables to the model process.
        env = {'PATH': os.defpath, 'PYTHONPATH': str(ROOT / 'src'),
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'}
        try:
            with self.lock:
                if self.stop.is_set() or self.busy():
                    return
                self.process = subprocess.Popen([sys.executable, '-m', 'qaproject.embedding', str(path)],
                                                env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                process = self.process
            while process.poll() is None:
                timed_out = time.monotonic() - started > 120
                if self.stop.wait(0.2) or self.busy() or timed_out:
                    canceled = True
                    self.interrupt()
                    break
            code = process.wait()
            canceled = (canceled or code < 0) and not timed_out
            error_type = 'EmbeddingTimeout' if timed_out else 'EmbeddingProcessError'
            with cache_connection(service.path) as db:
                db.execute('INSERT INTO attempts(created,seconds,status,error_type) VALUES(?,?,?,?)',
                           (time.time(), round(time.monotonic()-started, 3),
                            'interrupted' if canceled else 'completed' if code == 0 else 'failed',
                            '' if canceled or code == 0 else error_type))
                if code and not canceled:
                    db.execute("INSERT OR REPLACE INTO state VALUES('error',?)", (error_type,))
                    self.retry_after[str(service.path)] = time.monotonic() + 60
                elif code == 0:
                    db.execute("DELETE FROM state WHERE key='error'")
        finally:
            self.interrupt()
            with self.lock:
                self.process = None
            path.unlink(missing_ok=True)

    def loop(self):
        while not self.stop.wait(2):
            try:
                if self.busy():
                    continue
                for service in self.services:
                    if time.monotonic() < self.retry_after.get(str(service.path), 0):
                        continue
                    job = service.next_job()
                    if job:
                        self.run_job(service, job)
                        break
            except (OSError, sqlite3.Error, ValueError):
                # Search indexing cannot terminate the PDF worker or expose raw errors.
                self.stop.wait(3)


class DedupUnavailable(ValueError):
    """A failed check must never masquerade as an empty duplicate result."""

    def __init__(self, reason='incomplete'):
        self.reason = reason
        super().__init__('跨文档查重未完成：请检查本地向量模型及缓存目录，或重试继续补算；本次未放行。')


def vector_values(raw):
    """Validate cached/child vectors without importing the model runtime."""
    if not isinstance(raw, bytes) or len(raw) != 512 * 4:
        return None
    import numpy as np
    values = np.frombuffer(raw, dtype='<f4')
    if not np.isfinite(values).all() or abs(float(values @ values)-1) > .02:
        return None
    return values


def encode_texts(model_dir, model_id, cache, texts, timeout=None, check=None):
    """Read QA cache first; encode only missing texts in recoverable bounded jobs."""
    texts = list(dict.fromkeys(t for t in texts if t and t.strip()))
    if not texts:
        return {}
    init_cache(cache)
    result = {}
    with cache_connection(cache) as db:
        for text in texts:
            row = db.execute('SELECT vector FROM qa_vectors WHERE hash=?', (digest([model_id, text]),)).fetchone()
            values = vector_values(row['vector']) if row else None
            if values is not None:
                result[text] = values
    missing = [text for text in texts if text not in result]
    if missing and not all((Path(model_dir) / f).is_file() for f in ('model.onnx', 'tokenizer.json')):
        raise DedupUnavailable('model_missing')
    env = {'PATH': os.defpath, 'PYTHONPATH': str(ROOT / 'src'),
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'}
    for start in range(0, len(missing), 256):
        if check:
            check()
        batch = missing[start:start+256]
        job_path = Path(cache).parent / ('dedup-job-'+uid()+'.json')
        job = {'model_dir': str(model_dir), 'model_id': model_id, 'cache': str(cache), 'texts': batch}
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        try:
            done = subprocess.run([sys.executable, '-m', 'qaproject.embedding', str(job_path), 'encode-only'],
                                  env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  timeout=timeout if timeout is not None else max(30, len(batch)*.2))
            if done.returncode != 0:
                raise DedupUnavailable('encoder_failed')
            vectors = json.loads(job_path.read_text(encoding='utf-8')).get('vectors', {})
            if not isinstance(vectors, dict):
                raise DedupUnavailable('invalid_result')
            for text in batch:
                raw = vectors.get(digest([model_id, text]))
                values = vector_values(bytes.fromhex(raw)) if isinstance(raw, str) else None
                if values is None:
                    raise DedupUnavailable('invalid_vector')
                result[text] = values
        except DedupUnavailable:
            raise
        except subprocess.TimeoutExpired:
            raise DedupUnavailable('timeout') from None
        except OSError:
            raise DedupUnavailable('encoder_io') from None
        except (ValueError, AttributeError):
            raise DedupUnavailable('invalid_result') from None
        finally:
            job_path.unlink(missing_ok=True)
    if check:
        check()
    return result


def cross_document_duplicate(store, model_dir, model_id, cache, document_id, questions, threshold, check=None):
    """Check all other documents' approved Chinese questions; None means incomplete."""
    import numpy as np
    with store.connect() as db:
        rows = db.execute('''SELECT q.row_json FROM qas q JOIN candidates c ON c.id=q.id
            WHERE c.status='approved' AND c.document_id!=? AND json_extract(q.row_json,'$.language')='zh' ''',
            (document_id,)).fetchall()
    reference = list(dict.fromkeys(json.loads(r['row_json']).get('input','') for r in rows))
    reference = [q for q in reference if q and q.strip()]
    if not reference or not questions:
        return {}
    exact = set(reference).intersection(questions)
    result = {q:1.0 for q in exact}
    remaining = list(dict.fromkeys(q for q in questions if q not in exact))
    if not remaining:
        return result
    vectors = encode_texts(model_dir, model_id, cache, [*remaining, *reference], check=check)
    if vectors is None or any(q not in vectors for q in [*remaining, *reference]):
        return None
    matrix = np.asarray([vectors[q] for q in reference], dtype='<f4')
    for question in remaining:
        if check:
            check()
        scores = matrix @ np.asarray(vectors[question], dtype='<f4')
        best = float(scores.max()) if scores.size else 0.0
        if best >= threshold:
            result[question] = best
    return result
