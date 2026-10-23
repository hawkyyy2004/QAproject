"""Optional, idle-time directory catalogue. Listing never reads document bodies."""
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import threading
import time

from .persistence import file_hash
from .source_files import signature
from .store import digest, uid
from .settings import ROOT

FORMATS = {'.pdf','.png','.jpg','.jpeg','.webp','.txt','.md','.csv'}


class LibraryIndex:
    def __init__(self, store, root=None, busy=lambda:False):
        self.store = store
        source = Path(root).expanduser() if root else None
        self.root = ((source if source.is_absolute() else ROOT/source).resolve() if source else None)
        if self.root and self.root==Path(self.root.anchor):
            raise ValueError('资料库必须配置具体资料目录，不能扫描整个磁盘。')
        self.path = store.root/'library.sqlite3'
        self.busy = busy
        self.stop = threading.Event()
        self.requested = threading.Event()
        self.scanning = threading.Lock()
        self.thread = threading.Thread(target=self.loop,daemon=True,name='library-idle')
        self.running = False
        if self.root:
            with self.connect() as db:
                db.execute('PRAGMA journal_mode=WAL')
                db.executescript('''CREATE TABLE IF NOT EXISTS entries(
                    id TEXT PRIMARY KEY,root TEXT,relative_path TEXT,name TEXT,name_fold TEXT,size INTEGER,
                    modified REAL,version TEXT,seen TEXT,missing INTEGER DEFAULT 0,
                    document_id TEXT,registered_version TEXT,content_hash TEXT);
                    CREATE INDEX IF NOT EXISTS library_root_name ON entries(root,missing,name_fold,id);
                    CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT);''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path,timeout=2)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def status(self):
        if not self.root:
            return dict(enabled=False,scanning=False,pending=False,error='',updated=None)
        with self.connect() as db:
            state = dict(db.execute('SELECT key,value FROM state'))
        return dict(enabled=True,scanning=self.running,pending=self.requested.is_set(),
                    error=state.get('error',''),updated=state.get('updated'))

    def request_refresh(self):
        if not self.root:
            return self.status()
        self.requested.set()
        return self.status()

    def start(self):
        if self.root:
            self.requested.set()
            self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join()

    def checkpoint(self):
        while not self.stop.is_set() and self.busy():
            self.stop.wait(.5)
        return not self.stop.is_set()

    def scan(self):
        if not self.root or not self.scanning.acquire(blocking=False):
            return
        self.running = True
        generation = uid()
        try:
            if not self.root.is_dir():
                raise OSError('UnavailableLibrary')
            pending, seen = [], 0
            stack = [self.root]
            while stack:
                if not self.checkpoint():
                    return
                folder = stack.pop()
                with os.scandir(folder) as entries:
                    for entry in entries:
                        seen += 1
                        if seen%32==0 and not self.checkpoint():
                            return
                        # Do not follow symlinks or catalogue the application's
                        # own database, generated images and response caches.
                        if entry.is_symlink() or entry.name.startswith('.'):
                            continue
                        path = Path(entry.path)
                        if any(path.is_relative_to(base) for base in (self.store.root,
                            self.store.root.parent/'demo',self.store.root.parent/'models',self.store.root.parent/'backups')):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(path)
                        elif path.suffix.lower() in FORMATS and entry.is_file(follow_symlinks=False):
                            stat = entry.stat(follow_symlinks=False)
                            relative = path.relative_to(self.root).as_posix()
                            pending.append((digest([str(self.root),relative]),str(self.root),relative,path.name,
                                relative.casefold(),stat.st_size,stat.st_mtime,signature(path,stat),generation))
                        if len(pending)>=64:
                            self.flush(pending)
                            pending.clear()
            if not self.checkpoint():
                return
            self.flush(pending)
            with self.connect() as db:
                db.execute('UPDATE entries SET missing=1 WHERE root=? AND seen!=?',(str(self.root),generation))
                db.execute("INSERT OR REPLACE INTO state VALUES('updated',?)",(str(time.time()),))
                db.execute("DELETE FROM state WHERE key='error'")
        except OSError:
            with self.connect() as db:
                db.execute("INSERT OR REPLACE INTO state VALUES('error','资料目录暂不可用，请检查挂载后刷新。')")
        finally:
            self.running = False
            self.scanning.release()

    def flush(self, rows):
        if rows:
            with self.connect() as db:
                db.executemany('''INSERT INTO entries(id,root,relative_path,name,name_fold,size,modified,version,seen)
                    VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,
                    name_fold=excluded.name_fold,size=excluded.size,modified=excluded.modified,
                    version=excluded.version,seen=excluded.seen,missing=0''',rows)

    def loop(self):
        while not self.stop.wait(.5):
            if self.requested.is_set() and not self.busy():
                self.requested.clear()
                try:
                    self.scan()
                except (OSError,sqlite3.Error):
                    # A catalogue failure must not terminate file processing.
                    self.stop.wait(1)

    def list_files(self, query='', page=1, page_size=30):
        state = self.status()
        if not self.root:
            return dict(state,items=[],page=1,pages=1,total=0)
        args = [str(self.root)]
        where = 'root=? AND missing=0'
        if query.strip():
            escaped = query.strip().casefold().replace('\\','\\\\').replace('%','\\%').replace('_','\\_')
            where += " AND name_fold LIKE ? ESCAPE '\\'"
            args.append('%'+escaped+'%')
        with self.connect() as db:
            total = db.execute('SELECT COUNT(*) FROM entries WHERE '+where,args).fetchone()[0]
            pages = max(1,(total+page_size-1)//page_size)
            page = min(page,pages)
            rows = [dict(r) for r in db.execute('SELECT id,name,relative_path,size,modified,version,document_id,registered_version FROM entries WHERE '+where+
                ' ORDER BY name_fold,id LIMIT ? OFFSET ?',[*args,page_size,(page-1)*page_size])]
        ids = [r['document_id'] for r in rows if r['document_id']]
        states = {}
        if ids:
            with self.store.connect() as db:
                states = dict(db.execute('SELECT id,status FROM documents WHERE id IN ('+','.join('?' for _ in ids)+')',ids))
        for row in rows:
            row['status'] = ('updated' if row['document_id'] and row['registered_version']!=row['version']
                             else states.get(row['document_id'],'available'))
            row.pop('registered_version',None)
        return dict(state,items=rows,page=page,pages=pages,total=total)

    def import_entry(self, entry_id, version, max_bytes):
        if not self.root:
            raise ValueError('尚未配置服务器资料目录；本机上传仍可使用。')
        with self.connect() as db:
            row = db.execute('SELECT * FROM entries WHERE id=? AND root=? AND missing=0',(entry_id,str(self.root))).fetchone()
        if not row or row['version']!=version:
            raise ValueError('资料目录记录已更新，请刷新后重新选择。')
        path = self.root/row['relative_path']
        if path.is_symlink() or path.resolve()!=path or not path.is_file() or not path.is_relative_to(self.root):
            raise ValueError('原件位置已变化或目录不可用，请刷新资料库。')
        if path.suffix.lower() not in FORMATS or not 0<path.stat().st_size<=max_bytes:
            raise ValueError('原件为空、格式不支持或超出文件大小限制。')
        before = signature(path)
        if before!=version:
            raise ValueError('资料原件已更新，请刷新资料库后选择当前版本。')
        h = row['content_hash'] if row['registered_version']==before and row['content_hash'] else file_hash(path)
        if signature(path)!=before:
            raise ValueError('读取期间原件发生变化，请稍后重试。')
        doc_id,created = self.store.import_reference(path,h,str(self.root),before)
        with self.connect() as db:
            db.execute('UPDATE entries SET document_id=?,registered_version=?,content_hash=? WHERE id=?',
                       (doc_id,before,h,entry_id))
        return dict(id=doc_id,name=row['name'],created=created)
