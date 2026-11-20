"""Immutable Corpus snapshots with independent structured reading/quality stages.

Original QA evidence stays frozen. Corpus-only quality calls share execution limits;
raw originals, retained text spans, structure and review decisions remain traceable.
"""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import time
import zipfile

from .execution import WorkInterrupted
from .parsers import extract
from .persistence import file_hash
from .settings import settings
from .source_files import original_path, verify_reference
from .store import digest, encode, uid
from . import corpus_processing

KINDS = {'paper', 'web', 'image', 'data'}
PAPER_FIELDS = ['paper_id', 'title', 'pdf_path', 'md_path', 'images_dir', 'image_count']
WEB_FIELDS = ['entity_name', 'entity_type', 'entity_description', 'page_urls',
              'page_titles', 'full_text', 'full_text_length']


def init_schema(store):
    with store.connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS corpus_sources(
          document_id TEXT PRIMARY KEY REFERENCES documents(id), kind TEXT NOT NULL,
          title TEXT NOT NULL DEFAULT '', source_url TEXT NOT NULL DEFAULT '',
          source_label TEXT NOT NULL DEFAULT '', updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS corpus_versions(
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
          kind TEXT NOT NULL, status TEXT NOT NULL, title TEXT NOT NULL,
          source_url TEXT NOT NULL, source_label TEXT NOT NULL,
          units_json TEXT NOT NULL, manifest_json TEXT NOT NULL,
          content_hash TEXT NOT NULL, created REAL NOT NULL, version_hash TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS corpus_version_documents ON corpus_versions(document_id,created DESC,id);
        CREATE TABLE IF NOT EXISTS corpus_jobs(
          document_id TEXT PRIMARY KEY REFERENCES documents(id), task_id TEXT NOT NULL,
          status TEXT NOT NULL, progress TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
          version_id TEXT, updated REAL NOT NULL, generate_after INTEGER NOT NULL DEFAULT 0,
          metadata_json TEXT NOT NULL DEFAULT '{}', units_json TEXT NOT NULL DEFAULT '[]',
          pages INTEGER NOT NULL DEFAULT 0, source_hash TEXT NOT NULL DEFAULT '');
        CREATE TRIGGER IF NOT EXISTS corpus_versions_no_update BEFORE UPDATE ON corpus_versions
          BEGIN SELECT RAISE(ABORT,'Corpus versions are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS corpus_versions_no_delete BEFORE DELETE ON corpus_versions
          BEGIN SELECT RAISE(ABORT,'Corpus versions are immutable'); END;
        ''')


def _document(store, doc_id):
    # Read only the registered source and frozen evidence, not candidate bodies.
    with store.connect() as db:
        row = db.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
    if not row:
        raise ValueError('找不到文件')
    doc = dict(row)
    doc['units'] = json.loads(doc['units'])
    return doc


def _infer(doc):
    suffix = Path(doc['name']).suffix.lower()
    if suffix in {'.html', '.htm'}: return 'web'
    if suffix in {'.png', '.jpg', '.jpeg', '.webp'}: return 'image'
    if suffix in {'.csv', '.tsv', '.nc', '.nc4'}: return 'data'
    return 'paper'


def _source(store, doc):
    with store.connect() as db:
        row = db.execute('SELECT kind,title,source_url,source_label FROM corpus_sources WHERE document_id=?', (doc['id'],)).fetchone()
    return dict(row) if row else dict(kind=_infer(doc), title='', source_url='', source_label='')


def register_source(store, doc_id, kind, title='', source_url='', source_label=''):
    doc = _document(store, doc_id)
    previous = _source(store, doc)
    kind = kind or previous['kind']
    title = title or previous['title']
    source_url = source_url or previous['source_url']
    source_label = source_label or previous['source_label']
    if kind not in KINDS: raise ValueError('无效的 Corpus 类型')
    if any(not isinstance(v, str) for v in (title, source_url, source_label)):
        raise ValueError('来源元数据必须是文字')
    if source_url:
        from urllib.parse import urlsplit
        url = urlsplit(source_url)
        if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password:
            raise ValueError('来源网址必须为无凭据的 HTTP/HTTPS URL')
    with store.connect() as db:
        db.execute('''INSERT INTO corpus_sources VALUES(?,?,?,?,?,?) ON CONFLICT(document_id)
          DO UPDATE SET kind=excluded.kind,title=excluded.title,source_url=excluded.source_url,
          source_label=excluded.source_label,updated=excluded.updated''',
                   (doc_id, kind, title, source_url, source_label, time.time()))
    return dict(document_id=doc_id, kind=kind, title=title, source_url=source_url, source_label=source_label)


def enqueue(store, doc_id, kind='paper', *, title='', source_url='', source_label='', generate_after=False, metadata=None, reprocess=False,pdf_engine=None):
    doc = _document(store, doc_id)
    source = _source(store, doc)
    # Existing explicit web metadata wins over the historical paper default.
    if title or source_url or source_label or (kind and kind != source['kind'] and kind != 'paper'):
        selected_kind = source['kind'] if kind in {None, '', 'paper'} else kind
        register_source(store, doc_id, selected_kind, title or source['title'],
                        source_url or source['source_url'], source_label or source['source_label'])
        source = _source(store, doc)
    if kind not in KINDS | {None, ''}: raise ValueError('无效的 Corpus 类型')
    if not isinstance(metadata or {}, dict): raise ValueError('metadata 必须为对象')
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        job = db.execute('SELECT * FROM corpus_jobs WHERE document_id=?', (doc_id,)).fetchone()
        merged = dict(json.loads(job['metadata_json']) if job else {}, **(metadata or {}))
        if pdf_engine not in {None,'configured','native','mineru'}:raise ValueError('无效论文解析方式')
        if pdf_engine:merged['pdf_engine']=pdf_engine
        # Repair a requested legacy archive by creating a new immutable version.
        # Never rewrite the old archive or documents.units used by existing QA.
        if not reprocess and job and job['status']=='completed' and Path(doc['path']).suffix.lower()=='.pdf':
            prior=db.execute('SELECT units_json FROM corpus_versions WHERE id=?',(job['version_id'],)).fetchone()
            if prior and any(u.get('evidence_kind')=='pdf_page' and u.get('layout_engine')!='mineru' for u in json.loads(prior[0])):
                if corpus_processing.configured_engine(store,doc,settings(),merged.get('pdf_engine'))[0]=='mineru':
                    reprocess=True
        encoded = encode(merged)
        requested = bool(generate_after if reprocess else generate_after or (job and job['generate_after']))
        active = db.execute("SELECT id,status FROM tasks WHERE document_id=? AND initial_stage='corpus' AND status IN ('queued','running') ORDER BY rowid LIMIT 1", (doc_id,)).fetchone()
        if reprocess and active:raise ValueError('资料正在整理，请等待当前任务完成。')
        if reprocess:
            merged.pop('processing',None);merged['reprocess']=True;encoded=encode(merged)
            db.execute("UPDATE corpus_jobs SET version_id=NULL,units_json='[]',pages=0 WHERE document_id=?",(doc_id,))
        if active:
            task_id, status = active['id'], active['status']
        elif not reprocess and job and job['status'] == 'completed' and job['source_hash'] == digest(source):
            db.execute('UPDATE corpus_jobs SET generate_after=?,metadata_json=?,updated=? WHERE document_id=?',
                       (int(requested), encoded, time.time(), doc_id))
            return _job(dict(db.execute('SELECT * FROM corpus_jobs WHERE document_id=?', (doc_id,)).fetchone()))
        else:
            task_id, status = uid(), 'queued'
            db.execute("INSERT INTO tasks(id,document_id,status,stage,initial_stage) VALUES(?,?,'queued','corpus','corpus')", (task_id, doc_id))
            if job:
                db.execute("UPDATE task_dependencies SET prerequisite_id=? WHERE prerequisite_id=? AND task_id IN (SELECT id FROM tasks WHERE status='queued')",(task_id,job['task_id']))
        db.execute('''INSERT INTO corpus_jobs(document_id,task_id,status,updated,generate_after,metadata_json,source_hash)
          VALUES(?,?,?,?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET task_id=excluded.task_id,
          status=excluded.status,error='',updated=excluded.updated,generate_after=excluded.generate_after,
          metadata_json=excluded.metadata_json,source_hash=excluded.source_hash''',
                   (doc_id, task_id, status, time.time(), int(requested), encoded, digest(source)))
        return _job(dict(db.execute('SELECT * FROM corpus_jobs WHERE document_id=?', (doc_id,)).fetchone()))


def _lease(db, task, worker):
    if not db.execute("SELECT 1 FROM tasks WHERE id=? AND document_id=? AND worker=? AND status='running' AND lease_until>?",
                      (task['id'], task['document_id'], worker, time.time())).fetchone():
        raise ValueError('任务租约已失效')


def _original(store, doc):
    verify_reference(doc, store.root)
    path = original_path(doc, store.root)
    if file_hash(path) != doc['content_hash']:
        raise ValueError('原始资料内容已变更')
    return path


def _safe(root, relative):
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative:
        raise ValueError('Corpus 文件路径无效')
    path = (root / p).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Corpus 文件路径越界')
    return path


def _folder(store, version_id):
    if not re.fullmatch('[a-f0-9]{32}', version_id): raise ValueError('Corpus 版本 ID 无效')
    return _safe(store.root, 'corpus/versions/' + version_id)


def _units_valid(store, doc_id, units):
    if not isinstance(units, list) or not units: raise ValueError('没有可用的冻结证据')
    seen = set()
    with store.connect() as db:
        for u in units:
            if not isinstance(u, dict) or not {'id','text','page','seg','image','asset_id','locator','hash'}.issubset(u):
                raise ValueError('冻结证据结构不完整')
            if u['id'] in seen or digest({k:v for k,v in u.items() if k != 'hash'}) != u['hash']:
                raise ValueError('冻结证据 hash 不一致')
            seen.add(u['id'])
            if u['image']:
                asset = db.execute('SELECT * FROM assets WHERE id=? AND document_id=?', (u['asset_id'], doc_id)).fetchone()
                image = Path(u['image']).resolve()
                if (not asset or image != Path(asset['path']).resolve() or not image.is_relative_to(store.root.resolve())
                        or not image.is_file() or file_hash(image) != asset['hash']):
                    raise ValueError('冻结图片资产失效或已篡改')
            elif u['asset_id']:
                raise ValueError('冻结图片引用不完整')


def _relocated_images(store, doc_id, units):
    """New snapshot may reuse identical moved bytes, never rewrite old evidence."""
    output=[]
    for raw in units:
        unit=dict(raw)
        if digest({k:v for k,v in unit.items() if k!='hash'})!=unit.get('hash'):
            raise ValueError('冻结证据 hash 不一致')
        if unit.get('image') and not Path(unit['image']).is_file():
            with store.connect() as db:
                asset=db.execute('SELECT * FROM assets WHERE id=? AND document_id=?',
                                 (unit['asset_id'],doc_id)).fetchone()
            if not asset or asset['path']!=unit['image']:
                raise ValueError('冻结图片资产失效或已篡改')
            old=PureWindowsPath(asset['path']) if '\\' in asset['path'] else Path(asset['path'])
            parts=old.parts; tail=None
            for i in range(len(parts)-2):
                if parts[i:i+3]==('assets','images',doc_id):tail=parts[i:]
            if tail:
                actual=store.root.joinpath(*tail).resolve()
                if (actual.is_relative_to(store.root/'assets/images'/doc_id) and actual.is_file()
                        and file_hash(actual)==asset['hash']):
                    new_id=store.add_asset(doc_id,actual,unit['page'],'Corpus 同内容迁移图片')
                    unit.update(image=str(actual),asset_id=new_id)
                    unit.pop('hash');unit['hash']=digest(unit)
                    store.event(doc_id,'corpus_relocated_asset',dict(previous_asset_id=raw['asset_id'],
                                asset_id=new_id,content_hash_verified=True))
        output.append(unit)
    return output


def _clean(units, margin_lines=None):
    margins = {}
    pattern = re.compile(r'(?:Page\s+\d+\s+of\s+\d+|第\s*\d+\s*页\s*共\s*\d+\s*页)', re.I)
    for u in units:
        if not u['page'] or u['image']: continue
        lines = u['text'].splitlines(keepends=True)
        for line in lines[:1] + lines[-1:]:
            if pattern.fullmatch(line.strip()):
                key = re.sub(r'\d+', '#', line.strip().lower())
                margins.setdefault(key, set()).add(u['page'])
    output, mapping = [], []
    for raw in units:
        u = dict(raw); text = raw['text']; spans = []; offset = 0
        lines = text.splitlines(keepends=True)
        for i, line in enumerate(lines):
            key = re.sub(r'\d+', '#', line.strip().lower())
            remove = (not u['image'] and u['page'] and i in {0,len(lines)-1}
                      and pattern.fullmatch(line.strip()) and len(margins.get(key, ())) >= 3)
            remove=remove or (not u['image'] and line.strip() in (margin_lines or {}).get(u['page'],set()))
            if not remove: spans.append([offset, offset+len(line)])
            offset += len(line)
        u['text'] = ''.join(text[a:b] for a,b in spans)
        mapping.append(dict(unit_id=u['id'], raw_hash=raw['hash'], retained_spans=spans,
                            rule='verified-page-margins-and-repeated-counters'))
        output.append(u)
    return output, mapping


def _csv(row):
    f = io.StringIO(newline=''); writer = csv.DictWriter(f, fieldnames=PAPER_FIELDS)
    writer.writeheader(); writer.writerow(row)
    return f.getvalue().encode('utf-8-sig')


def _version_hash(row):
    return digest({k: row[k] for k in ('id','document_id','kind','status','title','source_url','source_label',
                                      'units_json','manifest_json','content_hash','created')})


def _publish(store, doc, source, raw, pages, heartbeat, processing=None):
    _units_valid(store, doc['id'], raw)
    processing=processing or {}
    margin_lines={p['page']:{line['text'] for line in p['lines'] if line.get('margin')}
                  for p in processing.get('structure',{}).get('pages',[])}
    cleaned, mapping = _clean(raw,margin_lines)
    cleaned=[u for u in cleaned if u['image'] or u['text'].strip()]
    version_id = uid(); folder = _folder(store, version_id)
    folder.mkdir(parents=True, exist_ok=False)
    units = []; images = {}; assets = []
    # The version title must stay stable across re-processing and parse engines.
    # First-page "title-level" inference once turned a thesis cover label into
    # the title, so the registered source title or the filename always wins.
    title=source['title'] or doc['name']
    for u in cleaned:
        heartbeat(); u = dict(u)
        if u['image']:
            old = u['image']
            if old not in images:
                name = 'images/' + str(len(images)+1).zfill(4) + Path(old).suffix.lower()
                target = _safe(folder, name); target.parent.mkdir(exist_ok=True)
                shutil.copyfile(old, target)
                images[old] = (name, target, uid())
                assets.append((images[old][2], doc['id'], str(target), file_hash(target), u['page'], 'Corpus 冻结图片'))
            name, target, asset_id = images[old]
            u.update(image=str(target), asset_id=asset_id)
        if source['kind'] == 'web':
            u['page_urls'] = source['source_url']
            u['page_titles'] = title
        u.pop('hash'); u['hash'] = digest(u); units.append(u)
    (folder/'document.md').write_text(corpus_processing.markdown(title,units,processing.get('structure')),encoding='utf-8')
    (folder/'processing.json').write_text(encode(processing),encoding='utf-8')
    (folder/'evidence.json').write_text(encode(dict(raw_units=raw, cleaning=mapping)), encoding='utf-8')
    image_count = len(images)
    index = None
    path = _original(store, doc)
    with path.open('rb') as original:
        is_pdf = path.suffix.lower() == '.pdf' and original.read(5) == b'%PDF-'
    if source['kind'] == 'paper' and is_pdf:
        index = dict(paper_id=doc['id'], title=title, pdf_path=doc['path'], md_path='document.md',
                     images_dir='images' if image_count else '', image_count=image_count)
        (folder/'papers.csv').write_bytes(_csv(index))
    files = {p.relative_to(folder).as_posix():file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    manifest = dict(format=1, files=files, source=dict(path=doc['path'], content_hash=doc['content_hash'],
                    archive_path='originals/source' + Path(doc['name']).suffix.lower()),
                    source_metadata=source, pages=pages, unit_count=len(units), image_count=image_count,
                    cleaning_rule='verified-page-margins-and-repeated-counters', index=index,
                    internal_audit_files=['evidence.json'],
                    processing_revision=corpus_processing.REVISION,processing=processing,
                    full_text_semantics='当前冻结来源全部非图像正文，按顺序去除完全相同段落后以空行连接',
                    validation='资料已归档：结构/hash校验通过；不表示全文科学内容审核通过')
    (folder/'manifest.json').write_text(encode(manifest), encoding='utf-8')
    row = dict(id=version_id, document_id=doc['id'], kind=source['kind'], status='ready', title=title,
               source_url=source['source_url'], source_label=source['source_label'], units_json=encode(units),
               manifest_json=encode(manifest), content_hash=doc['content_hash'], created=time.time())
    row['version_hash'] = _version_hash(row)
    return row, assets


def process(worker, task):
    store = worker.store; started = time.perf_counter(); stage = 'corpus_preparing'
    def heartbeat():
        worker.limits.check(worker.stop)
        store.task_update(task['id'], worker.name, stage)
    try:
        heartbeat()
        doc = _document(store, task['document_id']); source = _source(store, doc)
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE'); _lease(db, task, worker.name)
            db.execute('''INSERT INTO corpus_jobs(document_id,task_id,status,updated,source_hash)
              VALUES(?,?,'running',?,?) ON CONFLICT(document_id) DO UPDATE SET
              task_id=excluded.task_id,status='running',error='',updated=excluded.updated''',
                       (doc['id'], task['id'], time.time(), digest(source)))
            job = dict(db.execute('SELECT * FROM corpus_jobs WHERE document_id=?', (doc['id'],)).fetchone())
        path = _original(store, doc)
        # A committed version is the recovery checkpoint after interrupted dispatch.
        if job['version_id']:
            prior = verify(store, job['version_id'])
            if prior['manifest']['source_metadata'] == source:
                row = prior
            else: row = None
        else: row = None
        if row and path.suffix.lower()=='.pdf' and any(
                u.get('evidence_kind')=='pdf_page' and u.get('layout_engine')!='mineru' for u in row['units']):
            requested=json.loads(job['metadata_json']).get('pdf_engine')
            if corpus_processing.configured_engine(store,doc,settings(),requested)[0]=='mineru':row=None
        if row is None:
            metadata=json.loads(job['metadata_json']);processing=metadata.get('processing',{})
            engine='native'
            if path.suffix.lower()=='.pdf':
                if processing.get('evidence_policy')==corpus_processing.MINERU_EVIDENCE and processing.get('parser_selection'):
                    engine=processing['parser_selection']['resolved']
                else:
                    engine,note=corpus_processing.configured_engine(store,doc,settings(),metadata.get('pdf_engine'))
                    processing['parser_selection']=dict(requested=metadata.get('pdf_engine') or 'configured',resolved=engine,note=note)
            checkpoint=json.loads(job['units_json'])
            raw = checkpoint or doc['units']; pages = job['pages'] if checkpoint else doc['pages']
            if metadata.get('reprocess'):raw=json.loads(job['units_json']);pages=job['pages']
            if engine=='mineru' and processing.get('evidence_policy')!=corpus_processing.MINERU_EVIDENCE:
                if checkpoint:
                    metadata.setdefault('superseded_preparations',[]).append(dict(
                        units=checkpoint,pages=job['pages'],processing=processing,
                        reason='恢复原QA的MinerU独立插图证据，旧准备结果仅保留审计'))
                # A checkpoint from the regressed page-shot path must not be
                # republished as MinerU figure evidence. Historical QA is intact.
                raw=[];pages=0
                processing={k:v for k,v in processing.items() if k=='parser_selection'}
            if source['kind']=='web' and path.suffix.lower() in {'.html','.htm'} and not json.loads(job['units_json']):
                from .corpus_web import prepare
                raw,processing=prepare(worker,doc,source,path,heartbeat,metadata.get('web_charset'));pages=0
            if not raw and engine!='mineru':
                with store.connect() as db:
                    knowledge = db.execute('SELECT units FROM knowledge_jobs WHERE document_id=?', (doc['id'],)).fetchone()
                raw = json.loads(knowledge[0]) if knowledge else []
                if raw:
                    pages = doc['pages']
            if not raw:
                stage='corpus_layout'
                config=settings()
                layout_mode='manifest' if config.get('pdf_layout')=='manifest' and metadata.get('pdf_engine')!='mineru' else 'auto'
                config = dict(config, pdf_layout=layout_mode if engine=='mineru' else 'off',
                              asset_namespace='corpus-'+uid(),require_mineru=engine=='mineru',corpus_preparation=True)
                (raw, pages), wait, seconds = worker.limits.parse(
                    lambda: extract(store, dict(doc, path=str(path)), config, progress=lambda *_:heartbeat(),
                        **({'structure_sink':lambda value:processing.update(structure=value)} if path.suffix.lower()=='.pdf' and engine=='native' else {})),
                    worker.stop, heartbeat)
                if engine=='mineru':
                    processing['evidence_policy']=corpus_processing.MINERU_EVIDENCE;metadata['processing']=processing
                    with store.connect() as db:
                        db.execute('BEGIN IMMEDIATE');_lease(db,task,worker.name)
                        db.execute('UPDATE corpus_jobs SET units_json=?,pages=?,metadata_json=? WHERE document_id=?',(encode(raw),pages,encode(metadata),doc['id']))
            raw=_relocated_images(store,doc['id'],raw)
            if raw:_units_valid(store, doc['id'], raw)
            if path.suffix.lower()=='.pdf' and not processing.get('structure'):
                stage='corpus_structure'
                structure,_,_=worker.limits.parse(lambda:corpus_processing.mineru_structure(store,dict(doc,path=str(path)),heartbeat) if engine=='mineru' else corpus_processing.pdf_structure(path,heartbeat),worker.stop,heartbeat)
                if engine=='mineru':
                    expected=set(range(1,pages+1));actual={p['page'] for p in structure['pages']}
                    if expected!=actual:raise ValueError('MinerU页面不完整，未截断发布。')
                processing['structure']=structure
            # Freeze preparation before optional model quality checks. A provider
            # failure must not force another PDF/image preparation on retry.
            metadata['processing']=processing
            with store.connect() as db:
                db.execute('BEGIN IMMEDIATE');_lease(db,task,worker.name)
                db.execute('UPDATE corpus_jobs SET units_json=?,pages=?,metadata_json=? WHERE document_id=?',
                           (encode(raw),pages,encode(metadata),doc['id']))
            if not processing.get('quality_complete'):
                stage='corpus_quality'
                from .corpus_quality import refine
                raw,decisions=refine(worker,doc,raw,heartbeat)
                processing['semantic_dedup']=decisions
                processing['quality_complete']=True
            metadata['processing']=processing
            with store.connect() as db:
                db.execute('BEGIN IMMEDIATE'); _lease(db, task, worker.name)
                db.execute('UPDATE corpus_jobs SET units_json=?,pages=?,progress=?,metadata_json=?,updated=? WHERE document_id=?',
                           (encode(raw), pages, encode({'units':len(raw),'pages':pages}),encode(metadata), time.time(), doc['id']))
            stage = 'corpus_archiving'; heartbeat()
            row, assets = _publish(store, doc, source, raw, pages, heartbeat,processing)
            _original(store, doc); _units_valid(store, doc['id'], raw)
            with store.connect() as db:
                db.execute('BEGIN IMMEDIATE'); _lease(db, task, worker.name)
                for asset in assets: db.execute('INSERT INTO assets VALUES(?,?,?,?,?,?)', asset)
                db.execute('INSERT INTO corpus_versions VALUES(:id,:document_id,:kind,:status,:title,:source_url,:source_label,:units_json,:manifest_json,:content_hash,:created,:version_hash)', row)
                db.execute('UPDATE corpus_jobs SET version_id=? WHERE document_id=?', (row['id'], doc['id']))
        verify(store, row['id'])
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE'); _lease(db, task, worker.name)
            db.execute("UPDATE corpus_jobs SET status='completed',progress=?,error='',source_hash=?,updated=? WHERE document_id=?",
                       (encode({'units':len(json.loads(row['units_json'])) if 'units_json' in row else len(row['units']), 'percent':100}), digest(source), time.time(), doc['id']))
            db.execute("UPDATE tasks SET status='succeeded',stage='corpus_completed',lease_until=0 WHERE id=?", (task['id'],))
        store.event(doc['id'], 'corpus_complete', dict(stage=stage, seconds=round(time.perf_counter()-started, 3), finish_reason='complete'))
        return dict(id=row['id'], version_id=row['id'], document_id=doc['id'], status='ready', generate_after=bool(job['generate_after']))
    except Exception as error:
        try:
            with store.connect() as db:
                db.execute('BEGIN IMMEDIATE'); _lease(db, task, worker.name)
                interrupted = isinstance(error, WorkInterrupted)
                message = str(error) if type(error) is ValueError else 'Corpus 处理失败，已保留恢复检查点。'
                db.execute("UPDATE corpus_jobs SET status=?,error=?,updated=? WHERE document_id=? AND task_id=?",
                           ('queued' if interrupted else 'failed', message, time.time(), task['document_id'], task['id']))
                db.execute("UPDATE tasks SET status=?,stage='corpus',worker=NULL,lease_until=0 WHERE id=?", ('queued' if interrupted else 'failed', task['id']))
                if not interrupted:
                    db.execute("""UPDATE documents SET progress=? WHERE id=? AND EXISTS(
                      SELECT 1 FROM task_dependencies dep JOIN tasks child ON child.id=dep.task_id
                      WHERE dep.prerequisite_id=? AND child.status='queued')""",('资料整理失败：'+message.rstrip('。')+'；后续任务等待重试',task['document_id'],task['id']))
            store.event(task['document_id'], 'corpus_failed', dict(stage=stage, error_type=type(error).__name__,
                        seconds=round(time.perf_counter()-started,3), finish_reason='interrupted' if interrupted else 'incomplete'))
        except ValueError: pass
        return None


def _job(row):
    row.pop('units_json', None)
    row['generate_after'] = bool(row['generate_after'])
    row['metadata'] = json.loads(row.pop('metadata_json'))
    return row


def _pagination(page, page_size):
    return max(1,int(page)), max(1,min(100,int(page_size)))


def jobs(store, q='', page=1, page_size=20):
    page, page_size = _pagination(page,page_size)
    where = ' WHERE instr(lower(d.name),lower(?))>0'; args = (q,)
    with store.connect() as db:
        total = db.execute('SELECT COUNT(*) FROM corpus_jobs j JOIN documents d ON j.document_id=d.id'+where, args).fetchone()[0]
        rows = db.execute('''SELECT j.document_id,j.task_id,j.status,j.progress,j.error,j.version_id,j.updated,
          j.generate_after,j.metadata_json,d.name FROM corpus_jobs j JOIN documents d ON j.document_id=d.id'''+where+
          ' ORDER BY j.updated DESC,j.document_id LIMIT ? OFFSET ?', (*args,page_size,(page-1)*page_size)).fetchall()
    return dict(items=[_job(dict(r)) for r in rows], total=total, page=page, pages=max(1,(total+page_size-1)//page_size), page_size=page_size)


def listing(store, q='', kind='', page=1, page_size=20):
    page,page_size = _pagination(page,page_size)
    if kind and kind not in KINDS: raise ValueError('无效的 Corpus 类型')
    where = ' WHERE (instr(lower(v.title),lower(?))>0 OR instr(lower(v.source_label),lower(?))>0)'
    args = [q,q]
    if kind: where += ' AND v.kind=?'; args.append(kind)
    with store.connect() as db:
        total = db.execute('SELECT COUNT(*) FROM corpus_versions v'+where,args).fetchone()[0]
        rows = db.execute('''SELECT v.id,v.document_id,v.kind,v.status,v.title,v.source_url,v.source_label,
          v.content_hash,v.created,v.version_hash,d.path AS source_path,d.name,
          json_extract(v.manifest_json,'$.unit_count') AS unit_count,
          json_extract(v.manifest_json,'$.image_count') AS image_count FROM corpus_versions v
          JOIN documents d ON d.id=v.document_id'''+where+' ORDER BY v.created DESC,v.id LIMIT ? OFFSET ?',
                          (*args,page_size,(page-1)*page_size)).fetchall()
    return dict(items=[dict(r) for r in rows], total=total, page=page, pages=max(1,(total+page_size-1)//page_size), page_size=page_size)


def detail(store, version_id):
    folder = _folder(store, version_id)
    with store.connect() as db:
        record = db.execute('SELECT * FROM corpus_versions WHERE id=?', (version_id,)).fetchone()
    if not record: raise ValueError('找不到 Corpus 版本')
    row = dict(record)
    if row['version_hash'] != _version_hash(row): raise ValueError('Corpus 版本 hash 不一致')
    units = json.loads(row.pop('units_json')); manifest = json.loads(row.pop('manifest_json'))
    if not (folder/'manifest.json').is_file():
        raise ValueError('Corpus manifest 文件失效')
    if encode(manifest) != (folder/'manifest.json').read_text(encoding='utf-8'):
        raise ValueError('Corpus manifest 已篡改')
    for rel, expected in manifest['files'].items():
        path = _safe(folder, rel)
        if not path.is_file() or file_hash(path) != expected: raise ValueError('Corpus 归档文件失效或已篡改')
    _units_valid(store, row['document_id'], units)
    if any(u['image'] and not Path(u['image']).resolve().is_relative_to(folder) for u in units):
        raise ValueError('Corpus 图像不在版本归档内')
    doc = _document(store, row['document_id']); actual = ''; error = ''
    if row['content_hash'] != doc['content_hash'] or manifest['source']['path'] != doc['path']:
        raise ValueError('Corpus 来源登记不一致')
    try: actual = str(_original(store, doc))
    except ValueError as failure: error = str(failure)
    row.update(units=units, manifest=manifest, source_path=manifest['source']['path'], actual_source_path=actual,
               source_valid=not error, source_error=error, markdown=(folder/'document.md').read_text(encoding='utf-8'))
    return row


def verify(store, version_id):
    row = detail(store, version_id)
    if not row['source_valid']: raise ValueError(row['source_error'])
    return row


def prepared_units(store, doc_id):
    with store.connect() as db:
        row = db.execute("SELECT id FROM corpus_versions WHERE document_id=? AND status='ready' ORDER BY created DESC,id DESC LIMIT 1", (doc_id,)).fetchone()
    if not row: return None
    version = verify(store, row['id'])
    return version['units'], version['manifest']['pages']


def _web_export(store, version):
    """Snapshot only approved facts bound to this version's exact evidence."""
    unit_map = {u['id']:u for u in version['units']}
    with store.connect() as db:
        records = db.execute("""SELECT * FROM knowledge_entries WHERE document_id=?
          AND status='approved' ORDER BY created,id""", (version['document_id'],)).fetchall()
        job = db.execute('SELECT status FROM knowledge_jobs WHERE document_id=?',
                         (version['document_id'],)).fetchone()
    groups = {}; bindings = []
    for record in records:
        fact = dict(record); unit = unit_map.get(fact['unit_id'])
        if not unit or unit['hash'] != fact['evidence_hash']:
            continue
        row = json.loads(fact['row_json']); review = json.loads(fact['review'])
        checks = ('supported','scope_correct','type_correct','topic_correct','language_correct')
        if (digest(row) != fact['row_hash'] or review.get('row_hash') != fact['row_hash']
                or review.get('evidence_hash') != unit['hash'] or not all(review.get(k) is True for k in checks)
                or not fact['quote'] or fact['quote'] not in unit['text']
                or row.get('full_text') != unit['text']):
            raise ValueError('网页知识事实或审核绑定已失效')
        key = (row['entity_name'].strip().translate(str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ',
               'abcdefghijklmnopqrstuvwxyz')), row['entity_type'])
        group = groups.setdefault(key, dict(entity_name=row['entity_name'], entity_type=row['entity_type'], descriptions=[],facts=[]))
        group['facts'].append(row)
        if row['entity_description'] not in group['descriptions']:
            group['descriptions'].append(row['entity_description'])
        bindings.append(dict(id=fact['id'], row_hash=fact['row_hash'], evidence_hash=unit['hash']))
    output = io.StringIO(newline=''); writer = csv.DictWriter(output, fieldnames=WEB_FIELDS)
    writer.writeheader()
    for group in groups.values():
        writer.writerow(corpus_processing.source_row(group['facts'],version['units'],version['source_url'],version['title']))
    warnings = []
    if not job or job['status'] != 'completed':
        warnings.append('网页知识任务尚未完成；只导出当前已审核且证据匹配的事实，不代表知识整理完成。')
    if not bindings:
        warnings.append('本版本暂无证据 hash 与审核绑定均匹配的知识事实；七字段 CSV 仅含表头。')
    return output.getvalue().encode('utf-8'), dict(facts=bindings, entities=len(groups), warnings=warnings,
        full_text_semantics='统一来源全文：全部冻结非图像正文，按顺序去除完全相同段落后以空行连接；非官方未公开算法的复刻。')


def export_bundle(store, version_id):
    version = verify(store, version_id); folder = _folder(store, version_id)
    manifest = json.loads(encode(version['manifest']))
    source = manifest['source']; original = Path(version['actual_source_path'])
    files = {}
    for rel, expected in manifest['files'].items():
        data = _safe(folder,rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected: raise ValueError('导出期间 Corpus 文件变化')
        files[rel] = data
    files[source['archive_path']] = original.read_bytes()
    if hashlib.sha256(files[source['archive_path']]).hexdigest() != version['content_hash']:
        raise ValueError('导出期间原件已变更')
    if manifest['index']:
        index = dict(manifest['index'], pdf_path=source['archive_path'])
        files['papers.csv'] = _csv(index)
        manifest['index'] = index
    if version['kind'] == 'web':
        data, knowledge = _web_export(store, version)
        files['Web/wiki_updated_full.csv'] = data
        manifest['web_knowledge'] = knowledge
        manifest['warnings'] = knowledge['warnings']
    portable_units = []
    for unit in version['units']:
        unit = dict(unit)
        if unit['image']:
            unit['image'] = Path(unit['image']).relative_to(folder).as_posix()
        unit.pop('hash'); unit['hash'] = digest(unit)
        portable_units.append(unit)
    files['units.json'] = encode(portable_units).encode('utf-8')
    from .corpus_annotations import reference_files
    files.update(reference_files(store,version))
    manifest['archive_units_hash'] = digest(portable_units)
    manifest['version_id'] = version_id
    manifest['version_hash'] = version['version_hash']
    manifest['files'] = {rel:hashlib.sha256(data).hexdigest() for rel,data in files.items()}
    files['manifest.json'] = encode(manifest).encode('utf-8')
    output = io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for rel,data in files.items():
            _safe(folder, rel)
            archive.writestr(rel,data)
    return output.getvalue()


def export_csv(store, version_id):
    version=verify(store,version_id)
    if version['kind']=='web':return _web_export(store,version)[0],'wiki_updated_full.csv'
    index=version['manifest']['index']
    if not index:raise ValueError('此类成果没有官方对应的CSV索引，请下载资料包中的图片与标注。')
    name=_document(store,version['document_id'])['name']
    base='Paper/CleanedData/ProcessedData/'+version['document_id']+'/'
    index=dict(index,pdf_path='Paper/Pdf/'+name,md_path=base+'document.md',
               images_dir=base+'images' if index['image_count'] else '')
    return _csv(index),'papers.csv'


def export_official_bundle(store, version_id):
    """Public dataset layout; internal snapshot/audit export remains separate."""
    version=verify(store,version_id)
    if version['kind'] not in {'paper','web'} or not version['manifest']['index'] and version['kind']=='paper':
        return export_bundle(store,version_id)
    folder=_folder(store,version_id);files={}
    def checked_bytes(path,expected):
        content=Path(path).read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError('导出期间资料内容变化，未交付不一致成果。')
        return content
    data,name=export_csv(store,version_id)
    if version['kind']=='web':
        files['Web/'+name]=data
        files['Web/originals/'+version['document_id']+Path(version['actual_source_path']).suffix]=checked_bytes(version['actual_source_path'],version['content_hash'])
        for rel in version['manifest']['files']:
            if rel.startswith('images/'):
                files['Web/'+rel]=checked_bytes(_safe(folder,rel),version['manifest']['files'][rel])
    else:
        files['Paper/CleanedData/'+name]=data
        base='Paper/CleanedData/ProcessedData/'+version['document_id']+'/'
        for rel in version['manifest']['files']:
            if rel=='document.md' or rel.startswith('images/'):
                files[base+rel]=checked_bytes(_safe(folder,rel),version['manifest']['files'][rel])
        files['Paper/Pdf/'+_document(store,version['document_id'])['name']]=checked_bytes(version['actual_source_path'],version['content_hash'])
    from .instructions import list_annotations
    if list_annotations(store,version_id):
        from .corpus_annotations import reference_files
        prefix='Web/images' if version['kind']=='web' else base+'images'
        files.update(reference_files(store,version,prefix))
    verify(store,version_id)
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path,content in files.items():archive.writestr(path,content)
    return output.getvalue()
