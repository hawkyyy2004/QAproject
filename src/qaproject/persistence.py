"""Exclusive runtime ownership and offline, verified data snapshots."""
import hashlib
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid


class RuntimeLease:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.file = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = (self.root/'.runtime.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0)
                self.file.write(b'0')
                self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            self.file = None
            raise ValueError('该数据目录正在使用，请先停止原服务；不能多开程序共用同一数据目录。') from None
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()
            self.file = None


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def backup_runtime(runtime, destination):
    runtime, destination = Path(runtime).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('备份目标已存在，请选择新的目录。')
    if destination == runtime or runtime.is_relative_to(destination):
        raise ValueError('备份目标不能包含运行数据目录。')
    staging = destination.parent/('.snapshot-'+uuid.uuid4().hex)
    with RuntimeLease(runtime):
        try:
            if any(destination.is_relative_to(runtime/name) for name in ('live','demo')):
                raise ValueError('备份目标不能放在被备份的工作区内。')
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.mkdir()
            files = []
            workspaces = [runtime/name for name in ('live','demo') if (runtime/name/'state.sqlite3').is_file()]
            if not workspaces:
                raise ValueError('数据目录中没有可备份的任务库。')
            for workspace in workspaces:
                if workspace.is_symlink():
                    raise ValueError('运行工作区不能是符号链接。')
                for path in workspace.rglob('*'):
                    if path.is_symlink():
                        raise ValueError('运行数据含符号链接，未创建不完整备份。')
                    if not path.is_file() or path.name.endswith(('-wal','-shm')):
                        continue
                    relative = path.relative_to(runtime)
                    target = staging/relative
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if path.suffix == '.sqlite3':
                        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as source:
                            with closing(sqlite3.connect(target)) as output:
                                source.backup(output)
                                if output.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
                                    raise ValueError('备份数据库校验失败。')
                    else:
                        shutil.copy2(path,target)
                    files.append(dict(path=relative.as_posix(),bytes=target.stat().st_size,sha256=file_hash(target)))
            manifest = dict(schema_version=1,runtime=str(runtime),created=time.time(),files=files,
                originals='External library originals remain in their configured source directory; back them up separately.')
            (staging/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return dict(path=str(destination),files=len(files))


def restore_runtime(snapshot, runtime):
    snapshot, runtime = Path(snapshot).resolve(), Path(runtime).resolve()
    manifest = json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version')!=1 or manifest.get('runtime')!=str(runtime):
        raise ValueError('此备份仅恢复到原持久数据路径；迁移路径需要单独校验来源映射。')
    files = manifest.get('files')
    if not isinstance(files,list) or not files:
        raise ValueError('备份清单无效。')
    checked, workspaces = [], set()
    for item in files:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts)<2 or rel.parts[0] not in {'live','demo'}:
            raise ValueError('备份清单包含无效路径。')
        source = snapshot/rel
        if source.resolve()!=source or not source.is_file() or source.stat().st_size!=item['bytes'] or file_hash(source)!=item['sha256']:
            raise ValueError('备份文件不完整或内容已变化，未覆盖现有数据。')
        checked.append((rel,source))
        workspaces.add(rel.parts[0])
    with RuntimeLease(runtime):
        if any((runtime/name).exists() for name in workspaces):
            raise ValueError('恢复目标工作区已存在，请先保留或移走旧工作区；不会覆盖现有数据。')
        staging = runtime/('.restore-'+uuid.uuid4().hex)
        moved = []
        try:
            for rel,source in checked:
                target = staging/rel
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
            for name in sorted(workspaces):
                (staging/name).rename(runtime/name)
                moved.append(name)
        except Exception:
            for name in moved:
                (runtime/name).rename(staging/name)
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return dict(restored=len(checked),runtime=str(runtime))
