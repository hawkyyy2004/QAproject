"""Registered original paths; library files stay at their original location."""
import hashlib
import json
from pathlib import Path

from .persistence import file_hash


def signature(path, stat=None):
    s = stat if stat is not None else Path(path).stat()
    return hashlib.sha256(json.dumps([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]).encode()).hexdigest()


def original_path(doc, workspace):
    path = Path(doc['path'])
    base = Path(doc['source_root']) if doc.get('source_kind')=='library' else Path(workspace)/'originals'
    resolved = path.resolve()
    if not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
        raise ValueError('原始资料不存在或来源目录不可用。')
    if doc.get('source_kind')=='library' and resolved!=path:
        raise ValueError('资料库文件位置已变化，请重新选择资料。')
    return resolved


def verify_reference(doc, workspace):
    if doc.get('source_kind')!='library':
        return
    path = original_path(doc,workspace)
    current = signature(path)
    if current != doc['source_signature']:
        if file_hash(path)!=doc['content_hash'] or signature(path)!=current:
            raise ValueError('资料库原件已变更，旧证据不会覆盖；请从资料库选择当前版本。')
        # Metadata-only changes do not invalidate identical, frozen bytes.
        doc['source_signature'] = current
