"""Read the MinerU layout manifest that separates body text from figures.

MinerU runs outside the service (``scripts/mineru_layout.py``, own virtualenv)
and freezes one manifest per document, keyed by the PDF content hash. The
service only reads that manifest: crops already exist and captions are already
attached, so evidence preparation needs no model and no GPU. A document without
a manifest keeps the previous whole-page behaviour, so MinerU stays optional.
"""
import json
from pathlib import Path

SCHEMA = 1
KINDS = {'text', 'figure', 'table', 'formula', 'page'}


def manifest_path(store, content_hash):
    return Path(store.root) / 'layout' / str(content_hash) / 'manifest.json'


def load(store, content_hash):
    """Return {page: [blocks]} for a frozen manifest; {} asks for the fallback."""
    if not content_hash:
        return {}
    path = manifest_path(store, content_hash)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if data.get('schema') != SCHEMA or data.get('content_hash') != content_hash:
        return {}
    pages = {}
    for entry in data.get('pages') or []:
        page = entry.get('page')
        if not isinstance(page, int) or page < 1:
            continue
        blocks = []
        for block in entry.get('blocks') or []:
            kind = block.get('kind')
            if kind not in KINDS:
                continue
            item = dict(block)
            if kind != 'text':
                crop = Path(block.get('path') or '')
                crop = crop if crop.is_absolute() else path.parent / crop
                if not crop.is_file():
                    # A missing crop must not silently degrade into a blank unit.
                    continue
                item['path'] = str(crop)
            if kind == 'text' and not (block.get('text') or '').strip():
                continue
            blocks.append(item)
        if blocks:
            pages[page] = blocks
    return pages
