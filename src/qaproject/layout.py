"""Read the MinerU layout manifest that separates body text from figures.

MinerU runs outside the service (``scripts/mineru_layout.py``, own virtualenv)
and freezes one manifest per document, keyed by the PDF content hash. The
service only reads that manifest: crops already exist and captions are already
attached, so evidence preparation needs no model and no GPU. A document without
a manifest keeps the previous whole-page behaviour, so MinerU stays optional.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

from .console import child_env

SCHEMA = 1
KINDS = {'text', 'figure', 'table', 'formula', 'page'}
_RUNNER = {}
# A failed probe is retried after this long: installing MinerU or pointing
# MINERU_PYTHON at the right interpreter must not need a service restart.
PROBE_RETRY_SECONDS = 300


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


def runner(config):
    """The MinerU interpreter: settings, then MINERU_PYTHON, then a local venv."""
    block = config.get('mineru') or {}
    explicit = str(block.get('python') or '').strip()
    if not explicit:
        from .settings import ROOT, local_setting
        explicit = local_setting('MINERU_PYTHON')
        if not explicit:
            venv = ROOT / 'runtime/mineru-venv' / ('Scripts/python.exe' if sys.platform == 'win32'
                                                   else 'bin/python')
            explicit = str(venv) if venv.is_file() else sys.executable
    # A successful probe is cached for the process lifetime; a failure expires
    # so installing MinerU takes effect without a restart.
    cached = _RUNNER.get(explicit)
    if cached and (cached[0] or time.monotonic() - cached[1] < PROBE_RETRY_SECONDS):
        return explicit if cached[0] else ''
    ok = False
    try:
        # Probing costs about a second. Decoding is pinned to UTF-8 because a
        # Chinese Windows console pipe is GBK and a non-GBK byte would abort.
        probe = subprocess.run([explicit, '-c', 'import mineru'], capture_output=True, text=True,
                               encoding='utf-8', errors='replace', env=child_env(), timeout=120)
        ok = probe.returncode == 0
    except (OSError, subprocess.SubprocessError, ValueError):
        # A missing or non-executable interpreter is "unavailable", not a crash.
        ok = False
    _RUNNER[explicit] = (ok, time.monotonic())
    return explicit if ok else ''


def prepare(store, doc, config):
    """Build the manifest for this exact file so parsing can crop instead of page-shot.

    Returns the page map, or {} when MinerU is unavailable or fails; the caller
    then keeps the previous whole-page behaviour instead of failing the file.
    """
    block = config.get('mineru') or {}
    python = runner(config)
    if not python:
        return {}, '未找到可用的 MinerU 解释器，保留整页'
    script = Path(__file__).resolve().parents[2] / 'scripts/mineru_layout.py'
    # The script needs this project's PyMuPDF to crop; only MinerU runs over there.
    command = [sys.executable, str(script), str(doc['path']), '--layout-dir', str(Path(store.root) / 'layout'),
               '--mineru-python', python, '--tier', str(block.get('tier') or 'basic'),
               '--dpi', str(int(block.get('dpi') or 200)), '--jobs', '1']
    try:
        # PYTHONIOENCODING pins the child's pipe to UTF-8 even on a GBK Windows
        # console; otherwise a non-GBK character in its output kills the print.
        run = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                             errors='replace', env=child_env(),
                             timeout=int(block.get('timeout_seconds') or 900))
    except subprocess.TimeoutExpired:
        return {}, 'MinerU 超时，保留整页'
    pages = load(store, doc.get('content_hash'))
    if pages:
        return pages, ''
    detail = (run.stderr or run.stdout or '').strip().splitlines()
    return {}, f'MinerU 未生成清单，保留整页：{detail[-1][:200] if detail else "无输出"}'
