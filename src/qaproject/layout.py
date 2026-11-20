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


def load(store, content_hash, *, strict=False):
    """Return {page: [blocks]} for a frozen manifest; {} asks for the fallback."""
    if not content_hash:
        return {}
    path = manifest_path(store, content_hash)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        if strict: raise ValueError('MinerU版面清单不可读取，请重试资料整理。') from None
        return {}
    if data.get('schema') != SCHEMA or data.get('content_hash') != content_hash:
        if strict: raise ValueError('MinerU版面清单与原件不匹配。')
        return {}
    pages = {}
    for entry in data.get('pages') or []:
        page = entry.get('page')
        if not isinstance(page, int) or page < 1:
            if strict: raise ValueError('MinerU版面清单页码无效。')
            continue
        if strict and page in pages: raise ValueError('MinerU版面清单含重复页面。')
        blocks = []
        for block in entry.get('blocks') or []:
            kind = block.get('kind')
            if kind not in KINDS:
                if strict: raise ValueError('MinerU版面清单含未知证据类型。')
                continue
            item = dict(block)
            if kind != 'text':
                crop = Path(block.get('path') or '')
                crop = crop if crop.is_absolute() else path.parent / crop
                if not crop.is_file():
                    # A missing crop must not silently degrade into a blank unit.
                    if strict: raise ValueError('MinerU独立插图文件缺失，未改用整页截图。')
                    continue
                item['path'] = str(crop)
            if kind == 'text' and not (block.get('text') or '').strip():
                continue
            blocks.append(item)
        if blocks or strict or entry.get('status') in {'filtered','blank'}:
            pages[page] = blocks
    if strict:
        total = data.get('pdf', {}).get('pages')
        if type(total) is not int or total < 1 or set(pages) != set(range(1, total+1)):
            raise ValueError('MinerU版面清单页面不完整，未截断或改用整页截图。')
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

    Returns the page map, or {} with a reason on failure. The original optional
    parser may fall back; shared MinerU preparation requires success and fails
    explicitly instead of substituting a page render.
    """
    block = config.get('mineru') or {}
    raw = Path(store.root) / 'corpus/mineru' / str(doc['content_hash']) / 'result.json'
    cached_raw = manifest_path(store,doc['content_hash']).parent/'mineru.json'
    failure_suffix = '未改用整页截图' if config.get('require_mineru') else '保留整页'
    python = runner(config)
    if not python and not (raw.is_file() or cached_raw.is_file()):
        return {}, '未找到可用的 MinerU 解释器，'+failure_suffix
    script = Path(__file__).resolve().parents[2] / 'scripts/mineru_layout.py'
    # The script needs this project's PyMuPDF to crop; only MinerU runs over there.
    command = [sys.executable, str(script), str(doc['path']), '--layout-dir', str(Path(store.root) / 'layout'),
               '--mineru-python', python or sys.executable, '--tier', str(block.get('tier') or 'basic'),
               '--dpi', str(int(block.get('dpi') or 200)), '--jobs', '1', '--rebuild']
    # Earlier Corpus versions stored raw MinerU JSON without figure assets.
    # Build the original figure manifest from it instead of running MinerU twice.
    if raw.is_file(): command.extend(['--raw-json', str(raw)])
    try:
        # PYTHONIOENCODING pins the child's pipe to UTF-8 even on a GBK Windows
        # console; otherwise a non-GBK character in its output kills the print.
        run = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                             errors='replace', env=child_env(),
                             timeout=int(block.get('timeout_seconds') or 900))
    except subprocess.TimeoutExpired:
        return {}, 'MinerU 超时，'+failure_suffix
    if run.returncode:
        return {}, preparation_error(run.stdout)
    pages = load(store, doc.get('content_hash'), strict=bool(config.get('require_mineru')))
    if pages:
        return pages, ''
    detail = (run.stderr or run.stdout or '').strip().splitlines()
    return {}, f'MinerU 未生成清单，{failure_suffix}：{detail[-1][:200] if detail else "无输出"}'


def preparation_error(output):
    """Expose controlled layout diagnostics; never echo arbitrary child output."""
    import re
    for line in str(output or '').splitlines():
        try:result=json.loads(line)
        except ValueError:continue
        if not isinstance(result,dict) or result.get('status')!='rejected':continue
        warnings=result.get('warnings')
        if not isinstance(warnings,list):continue
        for warning in warnings:
            if not isinstance(warning,str):continue
            match=re.fullmatch(r'第(\d+)页未识别出证据，未当作空白页跳过',warning)
            if match:return f'MinerU第{match[1]}页未识别出可用证据，请检查该页解析结果。'
            match=re.fullmatch(r'第(\d+)页独立图块无法保存，未丢弃图片或改用整页截图',warning)
            if match:return f'MinerU第{match[1]}页独立图块保存失败，未改用整页截图。'
            if warning=='MinerU页面不完整，未截断或改用整页截图':return warning
            match=re.fullmatch(r'MinerU 只识别出 (\d+) 字符，PDF 文字层有 (\d+) 字符，本次版面未通过完整性校验',warning)
            if match:return f'MinerU正文完整性校验未通过（识别{match[1]}字符，原文文字层{match[2]}字符）。'
    return 'MinerU解析或独立插图准备失败，未生成可用清单'
