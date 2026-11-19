"""Freeze a MinerU layout manifest so evidence preparation never renders a whole page.

Run with the project interpreter; MinerU itself is called through its own
virtualenv, so the service environment stays free of ONNX/GGUF tooling:

    python scripts/mineru_layout.py --list pdfs.txt --layout-dir runtime/live/layout

One directory per document, keyed by the PDF content hash:

    <layout-dir>/<sha256>/manifest.json
    <layout-dir>/<sha256>/mineru.json     raw MinerU output, kept for auditing
    <layout-dir>/<sha256>/crops/*.png     one crop per figure/table/formula

``qaproject.parsers`` reads the manifest and falls back to whole-page rendering
only for pages it does not describe, so this step is optional, resumable and
safe to skip on a machine without MinerU.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pymupdf

from qaproject.console import child_env, repair_streams

ROOT = Path(__file__).resolve().parents[1]
VISUAL = {'image', 'chart'}
TEXT = {'text', 'paragraph_title', 'doc_title', 'index', 'ref_text'}
CAPTION = re.compile(r'^\s*(图|表|Figure|Fig\.?|Table)\s*\d', re.I)
BRACES = re.compile(r'\{\s*([^{}]*?)\s*\}')
GROUP_GAP = 0.035
H_GROUP_GAP = 0.06
EQUATION_GAP = 0.012
PREFIX = {'figure': 'fig', 'table': 'tbl', 'formula': 'eq', 'page': 'page'}


def compact_math(text):
    """MinerU writes inline math with a space between every glyph: ^{ 2 1 0 }."""
    return BRACES.sub(lambda m: '{' + re.sub(r'\s+', '', m.group(1)) + '}', text)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(content):
    """MinerU segments one paragraph into styled runs; concatenate them as printed."""
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        value = item.get('content') if isinstance(item, dict) else None
        parts.append(flatten(value) if value is not None else '')
    return ''.join(parts)


def block_text(block):
    content = block.get('content')
    return flatten(content).strip() if not isinstance(content, list) or content else ''


def block_parts(block, marker):
    parts = []
    for item in block.get('content') or []:
        if isinstance(item, dict) and marker in str(item.get('type', '')):
            value = flatten(item.get('content')).strip()
            if value:
                parts.append(value)
    return parts


def block_caption(block):
    return compact_math(' '.join(block_parts(block, 'caption')).strip())


def caption_items(block):
    """Captions with their own boxes: two figures on one page can share a block."""
    out = []
    for item in block.get('content') or []:
        if isinstance(item, dict) and 'caption' in str(item.get('type', '')):
            text = compact_math(flatten(item.get('content')).strip())
            if text:
                out.append((text, item.get('bbox')))
    return out


def block_html(block):
    for item in block.get('content') or []:
        if isinstance(item, dict) and item.get('type') == 'table_body':
            return compact_math(flatten(item.get('content')).strip())
    return ''


def merge(*boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def overlap_ratio(a, b):
    """Horizontal overlap relative to the narrower box; panels share a column."""
    shared = min(a[2], b[2]) - max(a[0], b[0])
    narrow = min(a[2] - a[0], b[2] - b[0])
    return shared / narrow if narrow > 0 else 0


def vertical_gap(a, b):
    """Distance between two boxes vertically; negative when they overlap."""
    if b[1] >= a[3]:
        return b[1] - a[3]
    if a[1] >= b[3]:
        return a[1] - b[3]
    return -min(a[3] - a[1], b[3] - b[1])


def horizontal_gap(a, b):
    if b[0] >= a[2]:
        return b[0] - a[2]
    if a[0] >= b[2]:
        return a[0] - b[2]
    return -min(a[2] - a[0], b[2] - b[0])


def overlap_y(a, b):
    """Vertical overlap relative to the shorter box; side-by-side panels share a band."""
    shared = min(a[3], b[3]) - max(a[1], b[1])
    shorter = min(a[3] - a[1], b[3] - b[1])
    return shared / shorter if shorter > 0 else 0


def group_visuals(blocks):
    """Keep panels of one multi-panel figure in a single evidence unit.

    Panels stack vertically, sit side by side, or form a grid; two blocks that
    already carry their own caption are separate figures and stay separate.
    """
    groups = []
    for block in sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0])):
        for group in groups:
            box = block['bbox']
            stacked = overlap_ratio(group['bbox'], box) >= 0.5 and \
                vertical_gap(group['bbox'], box) <= GROUP_GAP
            side_by_side = overlap_y(group['bbox'], box) >= 0.5 and \
                horizontal_gap(group['bbox'], box) <= H_GROUP_GAP
            if (stacked or side_by_side) and not (group['caption'] and block.get('_caption')):
                group['members'].append(block)
                group['bbox'] = merge(group['bbox'], box)
                group['caption'] = group['caption'] or block.get('_caption', '')
                break
        else:
            groups.append({'members': [block], 'bbox': list(block['bbox']),
                           'caption': block.get('_caption', '')})
    return groups


def assign_captions(blocks, texts):
    """Give every visual and table its own 图/Fig line before any panel merging.

    Panels of one figure share a single caption, so a block without one may be
    merged into a captioned group. Two blocks that each own a caption are
    different figures and must not be merged, which is why this runs first.
    """
    claimed, found = set(), {}
    loose = []
    for block in blocks:
        items = caption_items(block)
        placed = [(text, box) for text, box in items if box]
        if placed:
            loose.extend(placed)
        elif items:
            # Without geometry the caption can only belong to the block carrying it.
            found[id(block)] = ' '.join(text for text, _ in items)
    for text, box in loose:
        nearest = None
        for block in blocks:
            below = vertical_gap(block['bbox'], box)
            above = vertical_gap(box, block['bbox'])
            offset = below if below >= 0 else (above if above >= 0 else 0)
            if offset <= 0.03 and overlap_ratio(block['bbox'], box) >= 0.5:
                if nearest is None or offset < nearest[0]:
                    nearest = (offset, block)
        if nearest:
            found[id(nearest[1])] = ' '.join(filter(None, [found.get(id(nearest[1])), text]))
    for block in blocks:
        if id(block) in found:
            continue
        best = None
        for text in texts:
            if id(text) in claimed or not CAPTION.match(block_text(text)):
                continue
            below = vertical_gap(block['bbox'], text['bbox'])
            above = vertical_gap(text['bbox'], block['bbox'])
            offset = below if below >= 0 else (above if above >= 0 else 0)
            if offset <= 0.03 and overlap_ratio(block['bbox'], text['bbox']) >= 0.5:
                if best is None or offset < best[0]:
                    best = (offset, text)
        if best:
            found[id(block)] = block_text(best[1])
            claimed.add(id(best[1]))
    return found, claimed


def plan_page(blocks):
    """Turn MinerU blocks into evidence items in reading order."""
    visuals = [b for b in blocks if b.get('type') in VISUAL]
    equations = [b for b in blocks if b.get('type') == 'equation']
    tables = [b for b in blocks if b.get('type') == 'table']
    texts = [b for b in blocks if b.get('type') in TEXT and len(block_text(b)) >= 2]
    captions, claimed = assign_captions(visuals + tables, texts)
    for block in visuals + tables:
        block['_caption'] = captions.get(id(block), '')
    groups = group_visuals(visuals)
    loose = []
    for equation in equations:
        for group in groups:
            # Parameter labels drawn inside a figure belong to that figure.
            if overlap_ratio(group['bbox'], equation['bbox']) >= 0.5 and \
                    vertical_gap(group['bbox'], equation['bbox']) <= EQUATION_GAP:
                group['members'].append(equation)
                group['bbox'] = merge(group['bbox'], equation['bbox'])
                break
        else:
            loose.append(equation)

    used = set(claimed)

    def inherit_caption(box):
        """A panel group may lose its 图/Fig pair to neighbouring text blocks."""
        found = []
        for block in texts:
            if id(block) in used or not CAPTION.match(block_text(block)):
                continue
            # Captions sit just below (or just above) the artwork they describe.
            offset = min(abs(vertical_gap(box, block['bbox'])), abs(vertical_gap(block['bbox'], box)))
            if offset <= 0.03 and overlap_ratio(box, block['bbox']) >= 0.5:
                used.add(id(block))
                found.append(block_text(block))
        return compact_math(' '.join(found))

    items = []
    for group in groups:
        caption = ' '.join(dict.fromkeys(c for c in (b.get('_caption', '') for b in group['members']) if c))
        if not caption:
            caption = inherit_caption(group['bbox'])
        kind = 'figure'
        coverage = (group['bbox'][2] - group['bbox'][0]) * (group['bbox'][3] - group['bbox'][1])
        if not any(len(block_text(b)) >= 15 for b in texts) and coverage >= 0.7:
            kind = 'page'
        items.append({'kind': kind, 'bbox': group['bbox'], 'caption': caption,
                      'sort': (group['bbox'][1], group['bbox'][0])})
    for table in tables:
        caption = table.get('_caption', '') or inherit_caption(table['bbox'])
        items.append({'kind': 'table', 'bbox': list(table['bbox']), 'caption': caption,
                      'html': block_html(table), 'sort': (table['bbox'][1], table['bbox'][0])})
    for equation in loose:
        items.append({'kind': 'formula', 'bbox': list(equation['bbox']),
                      'latex': block_text(equation), 'sort': (equation['bbox'][1], equation['bbox'][0])})
    for block in texts:
        if id(block) not in used:
            items.append({'kind': 'text', 'text': block_text(block),
                          'sort': (block['bbox'][1], block['bbox'][0])})
    items.sort(key=lambda item: item['sort'])
    counts = dict.fromkeys(PREFIX, 0)
    for item in items:
        if item['kind'] != 'text':
            counts[item['kind']] += 1
            item['index'] = counts[item['kind']]
    return items


def crop(page, box, target, dpi, pad=1.0):
    """Render only the block; thin formula boxes need vertical air to stay readable."""
    rect = pymupdf.Rect(box[0] * page.rect.width, box[1] * page.rect.height,
                        box[2] * page.rect.width, box[3] * page.rect.height)
    if rect.height < 18:
        pad = max(pad, 6.0)
    rect = (rect + (-pad, -pad, pad, pad)) & page.rect
    if rect.is_empty or rect.width < 10 or rect.height < 8:
        return False
    page.get_pixmap(dpi=dpi, clip=rect, alpha=False).save(target)
    return True


def build_manifest(pdf_path, data, target, dpi, tier, content_hash, warning):
    """Plan every page first: a layout that missed half the text must not be frozen."""
    manifest = {'schema': 1, 'engine': 'mineru', 'tier': tier, 'content_hash': content_hash,
                'pdf': {'name': pdf_path.name, 'pages': 0}, 'pages': [], 'warnings': []}
    raw_chars = sum(len(block_text(block)) + len(block_html(block))
                    for page in data.get('pages') or [] for block in page.get('blocks') or [])
    with pymupdf.open(pdf_path) as pdf:
        manifest['pdf']['pages'] = len(pdf)
        numbers = [p.get('page_idx') for p in data.get('pages') or []]
        if len(numbers) != len(pdf) or set(numbers) != set(range(len(pdf))):
            warning('MinerU页面不完整，未截断或改用整页截图')
            return None
        pdf_chars = sum(len(page.get_text()) for page in pdf)
        if pdf_chars >= 400 and raw_chars < pdf_chars * 0.5:
            warning(f'MinerU 只识别出 {raw_chars} 字符，PDF 文字层有 {pdf_chars} 字符，本次版面未通过完整性校验')
            return None
        crops = target / 'crops'
        crops.mkdir(parents=True, exist_ok=True)
        for page_data in data.get('pages') or []:
            number = int(page_data.get('page_idx', 0)) + 1
            if not 1 <= number <= len(pdf):
                continue
            page = pdf[number - 1]
            entry = {'page': number, 'blocks': []}
            for item in plan_page(page_data.get('blocks') or []):
                if item['kind'] == 'text':
                    entry['blocks'].append({'kind': 'text', 'text': item['text']})
                    continue
                name = f'page-{number:02d}-{PREFIX[item["kind"]]}-{item["index"]:02d}.png'
                if not crop(page, item['bbox'], crops / name, dpi):
                    warning(f'第{number}页独立图块无法保存，未丢弃图片或改用整页截图')
                    return None
                block = {'kind': item['kind'], 'bbox': [round(x, 4) for x in item['bbox']],
                         'path': f'crops/{name}'}
                for key in ('caption', 'html', 'latex'):
                    if item.get(key):
                        block[key] = item[key]
                entry['blocks'].append(block)
            if not entry['blocks'] and (page.get_text().strip() or page.get_images() or page.get_drawings()):
                warning(f'第{number}页未识别出证据，未当作空白页跳过')
                return None
            manifest['pages'].append(entry)
    manifest['chars'] = {'mineru': raw_chars, 'pdf_text_layer': pdf_chars}
    return manifest


def parse_args(argv):
    parser = argparse.ArgumentParser(description='生成 MinerU 版面清单与裁图')
    parser.add_argument('paths', nargs='*', type=Path, help='PDF 文件或目录')
    parser.add_argument('--list', type=Path, help='UTF-8 文本文件，每行一个 PDF 路径')
    parser.add_argument('--layout-dir', type=Path, default=ROOT / 'runtime/layout')
    parser.add_argument('--mineru-python', default=os.environ.get('MINERU_PYTHON', sys.executable))
    parser.add_argument('--tier', choices=['basic', 'standard'], default='basic')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--jobs', type=int, default=1, help='同时解析的文件数，4~6 为安全起点')
    parser.add_argument('--limit', type=int, help='只处理前 N 个文件')
    parser.add_argument('--force', action='store_true', help='已存在的清单也重新解析')
    parser.add_argument('--rebuild', action='store_true', help='复用原始MinerU结果重建独立插图清单')
    parser.add_argument('--raw-json', type=Path, help='同一原件已完成的MinerU JSON，仅与单份PDF一起使用')
    return parser.parse_args(argv)


def collect(args):
    files = []
    for path in args.paths:
        files.extend(sorted(path.rglob('*.pdf')) if path.is_dir() else [path])
    if args.list:
        for line in args.list.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip().strip('"')
            if line:
                files.append(Path(line))
    seen, unique = set(), []
    for path in files:
        resolved = path.resolve()
        if resolved.is_file() and resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique[:args.limit] if args.limit else unique


def process(pdf, args):
    started = time.time()
    content_hash = sha256_file(pdf)
    target = args.layout_dir / content_hash
    if (target / 'manifest.json').is_file() and not args.force and not getattr(args, 'rebuild', False):
        return {'file': pdf.name, 'status': 'skipped'}
    target.mkdir(parents=True, exist_ok=True)
    raw = target / 'mineru.json'
    # The MinerU worker also prints to a pipe: pin UTF-8 on both hops so a
    # Chinese Windows locale cannot mangle or crash the captured output.
    env = child_env(dict(os.environ, MINERU_MODEL_SOURCE=os.environ.get('MINERU_MODEL_SOURCE', 'modelscope')))
    worker = ROOT / 'scripts/mineru_parse_worker.py'
    prior = getattr(args, 'raw_json', None)
    if not raw.is_file() and prior and prior.is_file():
        shutil.copyfile(prior, raw)
    if args.force or not raw.is_file():
        run = subprocess.run([args.mineru_python, str(worker), str(pdf), str(raw), args.tier],
                             capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        if run.returncode:
            raw.unlink(missing_ok=True)
            return {'file': pdf.name, 'status': 'failed', 'error': (run.stderr or run.stdout)[-400:]}
    warnings = []
    manifest = build_manifest(pdf, json.loads(raw.read_text(encoding='utf-8')), target,
                              args.dpi, args.tier, content_hash, warnings.append)
    if manifest is None:
        return {'file': pdf.name, 'status': 'rejected', 'warnings': warnings}
    manifest['warnings'] = warnings
    temp = target / 'manifest.json.tmp'
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    temp.replace(target / 'manifest.json')
    return {'file': pdf.name, 'status': 'ok', 'pages': manifest['pdf']['pages'],
            'crops': sum(1 for p in manifest['pages'] for b in p['blocks'] if b['kind'] != 'text'),
            'seconds': round(time.time() - started, 1), 'warnings': warnings}


def main(argv=None):
    # Progress lines are JSON with Chinese and scientific symbols; a GBK console
    # must print them with replacements instead of aborting the run.
    repair_streams()
    args = parse_args(argv)
    files = collect(args)
    if not files:
        raise SystemExit('没有找到 PDF；请给出文件、目录或 --list 路径清单。')
    if args.raw_json and len(files) != 1:
        raise SystemExit('--raw-json只能用于同一份PDF，不能跨文件复用。')
    need_model=args.force or any(not (args.layout_dir/sha256_file(pdf)/'mineru.json').is_file()
        and not (args.raw_json and args.raw_json.is_file()) for pdf in files)
    if need_model:
        try:
            check = subprocess.run([args.mineru_python, '-c', 'import mineru'],
                                   capture_output=True, text=True, encoding='utf-8', errors='replace',
                                   env=child_env())
        except OSError as error:
            raise SystemExit(f'{args.mineru_python} 无法执行（{error.strerror or error}），'
                             f'请运行 scripts/setup_mineru.py 并用 --mineru-python 指向它的解释器。')
        if check.returncode:
            raise SystemExit(f'{args.mineru_python} 里没有 MinerU，请先运行 scripts/setup_mineru.py '
                             f'并用 --mineru-python 指向它的解释器。')
    args.layout_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'files': len(files), 'tier': args.tier, 'dpi': args.dpi,
                      'layout_dir': str(args.layout_dir)}, ensure_ascii=False), flush=True)
    totals = {'ok': 0, 'skipped': 0, 'failed': 0, 'rejected': 0}
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for result in pool.map(lambda item: process(item, args), files):
            totals[result['status']] = totals.get(result['status'], 0) + 1
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps(dict(totals, seconds=round(time.time() - started, 1)), ensure_ascii=False))
    return 0 if not (totals['failed'] or totals['rejected']) else 1


if __name__ == '__main__':
    sys.exit(main())
