import csv
import io
from pathlib import Path
import re

import pymupdf

from .store import digest

LONG_TEXT_THRESHOLD = 16000
TEXT_CHUNK_SIZE = 16000
VISUAL_TEXT_NOTICE = 'Read the complete actual page image. Its extracted text is registered separately.'


def text_spans(text):
    """Lossless, non-overlapping slices; ordinary evidence keeps its original shape."""
    if len(text) <= LONG_TEXT_THRESHOLD:
        yield 0, len(text)
        return
    start = 0
    while len(text)-start > TEXT_CHUNK_SIZE:
        end = start+TEXT_CHUNK_SIZE
        floor = start+TEXT_CHUNK_SIZE*3//4
        # Prefer a nearby line/sentence boundary; never scan the entire remainder.
        boundary = text.rfind('\n', floor, end)
        if boundary < floor:
            matches = list(re.finditer(r'[。！？.!?；;](?:\s|$)', text[floor:end]))
            boundary = floor+matches[-1].end()-1 if matches else text.rfind(' ', floor, end)
        if boundary >= floor:
            end = boundary+1
        yield start, end
        start = end
    yield start, len(text)


def model_text(unit):
    # The image already contains this full page. Avoid sending its long OCR twice.
    if unit.get('image') and len(unit['text']) > LONG_TEXT_THRESHOLD:
        return VISUAL_TEXT_NOTICE
    return unit['text']


def text_view_matches(unit, parent):
    if not parent or unit.get('companion_of') != parent['id'] or unit.get('image') or not parent.get('image'):
        return False
    if parent['text'] == unit['text'] and parent['locator'] == unit['locator']:
        return True
    span = unit.get('text_span')
    return (isinstance(span, list) and len(span)==2 and all(type(x) is int for x in span)
            and 0 <= span[0] < span[1] <= len(parent['text'])
            and unit.get('page') == parent.get('page')
            and unit['text'] == parent['text'][span[0]:span[1]])


def pdf_paragraphs(text):
    return [x.strip() for x in re.split(r'\n\s*\n', text) if x.strip()]


def has_full_page_image(page):
    """Recognize the common single direct scan without decoding it for metadata."""
    images = page.get_images(full=True)
    if len(images) != 1 or images[0][-1] != 0 or min(images[0][2:4]) <= 1:
        return False
    # Direct-resource bbox lookup uses the PDF display commands, not image hashes.
    bounds = page.rect * page.derotation_matrix
    rect = page.get_image_bbox(images[0]) & bounds
    if rect.width >= bounds.width*.95 and rect.height >= bounds.height*.95:
        return True
    return False


def has_graphics(page, textpage):
    """Detect displayed graphics; render their complete page without cropping."""
    bounds = page.rect * page.derotation_matrix
    # Same metadata path as Page.get_image_info(), reusing our already parsed TextPage.
    # Avoid a second page interpretation and Python image-byte/digest/xref extraction.
    for info in textpage.extractIMGINFO(hashes=False):
        rect = pymupdf.Rect(info['bbox']) & bounds
        if not rect.is_empty and info['width'] > 1 and info['height'] > 1:
            return True
    for drawing in page.get_cdrawings():
        rect = pymupdf.Rect(drawing['rect'])
        # Page backgrounds, borders and thin horizontal text rules are not figures.
        if rect.width >= bounds.width*.95 and rect.height >= bounds.height*.95 and len(drawing['items']) == 1:
            if drawing.get('fill') is None or all(c >= .98 for c in drawing['fill']):
                continue
        if rect.height <= 1 and rect.width > 20:
            continue
        if rect.width <= 1 and rect.height <= 1:
            continue
        rect = (rect + (-2, -2, 2, 2)) & bounds
        if not rect.is_empty:
            return True
    return False


def text_alternatives(units):
    """Keep frozen visual evidence and add a text view at the same true locator."""
    result = list(units)
    ids = {u['id'] for u in units}
    fragment_parents = {u.get('companion_of') for u in units if 'text_span' in u}
    for unit in units:
        text = unit['text']
        readable = len(text)>=120 and (len(re.findall('[\u4e00-\u9fff]',text))>=30
                                      or len(re.findall('[A-Za-z]',text))>=80)
        if not unit['image'] or not readable or unit['id']+':text' in ids or unit['id'] in fragment_parents:
            continue
        for index, (start, end) in enumerate(text_spans(text), 1):
            alternative = dict(unit, id=unit['id']+':text', image='', asset_id='',
                               companion_of=unit['id'], evidence_kind='extracted_text')
            if len(text)>LONG_TEXT_THRESHOLD:
                seg = unit['seg']+index
                alternative.update(id=unit['id']+f':text:{index}', text=text[start:end], seg=seg,
                                   locator=f'page-{unit["page"]:02d}-seg-{seg:02d}', text_span=[start,end])
            alternative.pop('hash', None)
            alternative['hash'] = digest(alternative)
            result.append(alternative)
            ids.add(alternative['id'])
        fragment_parents.add(unit['id'])
    return result


def extract(store, doc, config, progress=None):
    path = Path(doc['path'])
    folder = store.root / 'assets/images' / doc['id']
    folder.mkdir(parents=True, exist_ok=True)
    units = []
    seg_offsets = {}

    def add(text, page, seg, image='', asset_id='', locator=None, **metadata):
        spans = [(0,len(text))] if image else list(text_spans(text))
        offset = seg_offsets.get(page, 0)
        for index, (start,end) in enumerate(spans):
            actual_seg = seg+offset+index
            actual_locator = locator or f'page-{page:02d}-seg-{actual_seg:02d}'
            if locator and locator.startswith('seg-'):
                actual_locator = f'seg-{actual_seg:02d}'
            elif locator and len(spans)>1:
                actual_locator += f'-seg-{index+1:02d}'
            part = text[start:end]
            unit = {'id': f'{doc["id"]}:p{page}:s{actual_seg}', 'text': part, 'page': page, 'seg': actual_seg,
                    'locator': actual_locator, 'image': image, 'asset_id': asset_id,
                    'language': 'zh' if re.search('[\u4e00-\u9fff]', part) else 'en', **metadata}
            if len(spans)>1:
                unit.update(text_span=[start,end], source_block=locator or f'page-{page:02d}-seg-{seg:02d}')
            unit['hash'] = digest(unit)
            units.append(unit)
        seg_offsets[page] = offset+len(spans)-1

    suffix = path.suffix.lower()
    if suffix == '.pdf':
        with pymupdf.open(path) as pdf:
            if pdf.is_encrypted:
                raise ValueError('加密 PDF 暂不支持，请提供可读取副本。')
            max_pages = config.get('max_pages')
            if max_pages and len(pdf) > max_pages:
                raise ValueError(f'当前配置每份最多 {max_pages} 页，未截断处理。')
            if progress:
                progress(0, len(pdf))
            for n, page in enumerate(pdf, 1):
                visual = has_full_page_image(page)
                flags = pymupdf.TEXTFLAGS_TEXT | (0 if visual else pymupdf.TEXT_PRESERVE_IMAGES)
                textpage = page.get_textpage(flags=flags)
                text = page.get_text('text', sort=True, textpage=textpage).strip()
                if visual or has_graphics(page, textpage):
                    pixmap = page.get_pixmap(dpi=144, alpha=False)
                    # Check the final-resolution image: downsampling could erase faint/sparse evidence.
                    blank = not text and pixmap.is_unicolor and pixmap.pixel(0, 0) == (255, 255, 255)
                    if not blank:
                        out = folder / f'page-{n:04d}.png'
                        pixmap.save(out)
                        asset = store.add_asset(doc['id'], out, n, f'第 {n} 页完整图片')
                        add(text or 'Scan page: use actual visual evidence.', n, 1, str(out), asset,
                            evidence_kind='pdf_page')
                else:
                    for s, paragraph in enumerate(pdf_paragraphs(text), 1):
                        add(paragraph, n, s, evidence_kind='extracted_text')
                if progress and (n % 10 == 0 or n == len(pdf)):
                    progress(n, len(pdf))
            pages = len(pdf)
    elif suffix in {'.png', '.jpg', '.jpeg', '.webp'}:
        with pymupdf.open(path) as image_doc:
            page = image_doc[0]
            if page.rect.width * page.rect.height > 40_000_000:
                raise ValueError('图片像素过大，初版拒绝自动缩小而丢失细节。')
            out = folder / 'image-0001.png'
            page.get_pixmap(alpha=False).save(out)
        asset = store.add_asset(doc['id'], out, 0, '原始图像的 PNG 副本')
        add('Image evidence. Determine language from the actual image.', 0, 1, str(out), asset, 'image-0001')
        pages = 0
    elif suffix in {'.txt', '.md'}:
        text = path.read_text(encoding='utf-8-sig')
        for s, paragraph in enumerate(re.split(r'\n\s*\n', text), 1):
            if paragraph.strip():
                add(paragraph.strip(), 0, s, locator=f'seg-{s:02d}')
        pages = 0
    elif suffix == '.csv':
        rows = list(csv.reader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
        if not rows:
            raise ValueError('CSV 为空')
        for i in range(1, len(rows), 30):
            segment = rows[i:i+30]
            add(str({'columns': rows[0], 'rows': segment}), 0, i,
                locator=f'record-{i+1:04d}-{i+len(segment):04d}')
        pages = 0
    else:
        raise ValueError('初版支持 PDF、PNG/JPG/WebP、CSV、TXT/MD。')
    if not units:
        raise ValueError('没有可读取证据')
    for unit in units:
        if unit['image'] and Path(unit['image']).stat().st_size > 5*1024*1024:
            raise ValueError('图像超过初版 5MB 输入限额，未生成视觉 QA。')
    return text_alternatives(units), pages
