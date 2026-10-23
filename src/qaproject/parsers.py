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
    bounds = page.rect * page.derotation_matrix
    """A scan is a page whose artwork is one image covering the whole sheet."""
    for image in page.get_images(full=True):
        if min(image[2:4]) <= 1:
            continue
        # Direct-resource bbox lookup uses the PDF display commands, not image hashes.
        try:
            rect = page.get_image_bbox(image) & bounds
        except (ValueError, RuntimeError):
            continue  # resource kept but never drawn
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


def figure_regions(page):
    """Locate figure areas from PDF display commands, without decoding images.

    Returns page-space rectangles merged from raster placements and from filled or
    curved vector drawings. Straight rules are ignored so table grids stay text.
    """
    bounds = page.rect * page.derotation_matrix
    rasters, vectors = [], []
    for kind, bbox in page.get_bboxlog():
        if not ("image" in kind or "imgmask" in kind):
            continue
        rect = pymupdf.Rect(bbox) & bounds
        if not rect.is_empty and rect.width > 4 and rect.height > 4:
            rasters.append(rect)
    for drawing in page.get_cdrawings():
        # Bars carry a fill, curves use 'c'; multi-segment line charts are polylines.
        if not (drawing.get('fill') or any(item[0] == 'c' for item in drawing['items'])
                or len(drawing['items']) >= 3
                or (pymupdf.Rect(drawing['rect']).width >= 12
                    and pymupdf.Rect(drawing['rect']).height >= 12)):
            continue
        rect = pymupdf.Rect(drawing['rect']) & bounds
        # Zero-height bars still bridge neighbouring panels; the cluster is checked later.
        if rect.width <= 1 or rect.height < .5:
            continue
        if rect.width >= bounds.width*.95 and rect.height >= bounds.height*.95:
            continue
        vectors.append(rect)
    # Keep artwork tight; merging a wide cluster with text only gets rejected later.
    merged = merge_regions(rasters, gap=20) + merge_regions(vectors, gap=60)
    coverage = max((rect.get_area() for rect in rasters+vectors), default=0) / bounds.get_area()
    return bounds, merged, coverage, rasters+vectors


def merge_regions(parts, gap=8):
    """Union rectangles that touch once each is padded by gap points."""
    merged = True
    while merged and len(parts) > 1:
        merged = False
        result = []
        for rect in parts:
            padded = rect + (-gap, -gap, gap, gap)
            for index, other in enumerate(result):
                if padded.intersects(other):
                    result[index] = other | rect
                    merged = True
                    break
            else:
                result.append(rect)
        parts = result
    return parts


def text_outside(page, regions):
    """Text blocks whose centre is not inside a figure region."""
    kept = []
    for block in page.get_text('blocks', sort=True):
        x0, y0, x1, y1, text = block[:5]
        if block[6] != 0 or not text.strip():
            continue
        centre = pymupdf.Point((x0+x1)/2, (y0+y1)/2)
        if any(rect.contains(centre) for rect in regions):
            continue
        kept.append(text.strip())
    return kept


CAPTION = re.compile(r'^(fig(?:ure)?\.?|tab(?:le)?\.?|图|表|附图|附表)\s*[\d一二三四五六七八九十]', re.I)
TABLE_CAPTION = re.compile(r'^(tab(?:le)?\.?|表|附表)\s*[\d一二三四五六七八九十]', re.I)


def figure_caption(blocks, region, bounds):
    """Nearest caption line around a figure, used as grounding context for its image."""
    best = None
    for x0, y0, x1, y1, text, *_ in blocks:
        if not CAPTION.match(text.strip()):
            continue
        # Below the artwork is most common; tables and some journals place it above.
        if y0 < region.y0-48 or y0 > region.y1+48:
            continue
        if x1 < region.x0-24 or x0 > region.x1+24:
            continue
        distance = 0 if region.y0 <= y0 <= region.y1 else min(abs(y0-region.y0), abs(y0-region.y1))
        if best is None or distance < best[0] or (distance == best[0] and y0 > best[1]):
            best = (distance, y0, text.strip())
    return best[2] if best else ''


def usable_regions(regions, bounds):
    """Drop rules, hairline shading and page-wide backgrounds."""
    page_area = bounds.get_area()
    kept = []
    for rect in regions:
        if rect.width < 40 or rect.height < 24:
            continue
        area = rect.get_area()
        if area < page_area*.01 or area > page_area*.9:
            continue
        kept.append(rect)
    return kept


def near_rects(left, right, gap=40):
    """True when two boxes are close enough to belong to one multi-panel figure."""
    horizontal = min(left.y1, right.y1)-max(left.y0, right.y0)
    vertical = min(left.x1, right.x1)-max(left.x0, right.x0)
    return horizontal > -gap and vertical > -gap


def table_or_prose(rect, caption, blocks):
    """True when a candidate box holds a table or a dense text panel, not artwork."""
    if caption and TABLE_CAPTION.match(caption.strip()):
        return True
    chars = sum(len(b[4].strip()) for b in blocks
                if rect.contains(pymupdf.Point((b[0]+b[2])/2, (b[1]+b[3])/2)))
    area_k = rect.get_area()/1000 or 1
    return chars >= 150 and chars/area_k >= 9


def group_figures(regions, blocks, bounds):
    """Attach captions and merge adjacent panels that share one caption."""
    captions = [figure_caption(blocks, rect, bounds) for rect in regions]
    # Tables are text evidence; cropping them would drop every cell.
    keep = [(rect, caption) for rect, caption in zip(regions, captions)
            if not table_or_prose(rect, caption, blocks)]
    regions = [rect for rect, _ in keep]
    captions = [caption for _, caption in keep]
    merged = []
    for rect, caption in zip(regions, captions):
        for index, (other, other_caption) in enumerate(merged):
            if caption and caption == other_caption and near_rects(rect, other, 90):
                merged[index] = (other | rect, caption)
                break
        else:
            merged.append((rect, caption))
    # A merge must not swallow the text between two distant panels.
    kept = [(rect, caption) for rect, caption in merged
            if drop_text_heavy([rect], blocks, bounds)]
    # A raster figure and its vector overlay cover the same artwork; crop it once.
    unique = []
    for rect, caption in sorted(kept, key=lambda item: item[0].get_area(), reverse=True):
        if any((rect & other).get_area() >= rect.get_area()*.85 for other, _ in unique):
            continue
        unique.append((rect, caption))
    return [rect for rect, _ in unique], [caption for _, caption in unique]


def drop_text_heavy(regions, blocks, bounds):
    """Reject regions that swallow body text or tables instead of a figure."""
    page_area = bounds.get_area()
    total = sum(len(b[4].strip()) for b in blocks) or 1
    kept = []
    for rect in regions:
        inside = sum(len(b[4].strip()) for b in blocks
                     if rect.contains(pymupdf.Point((b[0]+b[2])/2, (b[1]+b[3])/2)))
        if inside/total > .25 or rect.get_area() > page_area*.8:
            continue
        kept.append(rect)
    return kept


def text_alternatives(units):
    """Keep frozen visual evidence and add a text view at the same true locator."""
    result = list(units)
    ids = {u['id'] for u in units}
    fragment_parents = {u.get('companion_of') for u in units if 'text_span' in u}
    for unit in units:
        text = unit['text']
        readable = len(text)>=120 and (len(re.findall('[\u4e00-\u9fff]',text))>=30
                                      or len(re.findall('[A-Za-z]',text))>=80)
        if (not unit['image'] or not readable or unit['id']+':text' in ids
                or unit['id'] in fragment_parents or unit.get('evidence_kind') == 'pdf_figure'):
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
                # With region cropping off, keep the historical whole-page fallback.
                bounds, regions, coverage, blocks, captions, parts = None, [], 1.0, [], [], []
                candidates = False
                if not visual and config.get('figure_regions', True):
                    bounds, regions, coverage, parts = figure_regions(page)
                    regions = usable_regions(regions, bounds)
                    candidates = bool(regions)
                    if regions:
                        blocks = [b for b in page.get_text('blocks', sort=True) if b[6] == 0 and b[4].strip()]
                        regions = drop_text_heavy(regions, blocks, bounds)
                        regions, captions = group_figures(regions, blocks, bounds)
                        if not regions and coverage >= .1:
                            # One big cluster swallowed body text or a table; retry per artwork box.
                            regions = usable_regions(parts, bounds)
                            regions = drop_text_heavy(regions, blocks, bounds)
                            regions, captions = group_figures(regions, blocks, bounds)
                # Candidates carrying only body text were shading or rules, not figures.
                text_page = candidates and not regions
                if (visual or (not regions and not text_page and coverage > .05
                               and has_graphics(page, textpage))):
                    pixmap = page.get_pixmap(dpi=144, alpha=False)
                    # Check the final-resolution image: downsampling could erase faint/sparse evidence.
                    blank = not text and pixmap.is_unicolor and pixmap.pixel(0, 0) == (255, 255, 255)
                    if not blank:
                        out = folder / f'page-{n:04d}.png'
                        pixmap.save(out)
                        asset = store.add_asset(doc['id'], out, n, f'第 {n} 页完整图片')
                        add(text or 'Scan page: use actual visual evidence.', n, 1, str(out), asset,
                            evidence_kind='pdf_page')
                elif regions:
                    # A caption belongs to its figure unit; keep it out of body text.
                    taken = set()
                    for rect, caption in zip(regions, captions):
                        if not caption:
                            continue
                        for bi, block in enumerate(blocks):
                            if bi in taken or block[4].strip() != caption:
                                continue
                            if rect.y1-2 <= block[1] <= rect.y1+48:
                                taken.add(bi)
                                break
                    segment = 0
                    for bi, (x0, y0, x1, y1, block_text, *_) in enumerate(blocks):
                        if bi in taken:
                            continue
                        body = block_text.strip()
                        if any(rect.contains(pymupdf.Point((x0+x1)/2, (y0+y1)/2)) for rect in regions):
                            continue
                        segment += 1
                        add(body, n, segment, evidence_kind='extracted_text')
                    for index, (rect, caption) in enumerate(zip(regions, captions), 1):
                        clip = ((rect + (-4, -4, 4, 4)) & bounds) * page.rotation_matrix
                        pixmap = page.get_pixmap(clip=clip, dpi=144, alpha=False)
                        if pixmap.is_unicolor and pixmap.pixel(0, 0) == (255, 255, 255):
                            continue
                        out = folder / f'page-{n:04d}-fig-{index:02d}.png'
                        pixmap.save(out)
                        asset = store.add_asset(doc['id'], out, n, f'第 {n} 页插图 {index}')
                        segment += 1
                        add(caption or 'Figure region: use actual visual evidence.', n, segment,
                            str(out), asset, f'page-{n:02d}-fig-{index:02d}', evidence_kind='pdf_figure')
                else:
                    for s, paragraph in enumerate(pdf_paragraphs(text), 1):
                        add(paragraph, n, s, evidence_kind='extracted_text')
                if progress and (n % 10 == 0 or n == len(pdf)):
                    progress(n, len(pdf))
            pages = len(pdf)
    elif suffix in {'.png', '.jpg', '.jpeg', '.webp'}:
        raw = path.read_bytes()
        expected = {'.png':'png', '.jpg':'jpeg', '.jpeg':'jpeg', '.webp':'webp'}[suffix]
        matches = {'png':raw.startswith(b'\x89PNG\r\n\x1a\n'), 'jpeg':raw.startswith(b'\xff\xd8\xff'),
                   'webp':raw.startswith(b'RIFF') and raw[8:12]==b'WEBP'}
        if not matches[expected]:
            raise ValueError('图片内容与扩展名不符，或图片已损坏。')
        if expected in {'png', 'jpeg'} and len(raw) > 5*1024*1024:
            raise ValueError('原图超过 5MB 视觉输入限额，未压缩或生成视觉 QA。')
        try:
            # Read native image dimensions, not DPI-dependent page dimensions.
            # image_profile is broken for byte streams in bundled PyMuPDF 1.26.5.
            with pymupdf.open(stream=raw, filetype=expected) as image_doc:
                profile = image_doc[0].get_image_info()[0]
        except (RuntimeError, ValueError, IndexError, pymupdf.mupdf.FzErrorBase) as error:
            raise ValueError('图片无法读取，请提供完整有效的 PNG/JPG/WebP 文件。') from error
        if profile['width'] * profile['height'] > 40_000_000:
            raise ValueError('图片像素过大，未自动缩小或裁剪。')
        if expected in {'png', 'jpeg'}:
            # Preserve resolution, transparency, color and EXIF; no page rendering.
            out = folder / ('image-0001.png' if expected == 'png' else 'image-0001.jpg')
            out.write_bytes(raw)
        else:
            # Retain the existing WebP compatibility path; its original is also kept.
            with pymupdf.open(path) as image_doc:
                out = folder / 'image-0001.png'
                image_doc[0].get_pixmap(alpha=False).save(out)
        asset = store.add_asset(doc['id'], out, 0, '完整原始图片' if expected != 'webp' else '完整图片的 PNG 兼容副本')
        add('Image evidence. Determine language from the actual image.', 0, 1, str(out), asset, 'image-0001',
            evidence_kind='standalone_image', width=profile['width'], height=profile['height'])
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
