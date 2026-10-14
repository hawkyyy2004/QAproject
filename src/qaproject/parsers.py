import csv
import io
from pathlib import Path
import re

import pymupdf

from .store import digest


def text_alternatives(units):
    """Keep frozen visual evidence and add a text view at the same true locator."""
    result = list(units)
    ids = {u['id'] for u in units}
    for unit in units:
        text = unit['text']
        readable = len(text)>=120 and (len(re.findall('[\u4e00-\u9fff]',text))>=30
                                      or len(re.findall('[A-Za-z]',text))>=80)
        if not unit['image'] or not readable or unit['id']+':text' in ids:
            continue
        alternative = dict(unit, id=unit['id']+':text', image='', asset_id='',
                           companion_of=unit['id'], evidence_kind='extracted_text')
        alternative.pop('hash', None)
        alternative['hash'] = digest(alternative)
        result.append(alternative)
        ids.add(alternative['id'])
    return result


def extract(store, doc, config, progress=None):
    path = Path(doc['path'])
    folder = store.root / 'assets/images' / doc['id']
    folder.mkdir(parents=True, exist_ok=True)
    units = []

    def add(text, page, seg, image='', asset_id='', locator=None):
        locator = locator or f'page-{page:04d}-seg-{seg:04d}'
        unit = {'id': f'{doc["id"]}:p{page}:s{seg}', 'text': text, 'page': page, 'seg': seg,
                'locator': locator, 'image': image, 'asset_id': asset_id,
                'language': 'zh' if re.search('[\u4e00-\u9fff]', text) else 'en'}
        unit['hash'] = digest(unit)
        units.append(unit)

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
                text = page.get_text('text', sort=True).strip()
                paragraphs = [x.strip() for x in re.split(r'\n\s*\n', text) if x.strip()]
                visual = bool(page.get_images()) or len(page.get_drawings()) > 2 or len(text) < 60
                image, asset = '', ''
                if visual:
                    out = folder / f'page-{n:04d}.png'
                    page.get_pixmap(dpi=144, alpha=False).save(out)
                    image = str(out)
                    asset = store.add_asset(doc['id'], out, n, f'第 {n} 页图文证据')
                    # Whole-page unit preserves the true relationship of chart, labels and caption.
                    add(text or 'Scan page: use actual visual evidence.', n, 1, image, asset)
                else:
                    for s, paragraph in enumerate(paragraphs, 1):
                        add(paragraph, n, s)
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
                add(paragraph.strip(), 0, s, locator=f'seg-{s:04d}')
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
        if len(unit['text']) > 16000:
            raise ValueError('完整证据单元过长，初版不截断；请拆分来源或调整准备方案。')
        if unit['image'] and Path(unit['image']).stat().st_size > 5*1024*1024:
            raise ValueError('图像超过初版 5MB 输入限额，未生成视觉 QA。')
    return text_alternatives(units), pages
