"""Isolated cold PDF preparation comparison; no GLM calls or live state changes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import pymupdf
from qaproject.parsers import extract
from qaproject.settings import settings
from qaproject.store import Store


def synthetic(mixed):
    with pymupdf.open() as pdf:
        for n in range(180 if mixed else 120):
            page = pdf.new_page()
            if not mixed or n % 6 < 3:
                remaining = page.insert_textbox((40, 40, 550, 720),
                                    (f'Observation {n+1}: water temperature was 15 degrees Celsius. '
                                     'The calibrated measurements cover only the stated location and period.\n')*18)
                assert remaining >= 0, 'Synthetic evidence must fit its text box'
            elif n % 6 == 3:
                page.insert_text((40, 60), f'Temperature {n+1}: 15 C')
            elif n % 6 == 5:
                page.insert_text((40, 60), f'Observation {n+1}: calibrated synthetic chart.')
                page.draw_rect((120, 200, 350, 400))
                for x, height in [(150, 70), (210, 130), (270, 170)]:
                    page.draw_rect((x, 400-height, x+30, 400), fill=(.2, .6, .8))
                page.insert_text((120, 425), 'Figure 1. Synthetic temperature comparison, degrees C.')
        return pdf.tobytes()


def measure(folder, raw, parser):
    store = Store(folder)
    doc_id, _ = store.import_file('sample.pdf', raw, 'live')
    doc = store.document(doc_id)
    started = time.perf_counter()
    units, pages = parser(store, doc, settings())
    store.update_document(doc_id, units=units, pages=pages)
    elapsed = time.perf_counter()-started
    # Verify that no readable text page disappeared, including the final page.
    by_page = {}
    for unit in units:
        by_page.setdefault(unit['page'], []).append(unit['text'])
    with pymupdf.open(stream=raw, filetype='pdf') as pdf:
        for n, page in enumerate(pdf, 1):
            text = page.get_text('text', sort=True).strip()
            if text and ''.join(text.split()) not in ''.join(''.join(by_page.get(n, [])).split()):
                raise AssertionError(f'Readable evidence missing on page {n}')
    images = list((store.root/'assets/images').rglob('*.png'))
    return {'preparation_seconds':round(elapsed, 6), 'pages':pages, 'units':len(units),
            'images':len(images), 'image_bytes':sum(p.stat().st_size for p in images),
            'text_files':len(list(store.root.rglob('*.txt')))+len(list(store.root.rglob('*.md')))}


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--baseline-parser', type=Path, required=True,
                      help='Saved pre-change parsers.py, retained only under runtime')
    args.add_argument('--pdf', type=Path, help='Optional actual paper, read only')
    args.add_argument('--rounds', type=int, default=3)
    options = args.parse_args()
    if not 1 <= options.rounds <= 10:
        args.error('rounds must be 1–10')
    spec = importlib.util.spec_from_file_location('qaproject._baseline_parser', options.baseline_parser)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    folder = ROOT/'runtime/pdf-image-check'/str(time.time_ns())
    folder.mkdir(parents=True)
    cases = {'text_120_pages':synthetic(False), 'mixed_180_pages':synthetic(True)}
    if options.pdf:
        cases['actual_paper'] = options.pdf.read_bytes()
    report = {'scope':'Cold local evidence preparation including asset/metadata writes; no AI, generation, review, commit or export timing.',
              'baseline_sha256':hashlib.sha256(options.baseline_parser.read_bytes()).hexdigest(),
              'parser_sha256':hashlib.sha256((ROOT/'src/qaproject/parsers.py').read_bytes()).hexdigest(),
              'cases':{}}
    for name, raw in cases.items():
        runs = {'before':[], 'after':[]}
        for n in range(options.rounds):
            # Alternate order. Each run imports into a new workspace with no reusable results.
            for label in (('before', 'after') if n % 2 == 0 else ('after', 'before')):
                runs[label].append(measure(folder/name/f'{label}-{n}', raw,
                                           baseline.extract if label == 'before' else extract))
        medians = {label:round(statistics.median(r['preparation_seconds'] for r in values), 6)
                   for label, values in runs.items()}
        report['cases'][name] = {'source_sha256':hashlib.sha256(raw).hexdigest(), 'runs':runs,
                                'median_seconds':medians,
                                'change_percent':round((medians['after']/medians['before']-1)*100, 2)}
        print(json.dumps({name:report['cases'][name]}, ensure_ascii=False), flush=True)
    path = folder/'report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(str(path))


if __name__ == '__main__':
    main()
