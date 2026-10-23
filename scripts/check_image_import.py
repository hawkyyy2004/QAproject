"""Compare cold standalone-image preparation; no company API calls or live data writes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pymupdf
from qaproject.parsers import extract
from qaproject.settings import settings
from qaproject.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-parser',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    args = parser.parse_args()
    if not 1<=args.rounds<=10:
        parser.error('rounds must be 1–10')
    spec = importlib.util.spec_from_file_location('qaproject._baseline_image_parser',args.baseline_parser)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    folder = ROOT/'runtime/image-import-check'/str(time.time_ns())
    folder.mkdir(parents=True)
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=1200,height=900)
        page.insert_text((60,80),'Synthetic ocean temperature observations, degrees Celsius.',fontsize=24)
        for x,height in [(150,150),(400,300),(650,500)]:
            page.draw_rect((x,750-height,x+150,750),fill=(.2,.5,.8))
        pix = page.get_pixmap()
    report = {'scope':'Cold local extraction, image asset writes and evidence persistence only; '
                      'excludes import, queue, model analysis/generation/review, commit and export.',
              'baseline_sha256':hashlib.sha256(args.baseline_parser.read_bytes()).hexdigest(),
              'current_sha256':hashlib.sha256((ROOT/'src/qaproject/parsers.py').read_bytes()).hexdigest(), 'cases':{}}
    for extension in ('png','jpeg'):
        raw = pix.tobytes(extension)
        runs = {'before':[],'after':[]}
        for n in range(args.rounds):
            for label in (('before','after') if n%2==0 else ('after','before')):
                store = Store(folder/extension/f'{label}-{n}')
                doc_id,_ = store.import_file('chart.'+extension,raw,'live')
                doc = store.document(doc_id)
                started = time.perf_counter()
                units,pages = (baseline.extract if label=='before' else extract)(store,doc,settings())
                store.update_document(doc_id,units=units,pages=pages)
                seconds = time.perf_counter()-started
                saved = Path(units[0]['image']).read_bytes()
                if label=='after':
                    assert saved==raw and pages==0 and units[0]['locator']=='image-0001'
                runs[label].append({'preparation_seconds':round(seconds,6),'image_bytes':len(saved),
                                    'original_bytes_preserved':saved==raw})
        report['cases'][extension] = {'source_bytes':len(raw),'runs':runs,
            'median_seconds':{label:round(statistics.median(r['preparation_seconds'] for r in values),6)
                              for label,values in runs.items()}}
    path = folder/'report.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'report':str(path),'cases':{name:case['median_seconds'] for name,case in report['cases'].items()}},ensure_ascii=False))


if __name__=='__main__':
    main()
