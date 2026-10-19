"""Local-only evidence coverage and request-size check; does not call GLM."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from check_pdf_extraction import synthetic
    import pymupdf
    from qaproject.glm import model_unit_refs
    from qaproject.parsers import extract, model_text
    from qaproject.pipeline import batches, compact_units
    from qaproject.settings import settings
    from qaproject.store import Store

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-parser',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('qaproject._baseline_parser',args.baseline_parser)
    baseline=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    folder=ROOT/'runtime/long-evidence-check'/str(time.time_ns())
    folder.mkdir(parents=True)
    report={'scope':'Local preparation and dry request plan only; no API calls or measured API tokens.',
            'baseline_sha256':hashlib.sha256(args.baseline_parser.read_bytes()).hexdigest(),
            'parser_sha256':hashlib.sha256((ROOT/'src/qaproject/parsers.py').read_bytes()).hexdigest(),'cases':{}}

    def plan(units):
        covered={u['companion_of'] for u in units if u.get('companion_of')}
        groups=list(batches(u for u in units if not u['image']))+list(batches(
            u for u in units if u['image'] and u['id'] not in covered))
        payloads=[]
        for group in groups:
            refs={u['id']:f'u{i}' for i,u in enumerate(group,1)}
            payloads.append(model_unit_refs({'units':compact_units(group)},refs))
        serialized=json.dumps(payloads,ensure_ascii=False,separators=(',',':'))
        return {'analysis_batches':len(groups),'analysis_text_chars':sum(len(model_text(u)) for g in groups for u in g),
                'analysis_payload_sha256':hashlib.sha256(serialized.encode()).hexdigest(),
                'analysis_payload_bytes':len(serialized.encode()),
                'analysis_images':sum(bool(u['image']) for g in groups for u in g)}

    def run(raw,label,name,version,trial):
        store=Store(folder/name/f'{label}-{trial}')
        doc_id,_=store.import_file('synthetic.pdf',raw,'live')
        doc=store.document(doc_id)
        start=time.perf_counter()
        try:
            units,pages=version(store,doc,settings())
        except ValueError as error:
            return {'ok':False,'error':str(error),'seconds':round(time.perf_counter()-start,6)}
        store.update_document(doc_id,units=units,pages=pages)
        seconds=round(time.perf_counter()-start,6)
        with pymupdf.open(stream=raw,filetype='pdf') as pdf:
            for index,page in enumerate(pdf,1):
                expected=page.get_text('text',sort=True).strip()
                views=[u['text'] for u in units if u['page']==index and not u['image']]
                if views:
                    assert ''.join(views)==expected, 'Text was omitted, duplicated or reordered'
        return {'ok':True,'seconds':seconds,'units':len(units),'images':sum(bool(u['image']) for u in units),**plan(units)}

    cases={'text_120_pages':synthetic(False),'mixed_180_pages':synthetic(True)}
    for graphic in (False,True):
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=1000,height=1400)
            line='Station Alpha measured temperature of 15 degrees Celsius with calibrated instruments. '*3
            page.insert_text((20,20),'\n'.join([line]*180)+'\nFINAL PAGE EVIDENCE.',fontsize=4)
            if graphic:
                page.draw_circle((700,1100),60,fill=(.2,.6,.8))
            cases['dense_graphic_page' if graphic else 'dense_text_page']=pdf.tobytes()
    for name,raw in cases.items():
        results={'before':[],'after':[]}
        for trial in range(3):
            for label in (('before','after') if trial%2==0 else ('after','before')):
                results[label].append(run(raw,label,name,baseline.extract if label=='before' else extract,trial))
        case={'runs':results,'median_seconds':{k:statistics.median(r['seconds'] for r in v) for k,v in results.items()}}
        if all(r['ok'] for v in results.values() for r in v):
            first,second=results['before'][0],results['after'][0]
            case['same_analysis_payload']=first['analysis_payload_sha256']==second['analysis_payload_sha256']
            case['same_analysis_batches']=first['analysis_batches']==second['analysis_batches']
        report['cases'][name]=case
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'case':name,**{k:v for k,v in case.items() if k!='runs'}},ensure_ascii=False),flush=True)
    print(folder/'report.json',flush=True)


if __name__=='__main__':
    main()
