"""Explicit real-GLM check on isolated synthetic evidence; never commits QA."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from qaproject.glm import GLMClient, ProviderError, ANALYZE
    from qaproject.pipeline import Worker, compact_units
    from qaproject.settings import settings, api_key
    from qaproject.store import Store
    import pymupdf

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare',action='store_true',help='额外比较两轮2048/8192的分析调用，不使用响应缓存')
    args=parser.parse_args()
    if not api_key():
        print('API_KEY is not configured.'); return 1
    folder=ROOT/'runtime/request-recovery-check'/str(time.time_ns())
    folder.mkdir(parents=True)
    report={'synthetic':True,'calls':[],'cold':True}
    report_path=folder/'report.json'

    def save():
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

    def probe(label,instruction,payload,units=(),max_tokens=8192):
        record={'label':label,'max_tokens':max_tokens}
        try:
            result,stats=GLMClient(settings()).call(instruction,payload,units,max_tokens=max_tokens)
            record.update(ok=True,stats=stats)
            if label=='request-ceiling':
                record['ok']=result.get('ok') is True
            else:
                record['ok']=isinstance(result.get('summary'),str) and isinstance(result.get('topics'),list) and bool(result['topics'])
        except ProviderError as error:
            record.update(ok=False,code=error.code,stats=error.stats)
        report['calls'].append(record)
        save()
        print(json.dumps(record,ensure_ascii=False),flush=True)

    cfg=settings()
    probe('request-ceiling','只返回 {"ok":true}。',{'test':'验证输出允许上限，不要求长输出'},
          max_tokens=cfg['max_output_tokens_ceiling'])
    # Build genuine image-only pages: no hidden PDF text layer is available to GLM.
    visual=[]
    pdf=pymupdf.open()
    passages=[
        'Synthetic offshore observation station Alpha measured water temperature hourly\n'
        'during January 2025. Its mean was 15 degrees Celsius; the range was 12 to 18.\n'
        'Hourly sampling captures short-term variability, but this single-station\n'
        'record cannot establish a regional climate trend. Monthly averaging hides\n'
        'the timing of short temperature peaks. No causal mechanism was tested.',
        'Synthetic offshore observation station Beta measured salinity in January 2025.\n'
        'Its mean was 32 PSU; the range was 30 to 34 PSU. Salinity describes dissolved\n'
        'salt concentration. Beta used sensor calibration before deployment to reduce\n'
        'systematic measurement error. This station record does not establish a\n'
        'regional trend. Temperature and salinity affect seawater density, with\n'
        'other conditions held constant. No regional causal effect was measured.',
    ]
    for index,text in enumerate(passages,1):
        drawing=pymupdf.open()
        page=drawing.new_page(width=760,height=360)
        page.insert_text((25,50),text,fontsize=13)
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5))
        image=folder/f'synthetic-{index}.png'
        pix.save(image)
        target=pdf.new_page(width=760,height=360)
        target.insert_image(target.rect,stream=image.read_bytes())
        visual.append({'id':f'probe-{index}','image':str(image),'text':'Read actual scanned page.', 'language':'en'})
        drawing.close()
    source=folder/'synthetic-scanned.pdf'
    pdf.save(source)
    pdf.close()

    if args.compare:
        text_units=[dict(u,image='',text=text) for u,text in zip(visual,passages)]
        for index,budget in enumerate([2048,8192,8192,2048],1):
            probe(f'text-budget-{index}',ANALYZE,{'units':compact_units(text_units),'existing_topics':[]},text_units,budget)
        comparison={}
        for budget in (2048,8192):
            rows=[c for c in report['calls'] if c['label'].startswith('text-budget') and c['max_tokens']==budget]
            comparison[budget]={'successful':sum(c['ok'] for c in rows),'attempts':len(rows),
                                'median_seconds':statistics.median(c['stats']['seconds'] for c in rows)}
        report['analysis_comparison']=comparison
        save()
    probe('two-real-images',ANALYZE,{'units':compact_units(visual),'existing_topics':[]},visual,cfg['analysis_max_tokens'])
    store=Store(folder/'workspace')
    doc_id,_=store.import_file(source.name,source.read_bytes(),'live')
    worker=Worker(store)
    worker.process(store.claim(worker.name,document_id=doc_id))
    doc=store.document(doc_id)
    with store.connect() as db:
        history=[dict(kind=r[0],detail=json.loads(r[1])) for r in db.execute(
            "SELECT kind,detail FROM events WHERE kind IN ('model_call','model_error','request_split','token_budget_retry','run_failed') ORDER BY created")]
    report['pipeline']={'status':doc['status'],'error':doc['error'],'seconds':doc['seconds'],'timings':doc['timings'],
                        'coverage':doc['coverage'],'pairs':doc['bilingual_count'],'candidates':len(doc['candidates']),
                        'committed':doc['qa_count'],'model_approved_pairs':sum(c['status']=='pending' and c['row']['language']=='zh' for c in doc['candidates']),
                        'events':history}
    save()
    print(json.dumps({'report':str(report_path),'pipeline':report['pipeline']},ensure_ascii=False),flush=True)
    return 0 if doc['status']=='waiting_review' and report['calls'][0]['ok'] else 1


if __name__=='__main__':
    raise SystemExit(main())
