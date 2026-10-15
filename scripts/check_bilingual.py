"""Small real-company-GLM check using isolated synthetic evidence; never commits QA.

Optional comparison measures generation/review on the same frozen evidence, without caches.
It is not a full-document or statistically reliable performance benchmark.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

EVIDENCE = '''Synthetic coastal observation protocol, for isolated software testing only.
Temperature sensors are calibrated against a reference thermometer before deployment to detect
systematic measurement bias; calibration does not remove changes in seawater temperature.
At synthetic station A, comparisons between temperature observations require the same sampling
depth and time window, because changing either can confound the comparison.
Missing salinity observations are marked as missing, rather than replaced with zero; zero would
be interpreted as an observed value and could distort an average.
An instrument's resolution describes the smallest displayed increment. It does not establish
measurement accuracy, which must be checked against an appropriate reference.
A short temperature record from synthetic station A describes that location and observation
window; it alone does not establish a long-term temperature trend across the entire ocean.'''

SINGLE_GENERATE = '''返回 {"qas":[{"unit_id":"给定单元ID","topic":"给定主题名称",
"input":"中文问题","thinking":"中文简短依据摘要","output":"中文答案","language":"zh"}],"skipped":[]}。
仅生成中文，qas 不超过 target，每条对应不同可靠知识点。问题写清对象和范围，依据一句话不超过60字，答案不超过200字。
只依据一个单元，不补齐、不重复 existing_questions，不输出路径。'''
SINGLE_REVIEW = '''返回 {"checks":[{"candidate_id":"原候选ID","supported":true,"independent":true,
"scope_correct":true,"knowledge_value":true,"distinct":true,"topic_correct":true,"note":"简短具体核查说明"}]}。
逐条对照各自原始证据审核事实、范围、价值和重复；任何不满足项填 false，不默认通过。'''


def main():
    from qaproject import pipeline
    from qaproject.glm import GLMClient
    from qaproject.settings import settings, api_key, source_ref
    from qaproject.store import Store
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',type=int,choices=range(1,6),default=3)
    parser.add_argument('--compare-single',action='store_true')
    parser.add_argument('--image',action='store_true',help='将合成证据渲染为真实图片，验证双语看图生成和审核')
    args=parser.parse_args()
    if not api_key():
        print(json.dumps({'status':'not_configured','message':'请在本机 .env.local 配置公司凭据。'},ensure_ascii=False))
        return 1
    folder=ROOT/'runtime/bilingual-check'/str(time.time_ns())
    store=Store(folder/'workspace')
    if args.image:
        import pymupdf
        with pymupdf.open() as pdf:
            page=pdf.new_page()
            if page.insert_textbox(pymupdf.Rect(40,40,555,800),EVIDENCE,fontsize=12)<0:
                raise ValueError('合成证据图片文字未完整放入，停止测试。')
            raw=page.get_pixmap(dpi=144,alpha=False).tobytes('png')
        doc_id,_=store.import_file('synthetic-observation.png',raw,'live')
    else:
        doc_id,_=store.import_file('synthetic-observation.txt',EVIDENCE.encode(),'live')
    cfg=dict(settings(),target_per_document=args.pairs,min_per_document=args.pairs)
    pipeline.settings=lambda:cfg
    worker=pipeline.Worker(store)
    worker.process(store.claim(worker.name,document_id=doc_id))
    doc=store.document(doc_id)
    report={'synthetic':True,'cached':False,'evidence_kind':'image' if args.image else 'text',
            'document_id':doc_id,'model':cfg['vision_model'] if args.image else cfg['text_model'],
            'bilingual':{'status':doc['status'],'pairs':doc['bilingual_count'],'rows':len(doc['candidates']),
                         'committed':doc['qa_count'],'seconds':doc['seconds'],'timings':doc['timings'],
                         'review_passed_pairs':len({c['pair_id'] for c in doc['candidates'] if c['status']=='pending'}),
                         'error':doc['error']}}
    with store.connect() as db:
        report['bilingual']['calls']=[json.loads(r[0]) for r in db.execute(
            "SELECT detail FROM events WHERE kind IN ('model_call','model_error') ORDER BY created")]
    if args.compare_single and doc['candidates']:
        client=GLMClient(cfg)
        unit=doc['units'][0]
        topic=doc['topics'][0]
        rows=[]
        calls=[]
        started=time.perf_counter()
        remaining=args.pairs
        try:
            while remaining>0:
                target=min(3,remaining)
                result,stats=client.call(SINGLE_GENERATE,{'units':pipeline.compact_units([unit]),
                    'topics':[{'name':topic['name'],'unit_ids':[unit['id']]}],'target':target,
                    'existing_questions':[c['qa']['input'] for c in rows]},[unit],max_tokens=4096)
                calls.append(dict(stats,stage='generation'))
                generated=[]
                for q in result.get('qas',[])[:target]:
                    if q.get('unit_id')!=unit['id'] or q.get('topic')!=topic['name']:
                        raise ValueError('单语对照引用无效。')
                    row={k:q[k] for k in ('input','thinking','output','language')}
                    row.update(source=source_ref(doc['path'],unit['locator'],topic['name']),image=unit['image'])
                    pipeline.validate_row(row,unit,topic,doc['path'])
                    generated.append({'candidate_id':str(len(rows)+len(generated)),'unit_id':unit['id'],'qa':row})
                if not generated:break
                result,stats=client.call(SINGLE_REVIEW,{'candidates':generated,'units':pipeline.compact_units([unit]),
                    'compare_questions':[c['qa']['input'] for c in rows]},[unit],max_tokens=2048)
                calls.append(dict(stats,stage='review'))
                expected={c['candidate_id'] for c in generated}
                checks=result.get('checks',[])
                by_id={c.get('candidate_id'):c for c in checks if isinstance(c,dict)}
                passed=sum(all(by_id.get(cid,{}).get(k) is True for k in pipeline.CHECKS) for cid in expected)
                rows.extend(generated)
                remaining-=len(generated)
                if passed<len(generated):
                    raise ValueError('单语对照存在未通过审核项，耗时不可视为相同质量产量。')
            report['single']={'status':'measured','rows':len(rows),'seconds':round(time.perf_counter()-started,3),'calls':calls}
        except ValueError as error:
            report['single']={'status':'failed','seconds':round(time.perf_counter()-started,3),'error':str(error),'calls':calls}
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'report':str(folder/'report.json'),**report},ensure_ascii=False,indent=2))
    return 0 if doc['status']=='waiting_review' and doc['bilingual_count']==args.pairs and report['bilingual']['review_passed_pairs']==args.pairs else 1


if __name__=='__main__':
    raise SystemExit(main())
