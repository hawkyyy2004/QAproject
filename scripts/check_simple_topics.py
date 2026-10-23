"""Small real-company classification check on synthetic text, without generating QA."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from qaproject.glm import ANALYZE, GLMClient
    from qaproject.store import Store, classification_catalog, digest
    from qaproject.taxonomy import TopicSelection
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,choices=(1,2,3),default=3)
    args = parser.parse_args()
    folder = ROOT/'runtime/simple-topic-check'/str(time.time_ns())
    store = Store(folder/'isolated')
    samples = [
        ('synthetic-fish-genome', '隔离合成分类测试资料：石斑鱼染色体水平参考基因组的组装、基因注释及基因家族比较。关键词：石斑鱼；参考基因组；染色体；基因组组装。'),
        ('synthetic-fish-cells', '隔离合成分类测试资料：黑线鳕胚胎来源细胞系的建立，比较细胞培养条件及冷冻保存后的细胞活性。关键词：黑线鳕；细胞系；细胞培养；冷冻保存。'),
        ('synthetic-battery', '隔离合成分类测试资料：聚合物固态电解质用于锂电池，比较聚合物结构对离子传导及电化学稳定性的影响。关键词：锂电池；聚合物；固态电解质；离子传导。'),
    ]
    report = dict(synthetic=True, live_data_changed=False, qa_generated=0, classification_calls=0, runs=[])
    report_file = folder/'report.json'
    print('report='+str(report_file),flush=True)
    client = GLMClient()
    for name,text in samples[:args.limit]:
        doc_id,_ = store.import_file(name+'.txt',text.encode(),'live')
        unit = dict(id=doc_id+':s1',text=text,image='',asset_id='',locator='text-seg-0001',language='zh')
        unit['hash'] = digest(unit)
        store.update_document(doc_id,units=[unit])
        catalog = store.topics()
        payload = dict(analysis_phase='document',units=[{k:unit[k] for k in ('id','text','language')}],
                       existing_topics=classification_catalog(catalog),legacy_topics=[])
        task = store.claim('simple-topic-probe',document_id=doc_id)
        report['classification_calls'] += 1
        try:
            result,stats = client.call(ANALYZE,payload,[unit],max_tokens=1500)
            selection = TopicSelection(catalog,[unit])
            selection.add(result,[unit])
            selected,terms = selection.select()
            store.apply_topic_selection(doc_id,selected,terms,result['summary'],task['id'],'simple-topic-probe')
            store.task_update(task['id'],'simple-topic-probe','classified','succeeded')
            run = dict(name=name,topics=[t['name'] for t in selected],stats=stats)
        except ValueError as error:
            run = dict(name=name,error=str(error),stats=getattr(error,'stats',{}))
        report['runs'].append(run)
        report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(run,ensure_ascii=False),flush=True)
        if 'error' in run:
            return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
