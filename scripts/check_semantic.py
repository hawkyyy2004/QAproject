"""Real local-model smoke check; synthetic documents in an isolated runtime."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from qaproject.semantic import SemanticSearch, SearchCoordinator, cache_connection
from qaproject.store import Store


def main():
    root = ROOT / 'runtime/search-check' / str(time.time_ns())
    store = Store(root / 'live')
    examples = [
        ('海洋探测', '利用仪器探测海洋环境、海底结构及海水性质。'),
        ('海洋探索', '利用调查与探测手段认识海洋环境和海底未知区域。'),
        ('海洋观测', '海洋观测、遥感、监测和数据分析方法。'),
        ('海洋遥感', '利用卫星和雷达遥感观测海面环境与海洋过程。'),
        ('海洋声学', '利用声音传播研究海洋环境和水下声学探测。'),
        ('物理海洋', '海流、温度、盐度和海洋动力过程研究。'),
        ('海洋地质', '研究海底地形、海洋沉积物和海底地质构造。'),
        ('数学', '数学概念、关系、方法与应用。'),
        ('医学诊断', '疾病诊断、临床症状与医学检查方法。'),
    ]
    for name, definition in examples:
        topic = store.topic(name, definition, '仅隔离语义检索验收资料。')
        doc_id, _ = store.import_file(name+'.txt', ('合成测试：'+definition).encode(), 'live')
        store.link_topic(doc_id, topic['id'], [], [])
    with store.connect() as db:
        db.execute("UPDATE tasks SET status='succeeded'")
        db.execute("UPDATE documents SET status='completed'")
    service = SemanticSearch(store)
    coordinator = SearchCoordinator([service])
    assert service.model_available(), '先运行 scripts/setup_search_model.py'
    started = time.perf_counter()
    coordinator.run_job(service, service.next_job())
    initial_seconds = time.perf_counter()-started
    exact = service.recommend('海洋探测')
    names = [r['topic']['name'] for r in exact['recommendations']]
    assert exact['status'] == 'ready'
    assert '海洋探索' in names[:3] and '海洋观测' in names[:3], names
    assert '数学' not in names and '医学诊断' not in names and '海洋探测' not in names
    assert len(names) <= 5
    query = '海水探测'
    assert service.recommend(query)['status'] == 'pending'
    started = time.perf_counter()
    coordinator.run_job(service, service.next_job())
    new_query_seconds = time.perf_counter()-started
    fuzzy = service.recommend(query)
    assert fuzzy['status'] == 'ready' and fuzzy['recommendations']
    durations = []
    for _ in range(30):
        started = time.perf_counter()
        service.recommend(query)
        durations.append((time.perf_counter()-started)*1000)
    store.import_file('pending.txt', b'synthetic pending task', 'live')
    assert service.recommend('未缓存的全新表达')['status'] == 'busy'
    assert service.recommend(query)['status'] == 'ready'
    with cache_connection(service.path) as db:
        attempts_before = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    coordinator.run_job(service, service.next_job())
    with cache_connection(service.path) as db:
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == attempts_before
    assert coordinator.process is None
    report = {'model': service.model_id, 'synthetic_topics': len(examples),
              'initial_index_seconds': round(initial_seconds, 4),
              'new_query_job_seconds': round(new_query_seconds, 4),
              'cached_lookup_p50_ms': round(statistics.median(durations), 3),
              'cached_lookup_p95_ms': round(sorted(durations)[28], 3),
              'exact_query': '海洋探测', 'exact_recommendations': names,
              'new_query': query, 'new_query_recommendations': [r['topic']['name'] for r in fuzzy['recommendations']],
              'pending_pdf_defers_inference': True, 'cached_search_while_busy': True,
              'scope': '真实本地模型、隔离合成主题；不调用公司API，不含PDF端到端性能对照。'}
    output = root / 'report.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(output)


if __name__ == '__main__':
    main()
