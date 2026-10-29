"""Offline cache timing with isolated synthetic text; never calls GLM or edits live data."""
import argparse
import json
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def measure(callback):
    started=time.perf_counter()
    value=callback()
    return value,round(time.perf_counter()-started,6)


def implementation(args):
    sys.path.insert(0,str(Path(args.implementation_root)/'src'))
    from qaproject.semantic import init_cache, cache_connection, encode_texts
    from qaproject.embedding import build_index
    from qaproject.store import digest
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    model=Path(args.model_dir).resolve()
    texts=[f'隔离缓存测试问题{i}：观测站温度记录的测量单位是什么？' for i in range(args.questions)]
    measurements=[]
    for n in range(args.rounds):
        cache=output/f'real-{n}.sqlite3'
        init_cache(cache)
        vectors,cold=measure(lambda:encode_texts(model,'offline-check',cache,texts))
        assert len(vectors)==len(texts)
        _,warm=measure(lambda:encode_texts(model,'offline-check',cache,texts))
        build_index(dict(cache=str(cache),model_dir=str(model),model_id='offline-check',catalog='changed',
            topics=[dict(id='isolated',name='隔离主题',definition='缓存生命周期测试',aliases=[])],queries=[]))
        _,rebuilt=measure(lambda:encode_texts(model,'offline-check',cache,texts))
        measurements.append(dict(cold=cold,warm=warm,after_topic_rebuild=rebuilt))
    # A large warmed cache tests local reading/subprocess overhead, not model quality.
    cache=output/'warm-large.sqlite3'
    init_cache(cache)
    large=[f'隔离缓存问题{i}' for i in range(10000)]
    raw=struct.pack('<512f',1.,*([0.]*511))
    with cache_connection(cache) as db:
        table='qa_vectors' if db.execute("SELECT 1 FROM sqlite_master WHERE name='qa_vectors'").fetchone() else 'vectors'
        db.executemany(f'INSERT OR REPLACE INTO {table} VALUES(?,?)',[(digest(['offline-check',t]),raw) for t in large])
    large_times=[]
    for _ in range(args.rounds):
        vectors,seconds=measure(lambda:encode_texts(model,'offline-check',cache,large))
        assert len(vectors)==len(large)
        large_times.append(seconds)
    result=dict(questions=args.questions,rounds=args.rounds,real_model=measurements,
        median_seconds={k:statistics.median(r[k] for r in measurements) for k in measurements[0]},
        warmed_10000_seconds=large_times,warmed_10000_median=statistics.median(large_times),
        scope='Only local vector caching/encoding. No PDF parsing, GLM, review, commit or export timing.')
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root')
    parser.add_argument('--implementation-root',default=str(ROOT),help=argparse.SUPPRESS)
    parser.add_argument('--model-dir',default=str(ROOT/'runtime/models/bge-small-zh-v1.5'))
    parser.add_argument('--questions',type=int,default=256)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--output',default=str(ROOT/'runtime/dedup-cache-check'/str(time.time_ns())))
    args=parser.parse_args()
    if args.baseline_root:
        output=Path(args.output).resolve()
        reports={}
        for label,root in [('baseline',Path(args.baseline_root).resolve()),('current',ROOT)]:
            done=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--implementation-root',str(root),
                '--model-dir',str(Path(args.model_dir).resolve()),'--questions',str(args.questions),
                '--rounds',str(args.rounds),'--output',str(output/label)],check=True,capture_output=True,
                text=True,encoding='utf-8',errors='replace')
            reports[label]=json.loads(done.stdout)
        (output/'comparison.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(reports,ensure_ascii=False,indent=2))
    else:
        implementation(args)


if __name__=='__main__':
    main()
