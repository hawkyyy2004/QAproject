"""Independent short categories retain separate evidence scopes and reuse across files."""
from qaproject.glm import GLMClient
from qaproject.pipeline import Worker, CHECKS, BILINGUAL_CHECKS, approve
from qaproject.store import Store
from test_bilingual import pair
from test_document_topics import evidence, proposal


def test_independent_categories_reuse_across_files_without_extra_requests(tmp_path, monkeypatch):
    store = Store(tmp_path/'live')
    samples = [
        ([evidence(1,'锂电池的聚合物电解质合成记录，测试温度15摄氏度。'),
          evidence(2,'锂电池电极的电化学合成记录，测试温度15摄氏度。')],
         {'电池':[0,1], '聚合物':[0], '电化学':[1]}),
        ([evidence(3,'钠电池的电化学合成记录，测试温度15摄氏度。')],
         {'电池':[0], '电化学':[0]}),
    ]
    docs, calls = [], []
    active = None

    def call(self, instruction, payload, units=(), **kwargs):
        if 'existing_topics' in payload:
            calls.append('analysis')
            all_units, support = active
            # Each axis is directly returned by the original analysis call.
            return dict(summary='隔离材料测试。', topics=[dict(proposal(name,all_units[indexes[0]],keywords=[name+'检索词']),
                unit_ids=[all_units[i]['id'] for i in indexes]) for name,indexes in support.items()]), {}
        if 'candidates' in payload:
            calls.append('review')
            assert {c['unit_id'] for c in payload['candidates']}=={u['id'] for u in units}
            return dict(checks=[dict(candidate_id=c['candidate_id'],note='隔离模拟审核。',
                **{k:True for k in CHECKS+BILINGUAL_CHECKS}) for c in payload['candidates']]), {}
        calls.append('generation')
        allowed = active[1]
        all_units = active[0]
        for topic in payload['topics']:
            assert set(topic['unit_ids'])=={all_units[i]['id'] for i in allowed[topic['name']]}
        offset = len(payload['existing_questions'])//2
        generated = []
        for n in range(payload['target']):
            topic = payload['topics'][(offset+n)%len(payload['topics'])]
            generated.append(dict(pair(topic['unit_ids'][0],offset+n),topic=topic['name']))
        return dict(pairs=generated,skipped=[]), {}

    monkeypatch.setattr(GLMClient,'call',call)
    worker = Worker(store)
    for n, sample in enumerate(samples):
        active = sample
        doc_id, _ = store.import_file(f'isolated-{n}.txt',f'Synthetic material evidence {n}'.encode(),'live')
        store.update_document(doc_id,units=sample[0])
        worker.process(store.claim(worker.name,document_id=doc_id))
        doc = store.document(doc_id)
        assert doc['status']=='waiting_review' and doc['bilingual_count']==5, doc['error']
        assert {t['name'] for t in doc['topics']}==set(sample[1])
        assert 1<=len(doc['topics'])<=5
        for c in doc['candidates']:
            name = c['row']['source'].rsplit('|',1)[1]
            assert c['unit_id'] in {sample[0][i]['id'] for i in sample[1][name]}
            approve(store,c['id'],True,c['row_hash'])
        docs.append(doc_id)
    assert calls==['analysis','generation','review','generation','review']*2
    assert {d['id'] for d in store.search('电池')[0]['documents']}==set(docs)
    assert {d['id'] for d in store.search('电化学')[0]['documents']}==set(docs)
    assert {d['id'] for d in store.search('聚合物')[0]['documents']}=={docs[0]}
    assert all(store.document(doc_id)['qa_count']==10 for doc_id in docs)
    # The user's illustrative biological categories were not added as a fixed catalog.
    assert not {'鱼类','贝类','生物细胞'} & {t['name'] for t in store.topics()}
