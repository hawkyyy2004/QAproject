"""MinerU layout preparation overlaps across files; PyMuPDF stays on one thread."""
import threading
import time

import pymupdf
import pytest

from qaproject import parsers
from qaproject.execution import ExecutionLimits, execution_config
from qaproject.pipeline import Worker
from qaproject.store import Store


class Tracker:
    """One shared counter, so a peak above one proves the calls really overlapped."""

    # Longer than the 0.25s poll interval, so the heartbeat path is exercised.
    def __init__(self, seconds=.3):
        self.lock = threading.Lock()
        self.now = self.peak = 0
        self.seconds = seconds

    def run(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(self.seconds)
        with self.lock:
            self.now -= 1
        return 'done'


def pdf_bytes(text='isolated evidence page'):
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40, 60), text)
        return pdf.tobytes()


def test_two_files_prepare_layout_at_the_same_time():
    tracker = Tracker()
    limits = ExecutionLimits(layouts=2)
    try:
        results = []
        threads = [threading.Thread(target=lambda: results.append(
            limits.layout(tracker.run, threading.Event(), lambda: None))) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
    finally:
        limits.close()
    assert tracker.peak == 2 and results == ['done', 'done']


@pytest.mark.parametrize('layouts,expected', [(1, 1), (2, 2)])
def test_layout_concurrency_setting_bounds_the_overlap(layouts, expected):
    tracker = Tracker()
    limits = ExecutionLimits(layouts=layouts)
    try:
        threads = [threading.Thread(target=lambda: limits.layout(
            tracker.run, threading.Event(), lambda: None)) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
    finally:
        limits.close()
    assert tracker.peak == expected


@pytest.mark.parametrize('value', [0, 3, '2', True])
def test_layout_concurrency_setting_is_validated(value):
    with pytest.raises(ValueError, match='layout_concurrency'):
        execution_config({'layout_concurrency': value})


def test_prepared_layout_is_not_built_twice(tmp_path, monkeypatch):
    """A failed MinerU attempt must not be paid for again inside the parser."""
    calls = []

    def prepare(store, doc, config):
        calls.append(doc['id'])
        return {}, '未找到可用的 MinerU 解释器，保留整页'

    monkeypatch.setattr(parsers.layout_cache, 'prepare', prepare)
    store = Store(tmp_path / 'live')
    doc_id, _ = store.import_file('paper.pdf', pdf_bytes(), 'live')
    doc = store.document(doc_id)
    config = {'pdf_layout': 'auto'}

    prepared = parsers.preheat(store, doc, config)
    assert prepared == ({}, '未找到可用的 MinerU 解释器，保留整页') and calls == [doc_id]

    units, _ = parsers.extract(store, doc, config, prepared=prepared)
    assert units and calls == [doc_id], 'the parser asked MinerU for the same manifest again'
    assert store.document(doc_id)['layout_note'].startswith('未找到可用的 MinerU 解释器')


def test_preheat_skips_work_when_layout_is_off(tmp_path):
    store = Store(tmp_path / 'live')
    doc_id, _ = store.import_file('paper.pdf', pdf_bytes(), 'live')
    assert parsers.preheat(store, store.document(doc_id), {'pdf_layout': 'off'}) is None


def test_two_workers_prepare_layout_once_each_and_in_parallel(tmp_path, monkeypatch):
    """End to end through Worker: one MinerU attempt per file, and they overlap."""
    from concurrent.futures import ThreadPoolExecutor
    from test_file_concurrency import provider

    tracker = Tracker(seconds=.3)
    seen = []
    lock = threading.Lock()

    def prepare(store, doc, config):
        with lock:
            seen.append(doc['id'])
        tracker.run()
        return {}, '未找到可用的 MinerU 解释器，保留整页'

    provider(monkeypatch)
    monkeypatch.setattr(parsers.layout_cache, 'prepare', prepare)
    limits = ExecutionLimits(documents=2, layouts=2)
    try:
        store = Store(tmp_path / 'parallel')
        docs = [store.import_file(f'{n}.pdf', pdf_bytes(f'isolated page {n}, measured value 15'), 'live')[0]
                for n in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: Worker(store, limits=limits).run_once(), docs))
    finally:
        limits.close()
    assert tracker.peak == 2, 'layout preparation did not overlap'
    assert sorted(seen) == sorted(docs), 'a file asked MinerU more than once'
    for doc_id in docs:
        result = store.document(doc_id)
        assert result['status'] == 'waiting_review', result['error']
