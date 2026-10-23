"""Bound one application's file work, provider requests and native parsing."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
import threading
import time


class WorkInterrupted(Exception):
    """Stop at a recoverable boundary; never abandon an in-flight HTTP request."""


def execution_config(config):
    result = {}
    for key in ('document_concurrency', 'model_request_concurrency'):
        value = config.get(key, 2)
        if type(value) is not int or not 1 <= value <= 2:
            raise ValueError(f'{key} 必须为1或2，修改后重启生效。')
        result[key] = value
    return result


class ExecutionLimits:
    def __init__(self, model_requests=2, documents=2):
        self.documents = threading.BoundedSemaphore(documents)
        self.requests = threading.BoundedSemaphore(model_requests)
        # PyMuPDF is not thread-safe. All application parsing uses one thread,
        # while other files can continue waiting for / consuming model results.
        self.parser = ThreadPoolExecutor(max_workers=1, thread_name_prefix='evidence-parser')
        self.local_embedding = threading.Lock()
        self.claim_lock = threading.Lock()
        self.active_documents = set()

    @staticmethod
    def check(stop):
        if stop.is_set():
            raise WorkInterrupted()

    def claim(self, store, worker):
        # Keep a live file exclusive even if its persisted lease expires before
        # a slow operation returns. Reclaim only after the old slot has exited.
        root = str(store.db)
        with self.claim_lock:
            excluded = [doc for db, doc in self.active_documents if db == root]
            task = store.claim(worker, excluded_documents=excluded)
            if task:
                self.active_documents.add((root, task['document_id']))
            return task

    def release(self, store, task):
        with self.claim_lock:
            self.active_documents.discard((str(store.db), task['document_id']))

    @contextmanager
    def request(self, stop, heartbeat):
        started = time.perf_counter()
        next_renewal = started
        self.check(stop)
        while not self.requests.acquire(timeout=.25):
            self.check(stop)
            if time.perf_counter() >= next_renewal:
                heartbeat()
                next_renewal = time.perf_counter() + 10
        try:
            self.check(stop)
            heartbeat()
            yield round(time.perf_counter() - started, 6)
        finally:
            self.requests.release()

    def parse(self, run, stop, heartbeat):
        self.check(stop)
        started = time.perf_counter()

        # Measure executor wait separately from preparation itself.
        def timed_prepare():
            nonlocal began
            self.check(stop)
            began = time.perf_counter()
            value = run()
            return value, round(time.perf_counter() - began, 6)

        began = started
        next_renewal = started
        future = self.parser.submit(timed_prepare)
        try:
            while True:
                try:
                    value, seconds = future.result(timeout=.25)
                    return value, round(began - started, 6), seconds
                except TimeoutError:
                    if future.done():
                        # Completion can race with the wait timeout. Retrieve
                        # the completed result (or its actual parser exception).
                        value, seconds = future.result()
                        return value, round(began - started, 6), seconds
                    self.check(stop)
                    if time.perf_counter() >= next_renewal:
                        heartbeat()
                        next_renewal = time.perf_counter() + 10
        except BaseException:
            # A running native call must finish before its file can be reclaimed.
            if not future.cancel():
                try:
                    future.result()
                except Exception:
                    pass
            raise

    def close(self):
        self.parser.shutdown(wait=True, cancel_futures=True)


# Direct Worker users (diagnostic scripts/tests) share the same safety bounds.
DEFAULT_LIMITS = ExecutionLimits()
