"""Operation-scoped reuse only; never persist verification across requests."""
from contextvars import ContextVar
from functools import wraps

from .source_files import original_path, signature
from .persistence import file_hash

_active = ContextVar('knowledge_read', default=None)


def cached(key, load):
    cache = _active.get()['cache']
    if key not in cache:
        cache[key] = load()
    return cache[key]


def read_session(function):
    @wraps(function)
    def wrapped(store, *args, **kwargs):
        current = _active.get()
        if current is not None and current['store'] is store:
            return function(store, *args, **kwargs)
        state = dict(store=store, cache={}, sources={})
        token = _active.set(state)
        try:
            result = function(store, *args, **kwargs)
            # Detect replacement/change during the operation, including after hash reuse.
            try:
                for doc, path, stamp in state['sources'].values():
                    if original_path(doc, store.root) != path or signature(path) != stamp:
                        raise ValueError('原件在读取期间变更')
            except (OSError, ValueError):
                raise ValueError('原件在读取期间变更或失效，请重试。') from None
            return result
        finally:
            _active.reset(token)
    return wrapped


def source_state(store, doc):
    state = _active.get()
    def load():
        try:
            path = original_path(doc, store.root)
            stamp = signature(path)
            valid = file_hash(path) == doc['content_hash']
            if signature(path) != stamp:
                raise ValueError('原件在读取期间变更')
            if valid:
                state['sources'][doc['id']] = (doc, path, stamp)
            return valid, str(path) if valid else doc['path']
        except (OSError, ValueError):
            return False, doc['path']
    return cached(('source', doc['id']), load)
