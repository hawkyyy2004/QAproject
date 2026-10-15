"""Offline ONNX embedding subprocess; never imported by PDF processing."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .semantic import cache_connection, topic_text
from .store import digest
from .settings import ROOT


class LocalEmbedder:
    def __init__(self, directory):
        # Imported only when a new vector is actually needed, inside the child.
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np = np
        directory = Path(directory)
        manifest = json.loads((ROOT / 'config/embedding-model.json').read_text())
        for name in ('model.onnx', 'tokenizer.json'):
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest['files'][name]['sha256']:
                raise ValueError('ModelHashMismatch')
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(directory / 'model.onnx'), sess_options=options,
                                            providers=['CPUExecutionProvider'])

    def encode(self, texts):
        np = self.np
        tokens = self.tokenizer.encode_batch(texts)
        values = {'input_ids': [t.ids for t in tokens], 'attention_mask': [t.attention_mask for t in tokens],
                  'token_type_ids': [t.type_ids for t in tokens]}
        inputs = {x.name: np.asarray(values[x.name], dtype=np.int64) for x in self.session.get_inputs()}
        # BGE uses CLS pooling, not mean pooling. No asymmetric retrieval prefix
        # for this symmetric short-topic similarity task.
        output = self.session.run(None, inputs)[0]
        vectors = output[:, 0] if output.ndim == 3 else output
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if vectors.shape[1] != 512 or not np.isfinite(vectors).all() or (norms == 0).any():
            raise ValueError('InvalidEmbedding')
        return (vectors / norms).astype('<f4')


def build_index(job, encoder_factory=LocalEmbedder):
    import numpy as np
    model_id, catalog = job['model_id'], job['catalog']
    topics, queries = job['topics'], job['queries']
    texts = [topic_text(t) for t in topics] + [q['text'] for q in queries]
    hashes = [digest([model_id, text]) for text in texts]
    vectors = {}
    with cache_connection(job['cache']) as db:
        for h in hashes:
            row = db.execute('SELECT vector FROM vectors WHERE hash=?', (h,)).fetchone()
            if row and len(row['vector']) == 512 * 4:
                vector = np.frombuffer(row['vector'], dtype='<f4')
                if np.isfinite(vector).all() and abs(float(np.linalg.norm(vector))-1) < 0.01:
                    vectors[h] = vector
    missing = list(dict((h, text) for h, text in zip(hashes, texts) if h not in vectors).items())
    if missing:
        encoder = encoder_factory(job['model_dir'])
        for start in range(0, len(missing), 2):
            batch = missing[start:start+2]
            encoded = encoder.encode([text for _, text in batch])
            with cache_connection(job['cache']) as db:
                for (h, _), vector in zip(batch, encoded):
                    vectors[h] = vector
                    db.execute('INSERT OR REPLACE INTO vectors VALUES(?,?)', (h, vector.astype('<f4').tobytes()))
        del encoder
    matrix = np.stack([vectors[h] for h in hashes[:len(topics)]])
    keys = ['t:' + t['id'] for t in topics] + [q['key'] for q in queries]
    for key, h in zip(keys, hashes):
        with cache_connection(job['cache']) as db:
            if db.execute('SELECT 1 FROM rankings WHERE key=? AND catalog=?', (key, catalog)).fetchone():
                continue
            scores = matrix @ vectors[h]
            db.execute('INSERT OR REPLACE INTO rankings VALUES(?,?,?)', (key, catalog, scores.astype('<f4').tobytes()))
    with cache_connection(job['cache']) as db:
        db.execute("INSERT OR REPLACE INTO state VALUES('catalog',?)", (catalog,))
        db.execute("DELETE FROM state WHERE key='error'")
        db.execute('DELETE FROM rankings WHERE catalog!=?', (catalog,))
        db.execute("DELETE FROM rankings WHERE key LIKE 'q:%' AND key NOT IN (SELECT key FROM queries)")
        # Retain current topic and recent query vectors only. An interrupted job
        # may leave reusable vectors, never a falsely completed catalog.
        keep = {digest([model_id, topic_text(t)]) for t in topics}
        keep.update(digest([model_id, r['text']]) for r in db.execute('SELECT text FROM queries'))
        for row in db.execute('SELECT hash FROM vectors').fetchall():
            if row['hash'] not in keep:
                db.execute('DELETE FROM vectors WHERE hash=?', (row['hash'],))


def main():
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    started = time.monotonic()
    try:
        build_index(job)
    except Exception as error:
        with cache_connection(job['cache']) as db:
            db.execute("INSERT OR REPLACE INTO state VALUES('error',?)", (type(error).__name__,))
            db.execute('INSERT INTO attempts(created,seconds,status,error_type) VALUES(?,?,?,?)',
                       (time.time(), round(time.monotonic()-started, 3), 'failed', type(error).__name__))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
