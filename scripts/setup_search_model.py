"""Install the pinned public model locally; never send a document or API key."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import httpx
from qaproject.settings import settings


def main():
    spec = json.loads((ROOT / 'config/embedding-model.json').read_text())
    directory = ROOT / settings()['semantic_search']['model_dir']
    directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True, trust_env=False) as client:
        for name, item in spec['files'].items():
            path = directory / name
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
                print(name + ': 已验证')
                continue
            temporary = path.with_suffix(path.suffix + '.part')
            try:
                sha = hashlib.sha256()
                with client.stream('GET', item['source']) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as output:
                        for chunk in response.iter_bytes():
                            sha.update(chunk)
                            output.write(chunk)
                if sha.hexdigest() != item['sha256'] or temporary.stat().st_size != item['bytes']:
                    raise ValueError('模型校验不通过，未安装。')
                temporary.replace(path)
                print(name + ': 已下载并验证')
            finally:
                temporary.unlink(missing_ok=True)
    (directory / 'manifest.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    print('本地语义模型就绪；无需 Embedding API 密钥。')


if __name__ == '__main__':
    main()
