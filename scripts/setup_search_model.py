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
    spec = json.loads((ROOT / 'config/embedding-model.json').read_text(encoding='utf-8'))
    directory = ROOT / settings()['semantic_search']['model_dir']
    directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True, trust_env=False) as client:
        for name, item in spec['files'].items():
            path = directory / name
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
                print(name + ': 已验证')
                continue
            download(client, item, name, path)
    (directory / 'manifest.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    print('本地语义模型就绪；无需 Embedding API 密钥。')


def download(client, item, name, path):
    """Try the pinned source first, then the mirror; both are hash checked.

    huggingface.co is unreachable from mainland China, so a fetch that only
    knows the canonical URL cannot install the model there at all.
    """
    temporary = path.with_suffix(path.suffix + '.part')
    errors = []
    for label, url in (('主源', item['source']), ('镜像', item.get('mirror'))):
        if not url:
            continue
        try:
            sha = hashlib.sha256()
            with client.stream('GET', url) as response:
                response.raise_for_status()
                with temporary.open('wb') as output:
                    for chunk in response.iter_bytes():
                        sha.update(chunk)
                        output.write(chunk)
            if sha.hexdigest() != item['sha256'] or temporary.stat().st_size != item['bytes']:
                raise ValueError('模型校验不通过，未安装。')
            temporary.replace(path)
            print(name + f': 已从{label}下载并验证')
            return
        except Exception as error:
            errors.append(f'{label} {type(error).__name__}: {error}')
        finally:
            temporary.unlink(missing_ok=True)
    raise SystemExit(name + ': 主源和镜像都不可用（' + '；'.join(errors) + '）')


if __name__ == '__main__':
    main()
