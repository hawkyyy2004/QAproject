from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ['input', 'thinking', 'output', 'source', 'image', 'language']
MIN_DOCUMENT_TOPICS = 1
MAX_DOCUMENT_TOPICS = 5
MAX_DOCUMENT_SEARCH_TERMS = 10


def image_media_type(path):
    """Media types for registered visual assets, including frozen historical PNGs."""
    suffix = Path(path).suffix.lower()
    if suffix in {'.jpg', '.jpeg'}:
        return 'image/jpeg'
    if suffix == '.png':
        return 'image/png'
    raise ValueError('图片资产格式不受支持。')


def settings():
    return json.loads((ROOT / 'config/settings.json').read_text(encoding='utf-8'))


def api_key():
    """Read only the authorized credential variable; never evaluate shell code."""
    path = ROOT / '.env.local'
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            name, sep, value = line.strip().partition('=')
            if sep and name == 'GLM_API_KEY':
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in '\"\'':
                    value = value[1:-1]
                if value:
                    return value
    return os.environ.get('GLM_API_KEY', '').strip()


def escaped(value):
    for old, new in [('%', '%25'), ('|', '%7C'), ('\r', '%0D'), ('\n', '%0A')]:
        value = value.replace(old, new)
    return value


def source_ref(path, locator, topic):
    return '|'.join(escaped(str(x)) for x in [path, locator, topic])
