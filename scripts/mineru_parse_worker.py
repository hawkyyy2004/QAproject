"""Name one PDF and let MinerU return its layout as JSON.

This file runs inside the MinerU virtualenv, not the service environment:

    <mineru-python> scripts/mineru_parse_worker.py <pdf> <out.json> [tier] [page-range]

``scripts/mineru_layout.py`` calls it as a subprocess and turns the result into
a crop manifest; nothing here touches the service runtime.
"""
import json
import sys
import time
from pathlib import Path


def main():
    pdf, out = Path(sys.argv[1]), Path(sys.argv[2])
    tier = sys.argv[3] if len(sys.argv) > 3 else 'basic'
    page_range = sys.argv[4] if len(sys.argv) > 4 else ''
    from mineru.parser.mineru_parser import MinerUParser

    started = time.time()
    result = MinerUParser(tier=tier).parse(pdf, page_range=page_range)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.to_json(), encoding='utf-8')
    print(json.dumps({
        'file': pdf.name,
        'pages': len(result.pages),
        'seconds': round(time.time() - started, 1),
        'out': str(out),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
