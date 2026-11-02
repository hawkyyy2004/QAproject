"""Print recent workbench events for quick local diagnosis."""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import qaproject  # noqa: F401  # installs GBK-safe stdout on Windows consoles
from qaproject.settings import runtime_directory

KEYS = ('error', 'stage', 'seconds', 'finish_reason', 'reasons', 'candidates',
        'pairs', 'duplicates', 'questions', 'error_type', 'attempt')


def summarize(detail):
    text = detail or ''
    try:
        data = json.loads(text)
    except ValueError:
        return text[:180]
    if not isinstance(data, dict):
        return text[:180]
    parts = [f'{key}={data[key]}' for key in KEYS if key in data]
    return ' '.join(parts) if parts else text[:180]


def main():
    parser = argparse.ArgumentParser(description='查看最近的工作台事件日志')
    parser.add_argument('-n', type=int, default=30, help='显示条数，默认 30')
    parser.add_argument('--kind', default='', help='按事件类型过滤，如 run_failed')
    parser.add_argument('--doc', default='', help='按文件 ID 前缀过滤')
    args = parser.parse_args()
    db_path = runtime_directory() / 'live' / 'state.sqlite3'
    if not db_path.is_file():
        print('未找到事件库:', db_path)
        return
    con = sqlite3.connect(db_path)
    rows = con.execute(
        'SELECT created,document_id,kind,detail FROM events '
        'WHERE kind LIKE ? AND document_id LIKE ? ORDER BY created DESC LIMIT ?',
        (f'%{args.kind}%' if args.kind else '%', f'{args.doc}%' if args.doc else '%', args.n))
    found = False
    for created, doc, kind, detail in rows:
        found = True
        stamp = time.strftime('%m-%d %H:%M:%S', time.localtime(created))
        print(f"{stamp} {(doc or '-')[:8]} {kind} {summarize(detail)}")
    if not found:
        print('没有匹配的事件；可用 --kind run_failed 这类参数换过滤条件。')


if __name__ == '__main__':
    main()
