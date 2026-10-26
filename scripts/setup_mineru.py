"""Prepare the isolated MinerU environment used by scripts/mineru_layout.py.

    python scripts/setup_mineru.py                 # basic 版面模型（约 0.8 GB）
    python scripts/setup_mineru.py --tier both     # 再加 standard 主模型（约 1.2 GB）

MinerU is deliberately installed in its own virtualenv under ``runtime/``: it
needs ONNX Runtime, llama.cpp and its models, none of which the service itself
loads. The models are not committed to git; every machine downloads them once,
or copies ``~/.mineru/models`` from a machine that already has them.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = {'basic': ['MINERU_4_MODELS_ONNX'],
         'standard': ['MINERU_2_5_PRO_2605_1_2B_GGUF'],
         'both': ['MINERU_4_MODELS_ONNX', 'MINERU_2_5_PRO_2605_1_2B_GGUF']}
VERSION = '4.0.12'


def run(command, **kwargs):
    print('+', ' '.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='安装 MinerU 与版面模型')
    parser.add_argument('--venv', type=Path, default=ROOT / 'runtime/mineru-venv')
    parser.add_argument('--tier', choices=sorted(REPOS), default='basic')
    parser.add_argument('--skip-install', action='store_true', help='只补模型，不再装包')
    args = parser.parse_args()
    python = args.venv / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.is_file():
        run([sys.executable, '-m', 'venv', args.venv])
    if not args.skip_install:
        run([python, '-m', 'pip', 'install', '--upgrade', 'pip'])
        run([python, '-m', 'pip', 'install', f'mineru=={VERSION}'])
    names = ', '.join(repr(name) for name in REPOS[args.tier])
    code = ('from mineru.model.download import download_model_repo\n'
            'from mineru.model import registry\n'
            f'for name in [{names}]:\n'
            "    print(download_model_repo(getattr(registry, name), source='modelscope'))\n")
    run([python, '-c', code])
    print(f'\n完成。解析时把解释器指向：\n  {python}\n'
          f'例如：python scripts/mineru_layout.py --mineru-python "{python}" --list pdfs.txt')


if __name__ == '__main__':
    main()
