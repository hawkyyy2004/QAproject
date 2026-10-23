"""Back up / restore persistent work data while the application is stopped."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qaproject.persistence import backup_runtime, restore_runtime
from qaproject.settings import runtime_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['backup','restore'])
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--data-dir',type=Path)
    args = parser.parse_args()
    try:
        runtime = runtime_directory(args.data_dir)
        result = backup_runtime(runtime,args.snapshot) if args.action=='backup' else restore_runtime(args.snapshot,runtime)
    except (ValueError,OSError) as error:
        parser.exit(1,str(error)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
