#!/bin/zsh
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo '请先按 README 安装项目依赖。'
  read '?按回车关闭'
  exit 1
fi
exec .venv/bin/python run.py
