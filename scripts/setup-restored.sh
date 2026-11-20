#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if command -v corepack >/dev/null; then
  corepack npm@11.6.2 ci
else
  npm ci
fi
uv venv --allow-existing .venv --python 3.14.4
uv pip install --python .venv/bin/python -r requirements.txt

restore_env() {
  local name=$1
  shift
  local directory=".runtime/$name-venv"
  uv venv --allow-existing "$directory" --python 3.12.16
  uv pip install --python "$directory/bin/python" --no-deps \
    -r "scripts/restore/requirements-$name.txt" "$@"
}

restore_env manga
restore_env comfy --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match
restore_env lora --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

if [[ ! -f "${XDG_CONFIG_HOME:-$HOME/.config}/comfy-folder-lora/config.json" ]]; then
  echo '学習・生成に必要な設定がありません。scripts/restore.py --with-user-state で復元してください。' >&2
  exit 1
fi

export PATH="$PWD/.runtime/lora-venv/bin:$PATH"
.runtime/comfy-venv/bin/python plugins/dsh-minimax-h3/native/build_kitchen.py
./dsh anima build
./dsh anima prepare
