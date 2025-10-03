#!/bin/bash

TARGET_DIR="./workspace/d23"
mkdir -p "$TARGET_DIR"

BASE_URL="https://huggingface.co/datasets/AI-Art-Collab/dtasettar23/resolve/main/d23.tar."

(
  # Устанавливаем `set -e` внутри subshell, чтобы он завершился при первой ошибке curl
  set -e
  # Попробуем от 'a' до 'z' для первого символа суффикса
  for c1 in {a..z}; do
    # Попробуем от 'a' до 'z' для второго символа суффикса
    for c2 in {a..z}; do
      suffix="${c1}${c2}"
      url="${BASE_URL}${suffix}"
      echo "Fetching: $url" >&2
      # Качаем часть архива. --fail заставит curl завершиться с ошибкой, если файла нет.
      curl -LsS --fail "$url"
    done
  done
) 2>/dev/null | tar -xv -C "$TARGET_DIR" --wildcards '*.png'
#    └─ 1 ─┘   └────────── 2 ──────────┘ └─────────── 3 ───────────┘

echo "Extraction of PNG files finished. Check $TARGET_DIR"