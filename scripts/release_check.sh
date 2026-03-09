#!/usr/bin/env sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ -d "$ROOT_DIR/.venv/bin" ]; then
  PATH="$ROOT_DIR/.venv/bin:$PATH"
fi

echo "[1/4] Running preflight checks"
make preflight

echo "[2/4] Running backend tests"
make test

echo "[3/4] Running frontend lint"
make frontend-lint

echo "[4/4] Running clean frontend build"
rm -rf frontend/.next
make frontend-build

echo "Release check completed successfully."