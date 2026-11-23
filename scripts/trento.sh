#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
action="${1:-train}"
if (( $# )); then shift; fi
case "$action" in
  train|test) exec "${PYTHON:-python}" "${action}.py" --dataset trento "$@" ;;
  *) echo "Usage: bash scripts/trento.sh [train|test] [options]" >&2; exit 2 ;;
esac
