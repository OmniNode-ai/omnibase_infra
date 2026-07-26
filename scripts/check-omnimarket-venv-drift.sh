#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# check-omnimarket-venv-drift.sh (OMN-14060)
# ----------------------------------------------------------------------------
# Session/cron-tick companion to the in-process pre-flight guard
# (src/omnibase_infra/cli/omnimarket_drift_guard.py). The pre-flight guard is
# cheap and LOCAL-ONLY (compares the target venv's installed omnimarket commit
# against the ALREADY-CHECKED-OUT $OMNI_HOME/omnimarket clone -- no network,
# safe for every `onex skill` dispatch) — it detects and instructs, but never
# repairs.
#
# This script does the work the pre-flight intentionally skips: it refreshes
# the canonical clone from origin/dev (network), compares the resolved SHA
# against the target venv's installed commit, and — with --repair — re-runs
# the canonical co-install (install-node-skill-package.sh) to fix drift.
#
# Run this periodically (a session/cron tick), or by hand after the pre-flight
# guard raises "omnimarket venv is STALE". It never runs automatically on
# `onex skill` dispatch — the hot path only detects and instructs; this script
# is the actual repair.
#
# Usage:
#   scripts/check-omnimarket-venv-drift.sh [--repair] [PYTHON]
#     PYTHON  target venv python (default: $VIRTUAL_ENV/bin/python, else
#             ./.venv/bin/python)
#   Env:
#     OMNI_HOME  canonical repo registry root (required)
#
# Exit codes:
#   0  no drift (or drift found and successfully repaired with --repair)
#   1  drift detected and not repaired (either --repair was not passed, or
#      omnimarket is not installed / not a VCS install in the target venv)
# ----------------------------------------------------------------------------
set -euo pipefail

REPAIR=0
PYTHON_BIN=""
for arg in "$@"; do
  case "$arg" in
    --repair) REPAIR=1 ;;
    *) PYTHON_BIN="$arg" ;;
  esac
done

if [[ -z "${OMNI_HOME:-}" ]]; then
  echo "ERROR: OMNI_HOME is not set. Export OMNI_HOME=/path/to/omni_home." >&2
  exit 1
fi

OMNIMARKET_CLONE="$OMNI_HOME/omnimarket"
if [[ ! -d "$OMNIMARKET_CLONE/.git" ]]; then
  echo "ERROR: no canonical omnimarket clone at $OMNIMARKET_CLONE." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the target interpreter — fail fast, never silently pick a default.
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    echo "ERROR: no target python found. Activate the infra venv or pass PYTHON." >&2
    exit 1
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: target python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

echo "== refreshing canonical omnimarket clone from origin/dev =="
git -C "$OMNIMARKET_CLONE" fetch origin dev --quiet
CANONICAL_SHA="$(git -C "$OMNIMARKET_CLONE" rev-parse origin/dev)"
echo "  canonical origin/dev HEAD: $CANONICAL_SHA"

INSTALLED_SHA="$(env -u PYTHONPATH "$PYTHON_BIN" - <<'PYEOF'
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution

try:
    dist = distribution("omnimarket")
except PackageNotFoundError:
    print("")
    sys.exit(0)
raw = dist.read_text("direct_url.json") or ""
try:
    data = json.loads(raw) if raw else {}
except json.JSONDecodeError:
    data = {}
print(data.get("vcs_info", {}).get("commit_id", ""))
PYEOF
)"

if [[ -z "$INSTALLED_SHA" ]]; then
  echo "DRIFT: omnimarket is not installed (or not a VCS install — see OMN-14064) in $PYTHON_BIN."
elif [[ "$INSTALLED_SHA" != "$CANONICAL_SHA" ]]; then
  echo "DRIFT: installed $INSTALLED_SHA != canonical $CANONICAL_SHA"
else
  echo "OK: installed omnimarket matches canonical origin/dev HEAD ($INSTALLED_SHA)."
  exit 0
fi

if [[ "$REPAIR" -ne 1 ]]; then
  echo
  echo "Re-run with --repair to fix, or by hand:"
  echo "  OMNIMARKET_REF=$CANONICAL_SHA $SCRIPT_DIR/install-node-skill-package.sh --execute $PYTHON_BIN"
  exit 1
fi

echo
echo "== repairing: re-running canonical co-install at $CANONICAL_SHA =="
OMNIMARKET_REF="$CANONICAL_SHA" bash "$SCRIPT_DIR/install-node-skill-package.sh" --execute "$PYTHON_BIN"
