#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# install-node-skill-package.sh (OMN-13829)
# ----------------------------------------------------------------------------
# Install the node-backed skill package (`omnimarket`) into an omnibase_infra
# virtualenv so the `onex skill` / `onex run-node` operator CLIs can resolve the
# current mapped nodes (e.g. node_pr_lifecycle_orchestrator, node_aislop_sweep).
#
# WHY THIS SCRIPT EXISTS (instead of a pyproject dependency):
#   omnimarket cannot be declared as an omnibase_infra dependency and locked with
#   `uv lock`, for two independent, verified reasons:
#     1. Circular dependency: omnimarket depends back on omnibase-infra
#        (>=0.38.3,<0.39.0). Declaring it as a runtime dep would publish a cycle
#        in the omnibase-infra wheel; declaring it in a dev group still forces
#        uv to re-resolve the graph.
#     2. Upstream stale/self-referential pins: omnimarket's [tool.uv]
#        override-dependencies git-pin omnibase-core/omnibase-infra to foreign
#        revs (uv applies these transitively), and its required dependency
#        omninode-memory==0.15.0 hard-pins omnibase-infra==0.30.1 / spi 0.20.x.
#        Both collide with this repo's omnibase-core>=0.46.1 / spi>=0.23.0, so no
#        `uv lock` / `uv pip install` (deps mode) can co-resolve.
#   The published PyPI wheels (<=0.4.6) are worse still: they pin
#   omnibase-core<0.45.0 / spi<0.22.0, so a plain version-range bump is also
#   impossible. The only rev carrying the compatible pins AND the newer nodes is
#   omnimarket@dev, installed with `--no-deps` (this repo's venv already supplies
#   the omni-internal deps; only omnibase-compat + omninode-memory are added,
#   also `--no-deps`, to bypass their stale metadata).
#
# ROOT CAUSE TO FIX UPSTREAM (tracked for the durable fix): omnimarket + its
# omninode-memory dependency must drop the stale/self-referential internal pins
# and publish a wheel resolvable against current omnibase_infra. Once that lands,
# retire this script in favour of a plain `uv add omnimarket>=X,<Y`.
#
# BLAST RADIUS: this MUTATES the target venv. It is gated behind --execute;
# without it, the script only prints the plan.
#
# Usage:
#   scripts/install-node-skill-package.sh [--execute] [PYTHON]
#     PYTHON  path to the target venv python (default: $VIRTUAL_ENV/bin/python,
#             else ./.venv/bin/python). Never hardcode an absolute path.
#   Env:
#     OMNIMARKET_REF  git rev/branch to install (default: pinned SHA below).
# ----------------------------------------------------------------------------
set -euo pipefail

# Immutable dev-branch rev carrying compatible pins + the current node set.
# Override via OMNIMARKET_REF when bumping to a newer compatible rev.
OMNIMARKET_REF="${OMNIMARKET_REF:-363e4319aa7288bdf0f2858af7993bf8aa91fca0}"
OMNIMARKET_GIT="https://github.com/OmniNode-ai/omnimarket.git"

# Leaf deps not already provided by the omnibase_infra venv. Installed --no-deps
# so their stale internal metadata does not drag in incompatible omni pins.
COMPAT_PIN="omnibase-compat==0.5.5"
MEMORY_PIN="omninode-memory==0.15.0"
# Pure-PyPI leaf deps (safe to resolve normally; no omni-internal metadata).
PYPI_LEAF_DEPS=(anthropic radon "docker>=7.0.0" python-dateutil)

EXECUTE=0
PYTHON_BIN=""
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    *) PYTHON_BIN="$arg" ;;
  esac
done

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

echo "== node-skill-package install plan =="
echo "  target python : $PYTHON_BIN"
echo "  omnimarket ref: $OMNIMARKET_REF"
echo "  step 1 (--no-deps): git+${OMNIMARKET_GIT}@${OMNIMARKET_REF} ${COMPAT_PIN} ${MEMORY_PIN}"
echo "  step 2          : ${PYPI_LEAF_DEPS[*]}"
echo "  step 3          : verify node_pr_lifecycle_orchestrator resolves"

if [[ "$EXECUTE" -ne 1 ]]; then
  echo
  echo "DRY RUN — re-run with --execute to apply (mutates the venv)."
  exit 0
fi

echo
echo "== step 1: install omnimarket + omni-internal leaf deps (--no-deps) =="
uv pip install --python "$PYTHON_BIN" --no-deps \
  "omnimarket @ git+${OMNIMARKET_GIT}@${OMNIMARKET_REF}" \
  "$COMPAT_PIN" "$MEMORY_PIN"

echo "== step 2: install pure-PyPI leaf deps =="
uv pip install --python "$PYTHON_BIN" "${PYPI_LEAF_DEPS[@]}"

echo "== step 3: verify node resolution =="
"$PYTHON_BIN" - <<'PYEOF'
import sys
from importlib.metadata import entry_points

eps = {e.name for e in entry_points(group="onex.nodes")}
required = {"node_pr_lifecycle_orchestrator", "node_aislop_sweep"}
missing = sorted(required - eps)
if missing:
    print(f"FAIL: mapped nodes still unresolved: {missing}", file=sys.stderr)
    sys.exit(1)
print(f"OK: {len(eps)} onex.nodes entry points; required nodes resolved: {sorted(required)}")
PYEOF

echo "== done: node-skill package installed and verified =="
