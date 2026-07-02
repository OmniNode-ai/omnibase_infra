#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# install-node-skill-package.sh (OMN-13829)
# ----------------------------------------------------------------------------
# Install the node-backed skill package (`omnimarket`) into an omnibase_infra
# virtualenv so the `onex skill` / `onex run-node` operator CLIs can resolve the
# current mapped nodes (e.g. node_pr_lifecycle_orchestrator,
# node_session_orchestrator, node_aislop_sweep).
#
# THIS IS THE CANONICAL CO-INSTALL MECHANISM (the permanent, correct approach).
#   The onex CLI shipped in omnibase_infra composes market nodes at runtime via
#   co-installed `onex.nodes` entry-points. omnimarket is a *provider* discovered
#   at runtime, never a build/lock dependency of omnibase_infra. Installing it
#   with `--no-deps` is the correct, permanent composition step: the infra venv
#   already supplies every lower-layer omni dependency, and `--no-deps` layers
#   the provider on top without perturbing (or re-resolving) the layer beneath.
#
# WHY omnimarket IS NOT A pyproject DEPENDENCY (a layering boundary, not a bug):
#   Repo layering is compat -> core -> spi -> infra, and omnimarket sits ABOVE
#   infra (it depends on omnibase-infra >=0.38.3,<0.39.0). Declaring omnimarket
#   as an omnibase_infra dependency would INVERT the layer graph and publish a
#   cycle in the omnibase-infra wheel. The dependency direction is fixed by the
#   architecture: infra must not depend on market. See docs/decisions and the
#   OMN-13829 ticket for the recorded decision. This is why the skipped test in
#   tests/unit/runtime/test_event_bus_subscriber_container_resolution.py asserts
#   "omnimarket is no longer an omnibase_infra runtime dependency" — the runtime
#   composes market; it does not depend on it.
#
# BLAST RADIUS: this MUTATES the target venv. It is gated behind --execute;
# without it, the script only prints the plan. It never runs on import or in CI;
# nothing invokes it automatically. Real installs are operator-run only.
#
# Usage:
#   scripts/install-node-skill-package.sh [--execute] [PYTHON]
#     PYTHON  path to the target venv python (default: $VIRTUAL_ENV/bin/python,
#             else ./.venv/bin/python). Never hardcode an absolute path.
#   Env:
#     OMNIMARKET_REF  git rev/branch to install (default: pinned SHA below).
# ----------------------------------------------------------------------------
set -euo pipefail

# Immutable rev carrying the current node set. Pinned to omnimarket@dev HEAD as
# of 2026-07-02, which carries OMN-13836 (self-referential git-URL uv overrides
# dropped; omnibase-core>=0.46.1). Override via OMNIMARKET_REF to bump to a newer
# compatible rev without editing the script.
OMNIMARKET_REF="${OMNIMARKET_REF:-bc516ef5da67a348947fbb0e3c88dc964b2cd541}"
OMNIMARKET_GIT="https://github.com/OmniNode-ai/omnimarket.git"

# omnimarket's own required deps that live above/beside the infra layer. Pinned
# to the versions omnimarket@dev requires and installed --no-deps so their
# internal metadata does not re-resolve (or downgrade) the infra layer beneath.
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
echo "  step 3          : verify merge_sweep / session / aislop_sweep nodes resolve"

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
# Nodes behind the operator skills this install must keep resolvable:
#   merge_sweep -> node_pr_lifecycle_orchestrator
#   session     -> node_session_orchestrator
#   (aislop_sweep kept as a broad-coverage canary)
required = {
    "node_pr_lifecycle_orchestrator",
    "node_session_orchestrator",
    "node_aislop_sweep",
}
missing = sorted(required - eps)
if missing:
    print(f"FAIL: mapped nodes still unresolved: {missing}", file=sys.stderr)
    sys.exit(1)
print(f"OK: {len(eps)} onex.nodes entry points; required nodes resolved: {sorted(required)}")
PYEOF

echo "== done: node-skill package installed and verified =="
