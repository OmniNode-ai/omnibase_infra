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
# REF RESOLUTION (OMN-14060): a hand-edited SHA literal here goes stale the
# moment omnimarket@dev advances past it — that staleness is the OMN-13829 /
# OMN-14060 recurrence mechanism (an install pinned months ago silently missing
# fixes that landed since). The default now resolves the ref DYNAMICALLY at
# run time from omnimarket's live `dev` HEAD (`git ls-remote`, falling back to
# the local canonical clone at $OMNI_HOME/omnimarket when offline) instead of a
# baked-in literal. Set OMNIMARKET_REF to pin an exact rev for reproducibility
# or offline use — that override always wins and is never second-guessed.
#
# Usage:
#   scripts/install-node-skill-package.sh [--execute] [PYTHON]
#     PYTHON  path to the target venv python (default: $VIRTUAL_ENV/bin/python,
#             else ./.venv/bin/python). Never hardcode an absolute path.
#   Env:
#     OMNIMARKET_REF   git rev to install (default: resolved dynamically —
#                      see REF RESOLUTION above).
#     OMNI_HOME        canonical repo registry root; used only as the offline
#                      fallback source for ref resolution (optional).
# ----------------------------------------------------------------------------
set -euo pipefail

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

# Resolve the target interpreter first — cheap and local. Fail fast here,
# before ever touching the network for ref resolution below.
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

# Resolve the ref to install — fail fast, never silently fall back to a stale
# baked-in default (CLAUDE.md rule #8).
if [[ -n "${OMNIMARKET_REF:-}" ]]; then
  REF_SOURCE="OMNIMARKET_REF override (pinned/offline use)"
else
  echo "Resolving omnimarket ref from live dev HEAD (git ls-remote)..." >&2
  LS_REMOTE_OUTPUT="$(git ls-remote --heads "$OMNIMARKET_GIT" dev 2>/dev/null || true)"
  OMNIMARKET_REF="$(awk '{print $1}' <<<"$LS_REMOTE_OUTPUT" | head -n1)"
  if [[ -n "$OMNIMARKET_REF" ]]; then
    REF_SOURCE="git ls-remote ${OMNIMARKET_GIT} dev"
  elif [[ -n "${OMNI_HOME:-}" && -d "${OMNI_HOME}/omnimarket/.git" ]]; then
    # Offline fallback: the canonical local clone's checked-out HEAD.
    OMNIMARKET_REF="$(git -C "${OMNI_HOME}/omnimarket" rev-parse HEAD)"
    REF_SOURCE="local clone ${OMNI_HOME}/omnimarket (offline fallback — git ls-remote unreachable)"
  else
    echo "ERROR: could not resolve an omnimarket ref." >&2
    echo "  git ls-remote ${OMNIMARKET_GIT} dev failed (network?), and no local" >&2
    echo "  canonical clone found at \$OMNI_HOME/omnimarket for offline fallback." >&2
    echo "  Set OMNIMARKET_REF=<sha> to pin explicitly." >&2
    exit 1
  fi
fi

echo "== node-skill-package install plan =="
echo "  target python : $PYTHON_BIN"
echo "  omnimarket ref: $OMNIMARKET_REF (via $REF_SOURCE)"
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
