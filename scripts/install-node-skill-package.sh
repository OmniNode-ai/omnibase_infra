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
# BLAST RADIUS: this MUTATES the target venv, which on a developer machine is
# usually the SHARED plugin CLI venv serving `onex` for every lane on the host
# (omni_home/CLAUDE.md rule 11). It is gated behind --execute; without it, the
# script resolves and prints the real uv plan and changes nothing. It never runs
# on import or in CI; nothing invokes it automatically.
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
# PIN RESOLUTION (OMN-18675): the same staleness applied, unnoticed, to the two
# omni-internal leaf deps co-installed beside omnimarket. They were baked-in
# literals (`omnibase-compat==0.5.5`, `omninode-memory==0.15.0`, written
# 2026-07-02 and never touched since) installed with `--no-deps` and an EXACT
# `==`, which is an instruction rather than a floor. On 2026-09-18 a --repair
# run therefore force-DOWNGRADED a healthy shared plugin CLI venv from
# omnibase-compat 0.5.7 and omninode-memory 0.18.0 back to those 2026-07 pins,
# removing `omnibase_compat.contracts.pr_occ_stamp` and breaking the whole
# `onex` CLI — every command, every lane on the host. Both versions are now
# read from the pyproject.toml OF THE REF BEING INSTALLED, so they cannot
# disagree with omnimarket's own declared requirements, and every install step
# is planned before it is applied and REFUSED if it would move any installed
# package backwards (scripts/venv_install_plan.py). There is no bypass flag.
#
# READBACK (OMN-18663): steps 3 and 4 below prove the venv still WORKS -- the
# mapped nodes resolve, every onex.cli entry point imports. Neither proves the
# venv carries THIS REF. `uv pip install` exits 0 for "already satisfied" as
# readily as for "3 packages installed", so an install that resolved to nothing
# at all passed every check here and reported success, while the drift guard
# went on refusing the same interpreter ("a reconcile ran, reported SUCCESS,
# and the venv is STILL drifted", twice on 2026-09-18). Step 5 now reads the
# venv back -- installed omnimarket VCS commit and version against the ref, and
# every co-installed sibling against the requirement the ref declares -- and
# fails naming both values. Success is a readback, never an exit status.
#
# ONE WRITER, ONE READER (OMN-18663): the whole sequence -- plan, install, read
# back -- runs while this process holds an exclusive fcntl lock for the target
# venv, and so does a plan-only run, because a dry run resolved against a venv
# mid-install describes a state that never existed (2026-09-18: a VCS-ref bump
# read as a REMOVE and refused, then planned cleanly seconds later). A caller
# that already holds that exact lock is not made to take it twice. See
# scripts/lib/venv_reconcile_lock.sh.
#
# Usage:
#   scripts/install-node-skill-package.sh [--execute] [PYTHON]
#     PYTHON  path to the target venv python (default: $VIRTUAL_ENV/bin/python,
#             else ./.venv/bin/python). Never hardcode an absolute path.
#   Env:
#     OMNIMARKET_REF   git rev to install (default: resolved dynamically —
#                      see REF RESOLUTION above).
#     OMNI_HOME        canonical repo registry root. REQUIRED: the co-installed
#                      pin versions are read from the ref's own pyproject.toml
#                      via this clone, and there is no baked-in default to fall
#                      back to (omni_home/CLAUDE.md rule 8).
#     ONEX_VENV_LOCK_TIMEOUT
#                      how long to wait for the venv lock (default 120s). Not a
#                      bypass: it changes how long a waiter waits, never whether
#                      the lock is required.
#
# There is NO flag and NO variable that skips the readback or runs the install
# unserialized (omni_home/CLAUDE.md rule 10).
#
# Exit codes:
#   0  plan printed (dry run), or install applied, verified AND read back
#   1  a precondition failed, or verification/readback failed after install
#   3  REFUSED — an install step would downgrade/remove an installed package
#   4  BUSY — another lane holds this venv's reconcile lock; nothing was read
#      and nothing was changed
# ----------------------------------------------------------------------------
set -euo pipefail

OMNIMARKET_GIT="https://github.com/OmniNode-ai/omnimarket.git"

# omnimarket's own required deps that live above/beside the infra layer. They
# are installed --no-deps so their internal metadata does not re-resolve (or
# downgrade) the infra layer beneath. The NAMES are fixed here; the VERSIONS
# are resolved from the ref being installed (OMN-18675) — never literals.
OMNI_INTERNAL_NO_DEPS_PKGS=(omnibase-compat omninode-memory)
# Pure-PyPI leaf deps (safe to resolve normally; no omni-internal metadata).
PYPI_LEAF_DEPS=(anthropic radon "docker>=7.0.0" python-dateutil)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_TOOL="$SCRIPT_DIR/venv_install_plan.py"

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

if [[ ! -f "$PLAN_TOOL" ]]; then
  echo "ERROR: missing install planner at $PLAN_TOOL (OMN-18675)." >&2
  exit 1
fi

READBACK_TOOL="$SCRIPT_DIR/venv_readback.py"
if [[ ! -f "$READBACK_TOOL" ]]; then
  echo "ERROR: missing install readback at $READBACK_TOOL (OMN-18663)." >&2
  echo "  Without it this script could only report its own exit status as" >&2
  echo "  proof, which is the defect OMN-18663 closed. Refusing." >&2
  exit 1
fi

# --------------------------------------------------------------------------- #
# OMN-18663: hold the target venv's exclusive lock for the whole sequence.
# --------------------------------------------------------------------------- #
# Re-runs this script under the lock and returns. A caller that already holds
# this exact lock (check-omnimarket-venv-drift.sh, which wraps its own critical
# section) sets the marker, and the nested run proceeds without a second
# acquisition -- a child process taking its own parent's lock would deadlock.
# shellcheck source=scripts/lib/venv_reconcile_lock.sh
source "$SCRIPT_DIR/lib/venv_reconcile_lock.sh"
VENV_LOCK_PATH="$(venv_reconcile_lock_path "$PYTHON_BIN")"
if [[ "${ONEX_VENV_RECONCILE_LOCK:-}" != "$VENV_LOCK_PATH" ]]; then
  set +e
  venv_reconcile_run_locked "$PYTHON_BIN" "$VENV_LOCK_PATH" \
    "omnimarket co-install ($PYTHON_BIN)" \
    -- bash "${BASH_SOURCE[0]}" "$@"
  locked_status=$?
  set -e
  if [[ "$locked_status" -eq "$VENV_LOCK_TIMEOUT_EXIT" ]]; then
    venv_reconcile_say_busy "$VENV_LOCK_PATH" "$PYTHON_BIN"
    exit 4
  fi
  exit "$locked_status"
fi

# The canonical clone is REQUIRED: it is where the co-installed pin versions
# are read from. Fail fast naming the variable rather than defaulting.
if [[ -z "${OMNI_HOME:-}" || ! -d "${OMNI_HOME}/omnimarket/.git" ]]; then
  echo "ERROR: OMNI_HOME must point at a repo registry containing an omnimarket clone." >&2
  echo "  The co-installed pin versions are read from the installed ref's own" >&2
  echo "  pyproject.toml via that clone (OMN-18675). There is no baked-in default." >&2
  exit 1
fi
OMNIMARKET_CLONE="${OMNI_HOME}/omnimarket"

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
  elif [[ -d "${OMNIMARKET_CLONE}/.git" ]]; then
    # Offline fallback: the canonical local clone's checked-out HEAD.
    OMNIMARKET_REF="$(git -C "${OMNIMARKET_CLONE}" rev-parse HEAD)"
    REF_SOURCE="local clone ${OMNIMARKET_CLONE} (offline fallback — git ls-remote unreachable)"
  else
    echo "ERROR: could not resolve an omnimarket ref." >&2
    echo "  git ls-remote ${OMNIMARKET_GIT} dev failed (network?), and no local" >&2
    echo "  canonical clone found at \$OMNI_HOME/omnimarket for offline fallback." >&2
    echo "  Set OMNIMARKET_REF=<sha> to pin explicitly." >&2
    exit 1
  fi
fi

# ----------------------------------------------------------------------------
# OMN-18675: read the omni-internal pin versions from the ref's own metadata.
# The ref must be present as an object in the canonical clone; fetch once if it
# is not, and fail fast rather than substituting a different (stale) ref.
# ----------------------------------------------------------------------------
if ! git -C "$OMNIMARKET_CLONE" cat-file -e "${OMNIMARKET_REF}^{commit}" 2>/dev/null; then
  git -C "$OMNIMARKET_CLONE" fetch --quiet origin dev || true
fi
if ! git -C "$OMNIMARKET_CLONE" cat-file -e "${OMNIMARKET_REF}^{commit}" 2>/dev/null; then
  echo "ERROR: ref $OMNIMARKET_REF is not present in $OMNIMARKET_CLONE." >&2
  echo "  The co-installed pin versions are read from that ref's pyproject.toml." >&2
  echo "  Fetch it first:  git -C $OMNIMARKET_CLONE fetch origin dev" >&2
  exit 1
fi

# GNU mktemp requires the XXXXXX suffix in a template; BSD mktemp accepts it
# too. `-t <prefix>` alone is BSD-only and fails on Linux runners.
PYPROJECT_TMP="$(mktemp "${TMPDIR:-/tmp}/omnimarket-pyproject.XXXXXX")"
trap 'rm -f "$PYPROJECT_TMP"' EXIT
git -C "$OMNIMARKET_CLONE" show "${OMNIMARKET_REF}:pyproject.toml" >"$PYPROJECT_TMP"

RESOLVED_PINS="$(env -u PYTHONPATH "$PYTHON_BIN" "$SCRIPT_DIR/resolve_omnimarket_pins.py" \
  --pyproject "$PYPROJECT_TMP" "${OMNI_INTERNAL_NO_DEPS_PKGS[@]}")"

# Word-split deliberately: one requirement per line, none contain whitespace.
OMNI_INTERNAL_PINS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && OMNI_INTERNAL_PINS+=("$line")
done <<<"$RESOLVED_PINS"

if [[ "${#OMNI_INTERNAL_PINS[@]}" -eq 0 ]]; then
  echo "ERROR: ${OMNIMARKET_REF}:pyproject.toml declares none of: ${OMNI_INTERNAL_NO_DEPS_PKGS[*]}" >&2
  echo "  Refusing to guess versions for the co-installed omni-internal deps." >&2
  exit 1
fi

echo "== node-skill-package install plan =="
echo "  target python : $PYTHON_BIN"
echo "  omnimarket ref: $OMNIMARKET_REF (via $REF_SOURCE)"
echo "  omni pins     : ${OMNI_INTERNAL_PINS[*]}"
echo "                  (resolved from ${OMNIMARKET_REF}:pyproject.toml — OMN-18675)"
echo "  step 1 (--no-deps): git+${OMNIMARKET_GIT}@${OMNIMARKET_REF} ${OMNI_INTERNAL_PINS[*]}"
echo "  step 2          : ${PYPI_LEAF_DEPS[*]}"
echo "  step 3          : verify merge_sweep / session / aislop_sweep nodes resolve"
echo "  step 4          : verify every advertised onex.cli entry point imports"
echo "  step 5          : read the venv back — it must CARRY this ref (OMN-18663)"
echo

PLAN_ARGS=()
if [[ "$EXECUTE" -eq 1 ]]; then
  PLAN_ARGS+=(--apply)
fi

echo "== step 1: omnimarket + omni-internal leaf deps (--no-deps) =="
env -u PYTHONPATH "$PYTHON_BIN" "$PLAN_TOOL" \
  --python "$PYTHON_BIN" --no-deps --label "step 1" ${PLAN_ARGS[@]+"${PLAN_ARGS[@]}"} -- \
  "omnimarket @ git+${OMNIMARKET_GIT}@${OMNIMARKET_REF}" \
  "${OMNI_INTERNAL_PINS[@]}"

echo
echo "== step 2: pure-PyPI leaf deps =="
env -u PYTHONPATH "$PYTHON_BIN" "$PLAN_TOOL" \
  --python "$PYTHON_BIN" --label "step 2" ${PLAN_ARGS[@]+"${PLAN_ARGS[@]}"} -- \
  "${PYPI_LEAF_DEPS[@]}"

if [[ "$EXECUTE" -ne 1 ]]; then
  echo
  echo "DRY RUN — nothing was changed. Re-run with --execute to apply (mutates the venv)."
  exit 0
fi

echo
echo "== step 3: verify node resolution =="
env -u PYTHONPATH "$PYTHON_BIN" - <<'PYEOF'
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

# OMN-18675 AC2: a co-install that leaves the venv importable-but-broken is not
# a success. The 2026-09-18 breakage surfaced as an `onex.cli` entry point that
# could no longer be LOADED (its module had vanished from the downgraded
# dependency), which takes down every `onex` subcommand, not just the one whose
# entry point broke. Load them all and fail loudly naming the ones that raise.
echo
echo "== step 4: verify every advertised onex.cli entry point imports =="
env -u PYTHONPATH "$PYTHON_BIN" - <<'PYEOF'
import sys
from importlib.metadata import entry_points

failures: list[tuple[str, str]] = []
eps = list(entry_points(group="onex.cli"))
for ep in eps:
    try:
        ep.load()
    except Exception as exc:  # noqa: BLE001 - any import failure breaks the CLI
        failures.append((ep.name, f"{type(exc).__name__}: {exc}"))

if failures:
    print("FAIL: the onex CLI does not load — these entry points raise:", file=sys.stderr)
    for name, reason in failures:
        print(f"  {name}: {reason}", file=sys.stderr)
    print(
        "\nThe venv is left in this state; `onex <any subcommand>` will raise.\n"
        "Re-install the dependency the failing entry point imports.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"OK: all {len(eps)} onex.cli entry points import cleanly.")
PYEOF

# OMN-18663: steps 3 and 4 prove the venv WORKS. This step proves it carries
# what was just asked for. They are different questions, and only the second one
# can tell an install that landed from an install that resolved to nothing --
# the state in which every check above passes and the drift guard still refuses.
echo
echo "== step 5: read the venv back against the installed ref =="
READBACK_ARGS=()
for pin in "${OMNI_INTERNAL_PINS[@]}"; do
  READBACK_ARGS+=(--sibling "$pin")
done
if ! env -u PYTHONPATH "$PYTHON_BIN" "$READBACK_TOOL" \
    --python "$PYTHON_BIN" \
    --clone "$OMNIMARKET_CLONE" \
    --ref "$OMNIMARKET_REF" \
    --label "co-install readback (OMN-18663)" \
    ${READBACK_ARGS[@]+"${READBACK_ARGS[@]}"}; then
  echo >&2
  echo "FAIL: the install steps all reported success and $PYTHON_BIN does not" >&2
  echo "  carry ${OMNIMARKET_REF}. This is NOT a completed install -- do not" >&2
  echo "  dispatch from this interpreter and do not record this run as a" >&2
  echo "  repair. Re-run the failing uv step by hand and read its output." >&2
  exit 1
fi

echo "== done: node-skill package installed, verified and read back =="
