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
# against the target venv's installed commit, and — with --repair —
# fast-forwards the canonical clone itself to that same SHA before re-running
# the canonical co-install (install-node-skill-package.sh) to fix drift. The
# fast-forward step (OMN-16366) is what keeps both sides of the comparison in
# agreement afterward: the pre-flight guard compares the installed venv
# against the canonical clone's own checked-out HEAD, never origin/dev
# directly, so installing origin/dev into the venv without also advancing the
# clone would leave the guard immediately re-failing on the next dispatch,
# just with drift reversed. When the clone has diverged (e.g. an unpushed
# local commit) a clean fast-forward is impossible and --repair refuses
# rather than force it — see the printed message for the manual fix.
#
# Run this periodically (a session/cron tick), or by hand after the pre-flight
# guard raises "omnimarket venv is STALE". It never runs automatically on
# `onex skill` dispatch — the hot path only detects and instructs; this script
# is the actual repair.
#
# DOWNGRADE REFUSAL (OMN-18675): the target venv is usually the SHARED plugin
# CLI venv that serves `onex` for every lane on the host (omni_home/CLAUDE.md
# rule 11), so a repair that moves an unrelated package backwards breaks every
# lane, not just the one that ran it. That is not hypothetical: on 2026-09-18
# this remedy fixed the omnimarket pin and silently downgraded omnibase-compat
# 0.5.7 -> 0.5.5 and omninode-memory 0.18.0 -> 0.15.0, which removed
# `omnibase_compat.contracts.pr_occ_stamp` and took the whole `onex` CLI down
# host-wide. Every install step is now planned first and REFUSED if it would
# move any installed package backwards, and `--dry-run` prints that plan
# without touching the venv or the clone. There is no bypass flag.
#
# READBACK BEFORE SUCCESS (OMN-18663): this script used to exit on the install
# script's exit STATUS. uv exits 0 for "already satisfied" exactly as readily as
# for "3 packages installed", so a --repair that resolved to nothing at all was
# indistinguishable from one that landed, and the guard this remedy exists to
# satisfy went on refusing -- in its own words, "a reconcile ran, reported
# SUCCESS, and the venv is STILL drifted" (observed twice on 2026-09-18).
# --repair now RE-RUNS the same comparison it opened with, against the venv it
# just wrote, and exits non-zero naming both values when the venv does not
# carry the ref. Success is a readback, never a status.
#
# ONE WRITER, ONE READER (OMN-18663): the target is usually the SHARED plugin
# CLI venv (omni_home/CLAUDE.md rule 11) and every lane on this host uses it.
# The whole critical section -- resolve, plan, install, read back -- runs while
# this process holds an exclusive fcntl lock for that venv, and --dry-run takes
# the SAME lock, because a plan resolved against a venv mid-install describes a
# state that never existed (2026-09-18: a VCS-ref bump read as a REMOVE and
# refused, then planned cleanly seconds later). A reader that cannot take the
# lock reports BUSY and prints no verdict at all. See
# scripts/lib/venv_reconcile_lock.sh.
#
# Usage:
#   scripts/check-omnimarket-venv-drift.sh [--repair|--dry-run] [PYTHON]
#     --repair   fast-forward the clone and apply the repair
#     --dry-run  print the repair plan (resolved versions, before -> after for
#                every package that would change) and change NOTHING
#     PYTHON     target venv python (default: $VIRTUAL_ENV/bin/python, else
#                ./.venv/bin/python)
#   Env:
#     OMNI_HOME  canonical repo registry root (required)
#     ONEX_VENV_LOCK_TIMEOUT  how long to wait for the venv lock (default 120s).
#                Not a bypass: it changes how long a waiter waits, never whether
#                the lock is required.
#
# There is NO flag and NO variable that skips the readback or runs the install
# unserialized. A repair that cannot be proven is not a repair (rule 10).
#
# Exit codes:
#   0  no drift (or drift found, repaired with --repair, AND PROVEN by the
#      readback; or a clean plan printed with --dry-run)
#   1  drift detected and not repaired -- either no flag was passed, omnimarket
#      is not installed / not a VCS install, or --repair ran and the readback
#      shows the venv STILL does not carry the canonical ref
#   3  REFUSED — the repair would downgrade/remove an installed package
#   4  BUSY — another lane holds this venv's reconcile lock; nothing was read
#      and nothing was changed
# ----------------------------------------------------------------------------
set -euo pipefail

REPAIR=0
DRY_RUN=0
PYTHON_BIN=""
for arg in "$@"; do
  case "$arg" in
    --repair) REPAIR=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) PYTHON_BIN="$arg" ;;
  esac
done

if [[ "$REPAIR" -eq 1 && "$DRY_RUN" -eq 1 ]]; then
  echo "ERROR: --repair and --dry-run are mutually exclusive." >&2
  exit 1
fi

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

# The provider co-install this script delegates to. Overridable so the repair
# path is testable without a network install; not a bypass -- it changes WHICH
# installer runs, never whether the readback below has to pass.
INSTALL_SCRIPT="${ONEX_DRIFT_INSTALL_SCRIPT:-$SCRIPT_DIR/install-node-skill-package.sh}"

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

# --------------------------------------------------------------------------- #
# OMN-18663: hold this venv's exclusive lock for the WHOLE critical section.
# --------------------------------------------------------------------------- #
# Re-runs this script under the lock and returns. The nested invocation sees the
# ONEX_VENV_RECONCILE_LOCK marker for this exact lock path and proceeds without
# trying to take it again (a child is a different process; a second acquisition
# would deadlock against its own parent). Everything below this point therefore
# runs with the lock held -- the fetch, the comparison, the plan, the install
# and the readback -- so no peer can rewrite site-packages underneath any of it.
# shellcheck source=scripts/lib/venv_reconcile_lock.sh
source "$SCRIPT_DIR/lib/venv_reconcile_lock.sh"
VENV_LOCK_PATH="$(venv_reconcile_lock_path "$PYTHON_BIN")"
if [[ "${ONEX_VENV_RECONCILE_LOCK:-}" != "$VENV_LOCK_PATH" ]]; then
  set +e
  venv_reconcile_run_locked "$PYTHON_BIN" "$VENV_LOCK_PATH" \
    "omnimarket venv drift check ($PYTHON_BIN)" \
    -- bash "${BASH_SOURCE[0]}" "$@"
  locked_status=$?
  set -e
  if [[ "$locked_status" -eq "$VENV_LOCK_TIMEOUT_EXIT" ]]; then
    venv_reconcile_say_busy "$VENV_LOCK_PATH" "$PYTHON_BIN"
    exit 4
  fi
  exit "$locked_status"
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

# OMN-18675: --dry-run resolves and prints the exact plan the repair would
# apply — including the versions resolved from the ref's own pyproject.toml and
# a before -> after line for every package that would change — without touching
# the venv OR fast-forwarding the canonical clone. Run this before --repair on
# any shared venv.
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "== dry run: resolving the repair plan (nothing will be changed) =="
  OMNIMARKET_REF="$CANONICAL_SHA" bash "$INSTALL_SCRIPT" "$PYTHON_BIN"
  exit 0
fi

if [[ "$REPAIR" -ne 1 ]]; then
  echo
  echo "Re-run with --dry-run to see exactly what would change (nothing is"
  echo "mutated), then:"
  echo "Re-run with --repair to fix, or by hand:"
  echo "  OMNIMARKET_REF=$CANONICAL_SHA $INSTALL_SCRIPT --execute $PYTHON_BIN"
  exit 1
fi

# OMN-16366: --repair installs $CANONICAL_SHA (origin/dev) into the target
# venv, but the in-process guard (omnimarket_drift_guard.py) compares against
# the canonical clone's own checked-out HEAD, never origin/dev directly. If
# the clone itself is left behind, --repair "succeeds" but the guard
# immediately re-fails on the very next dispatch -- now with drift REVERSED
# (installed == origin/dev, canonical clone == stale). Fast-forward the clone
# to the same commit being installed FIRST, so both sides land in agreement.
# A clean fast-forward is refused (never forced) when the clone has diverged
# -- e.g. an unpushed local commit (OMN-14638 shape) -- because forcing it
# would silently discard that commit; the operator must resolve it by hand.
echo
echo "== fast-forwarding canonical clone to origin/dev before repair (OMN-16366) =="
if ! git -C "$OMNIMARKET_CLONE" merge --ff-only origin/dev; then
  echo
  echo "ERROR: canonical clone at $OMNIMARKET_CLONE cannot fast-forward to" >&2
  echo "  origin/dev ($CANONICAL_SHA) -- it has local commits origin/dev does" >&2
  echo "  not have, or a dirty working tree. Refusing to repair: installing" >&2
  echo "  $CANONICAL_SHA into $PYTHON_BIN while the clone stays behind would" >&2
  echo "  leave the guard immediately re-failing with drift REVERSED." >&2
  echo "  Resolve the canonical clone by hand, then re-run --repair:" >&2
  echo "    cd $OMNIMARKET_CLONE" >&2
  echo "    git status                      # inspect what's local-only" >&2
  echo "    git push origin HEAD:dev        # keep it: publish, then re-run" >&2
  echo "    # -- or, to discard it instead --" >&2
  echo "    git reset --hard origin/dev" >&2
  exit 1
fi

echo
echo "== repairing: re-running canonical co-install at $CANONICAL_SHA =="
OMNIMARKET_REF="$CANONICAL_SHA" bash "$INSTALL_SCRIPT" --execute "$PYTHON_BIN"

# OMN-18663: the install script's exit status is NOT the verdict. Re-run the
# comparison this script opened with, against the venv that was just written --
# the same installed-VCS-commit read, against the same canonical sha. A repair
# whose result the venv does not carry is a repair that did not happen, and
# reporting it as success is what made the guard and this remedy disagree.
echo
if ! env -u PYTHONPATH "$PYTHON_BIN" "$SCRIPT_DIR/venv_readback.py" \
    --python "$PYTHON_BIN" \
    --clone "$OMNIMARKET_CLONE" \
    --ref "$CANONICAL_SHA" \
    --label "post-repair readback (OMN-18663)"; then
  echo >&2
  echo "DRIFTED: the repair reported success and the venv does not carry it." >&2
  echo "  requested : $CANONICAL_SHA" >&2
  echo "  venv      : $PYTHON_BIN" >&2
  echo >&2
  echo "  Do NOT dispatch from this interpreter: the drift guard will refuse it" >&2
  echo "  and it would be right to. Re-run the install by hand and read the uv" >&2
  echo "  output:" >&2
  echo "    OMNIMARKET_REF=$CANONICAL_SHA $INSTALL_SCRIPT --execute $PYTHON_BIN" >&2
  exit 1
fi

echo
echo "== repaired and PROVEN: $PYTHON_BIN carries omnimarket ${CANONICAL_SHA:0:12} =="
