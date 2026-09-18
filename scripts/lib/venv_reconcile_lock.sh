#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# venv_reconcile_lock.sh (OMN-18663) -- single-writer, single-reader lock for
# one venv's reconcile critical section. SOURCE this; do not execute it.
# ----------------------------------------------------------------------------
# THE RACE THIS CLOSES
#
# The target of the omnimarket repair path is usually the SHARED plugin CLI
# venv that serves `onex` for every lane on this host (omni_home/CLAUDE.md rule
# 11). Many lanes run concurrently, and each of the two entry points
# (check-omnimarket-venv-drift.sh, install-node-skill-package.sh) is a
# multi-step sequence: RESOLVE a plan, then APPLY it, then READ IT BACK. Every
# step reads site-packages that another lane may be rewriting.
#
# Observed on 2026-09-18 (lane cli-venv-converge-1520): one `--dry-run`
# classified a VCS-ref bump 0.4.118 -> None as REMOVE and refused (exit 3),
# and a second identical run seconds later planned cleanly. Nothing had
# changed except that the first read landed inside a peer's install, where
# uv had uninstalled the old distribution and not yet written the new one. A
# half-applied venv read as a downgrade, and the refusal the reader printed
# described a state that never existed as a steady state.
#
# uv takes its own exclusive lock on `<venv>/.lock` around an INSTALL, which is
# why two installs do not corrupt each other. It does not serialize a READER
# against a writer, and a `uv pip install --dry-run` is a reader. That is the
# exact gap here.
#
# WHY fcntl AND NOT flock(1)
#
# macOS ships no flock(1) (memory reference_macos_no_flock_use_fcntl_shim); the
# idiom silently exits 127 without running the command, which is the worst
# possible failure for a lock. `scripts/heavy_lock.py` already implements the
# fcntl.flock(2) wrapper this repo uses for exactly that reason -- bounded wait,
# holder sidecar, fail-closed on timeout, kernel-released on process death --
# so this helper composes it rather than growing a second lock implementation.
#
# WHY A MARKER RATHER THAN A SECOND ACQUISITION
#
# check-omnimarket-venv-drift.sh invokes install-node-skill-package.sh, so the
# critical sections nest. POSIX advisory locks are per-process, and a CHILD
# process is a different process: a nested acquisition would block on the
# parent's own lock until the timeout expired -- a self-deadlock that would
# read as BUSY. So the holder exports ONEX_VENV_RECONCILE_LOCK carrying the
# lock PATH, and a nested entry point skips acquisition only when that path is
# byte-identical to the one it would itself take. A marker naming venv A can
# therefore never suppress the lock for venv B.
#
# The marker is a handshake between these two scripts, not a bypass switch:
# there is no flag, and no environment variable, that makes a reconcile run
# without the lock held by someone. A hand-set marker is not defended against
# and is not claimed to be -- the honest limit is that this enforces
# serialization between cooperating entry points, not against an operator who
# edits the environment to defeat it.
# ----------------------------------------------------------------------------

# Seconds to wait for the lock before reporting BUSY. Not a bypass: it changes
# how long a waiter waits, never whether the lock is required. `0` means do not
# wait at all, which is a legitimate choice for a probe that would rather
# report BUSY immediately than block a session-start line.
ONEX_VENV_LOCK_TIMEOUT="${ONEX_VENV_LOCK_TIMEOUT:-120s}"

# heavy_lock.py's fail-closed timeout code (EX_TEMPFAIL). The wrapped command
# did NOT run.
# Read by the scripts that source this file, not here.
# shellcheck disable=SC2034
readonly VENV_LOCK_TIMEOUT_EXIT=75

# Resolve the lock path for a target interpreter. One path per venv, derived
# from the venv root, so two lanes repairing two different venvs never contend
# and two lanes repairing the SAME venv always do.
venv_reconcile_lock_path() {
  local python_bin="$1"
  local venv_root
  venv_root="$(cd "$(dirname "$python_bin")/.." 2>/dev/null && pwd -P)" || venv_root=""
  if [[ -n "$venv_root" && -f "$venv_root/pyvenv.cfg" && -w "$venv_root" ]]; then
    printf '%s\n' "$venv_root/.onex-venv-reconcile.lock"
    return 0
  fi
  # Not a venv, or a venv this user cannot write to (a system interpreter, a
  # root-owned tree). Fall back to a path-derived name under TMPDIR so the lock
  # is still SHARED by every lane targeting that same interpreter -- a
  # per-process temp file would be a lock in name only.
  local slug="${venv_root:-$python_bin}"
  printf '%s\n' "${TMPDIR:-/tmp}/onex-venv-reconcile${slug//\//_}.lock"
}

# Run a command while holding the venv's exclusive lock, unless an ancestor
# already holds that exact lock. Returns the command's own exit status, or
# VENV_LOCK_TIMEOUT_EXIT (75) when the lock could not be acquired in time --
# in which case the command did NOT run.
#
#   venv_reconcile_run_locked <python> <lock-path> <label> -- <command...>
venv_reconcile_run_locked() {
  local python_bin="$1" lock_path="$2" label="$3"
  shift 3
  [[ "${1:-}" == "--" ]] && shift

  if [[ "${ONEX_VENV_RECONCILE_LOCK:-}" == "$lock_path" ]]; then
    "$@"
    return $?
  fi

  # Run the lock helper on the TARGET interpreter. heavy_lock.py is stdlib
  # only, so a venv whose PACKAGES are broken -- the state this path exists to
  # repair -- still runs it fine, and using the target keeps the lock from
  # depending on whatever `python3` happens to be first on PATH (a macOS
  # /usr/bin/python3 is old enough to lack `datetime.UTC`). A venv with no
  # working interpreter at all is out of scope: nothing in this path works
  # there, including the repair.
  local lock_tool
  lock_tool="$(dirname "${BASH_SOURCE[0]}")/../heavy_lock.py"
  if [[ ! -f "$lock_tool" ]]; then
    echo "ERROR: missing lock helper at $lock_tool (OMN-18663)." >&2
    echo "  Refusing to run unserialized against a shared venv." >&2
    return 1
  fi

  env ONEX_VENV_RECONCILE_LOCK="$lock_path" \
    "$python_bin" "$lock_tool" \
    --lock "$lock_path" \
    --timeout "$ONEX_VENV_LOCK_TIMEOUT" \
    --label "$label" \
    -- "$@"
}

# Print the BUSY report for a lock acquisition that timed out. The wording is
# deliberate: a reader that could not take the lock has observed NOTHING, so it
# must not print a plan, a verdict, or a drift classification -- the 2026-09-18
# REMOVE misread is exactly what printing one anyway looks like.
venv_reconcile_say_busy() {
  local lock_path="$1" python_bin="$2"
  echo >&2
  echo "BUSY: another lane holds the reconcile lock for this venv." >&2
  echo "  venv : $python_bin" >&2
  echo "  lock : $lock_path" >&2
  echo "  waited: $ONEX_VENV_LOCK_TIMEOUT" >&2
  echo >&2
  echo "  NOTHING was read and NOTHING was changed. A venv mid-install reads as" >&2
  echo "  a downgrade or a removal that is not real (OMN-18663), so this reports" >&2
  echo "  BUSY rather than a verdict it cannot honestly give. Re-run once the" >&2
  echo "  peer finishes; the holder's pid and start time are recorded beside the" >&2
  echo "  lock at ${lock_path}.holder." >&2
}
