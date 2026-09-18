#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# install-worktree-prune.sh — Install the .201 ledger-CLAIM-aware worktree prune
# systemd USER timer (OMN-18688), and retire the two passes it replaces.
#
# These are systemd USER units (NOT lane containers). Installing/enabling them is
# scoped to the runtime user and does not touch any docker-compose lane. No sudo.
#
# This is the *orchestrator's deploy step* — it is NOT run by the worker that
# ships the code PR. The worker ships the unit files; the operator runs this on
# .201 after the PR has merged.
#
# Usage (run on 192.168.86.201 after pulling latest):
#   bash deploy/disk-gc/install-worktree-prune.sh --preflight   # check only, changes nothing
#   bash deploy/disk-gc/install-worktree-prune.sh               # install + enable + start timer
#   bash deploy/disk-gc/install-worktree-prune.sh --status
#   bash deploy/disk-gc/install-worktree-prune.sh --uninstall
#
# WHAT THIS RETIRES, AND WHY (full evidence: README-worktree-prune.md)
# --------------------------------------------------------------------
# Installing this unit disables and removes onex-worktree-reaper.timer/.service.
# Leaving it running would leave TWO differently-safe pruners live on one host:
# the reaper drives prune-worktrees.sh, which removes a clean+pushed+merged
# worktree with `git worktree remove --force` and reads the rolling work ledger
# nowhere — and clean+pushed+merged is exactly the state a LIVE lane occupies
# between its push and its post-merge verification (OMN-15551).
#
# The worktree-gc.sh ExecStart line is removed from onex-disk-gc.service in the
# same repository change; this script re-copies that unit so the running host
# picks the removal up.
#
# Prerequisites:
#   - systemd user manager available (loginctl enable-linger $USER if headless)
#   - omni_home registry at $OMNI_HOME (default ~/Code/omni_home)
#   - an omniclaude clone alongside it, carrying scripts/worktree_auto_prune.py
#   - a python interpreter that can import omniclaude.hooks.lib.worktree_prune_policy
#
# The last two are CHECKED, not assumed, and this script REFUSES to install when
# either is unsatisfied rather than writing a unit that fails silently at 04:30
# every morning. Run --preflight to see exactly which item fails and its remedy.
#
# Measured on .201, 2026-09-18: the interpreter check PASSES
# (omniclaude/.venv/bin/python imports the policy module), and the LEDGER check
# FAILS -- that host's omni_home clone has been at a May 2026 commit since, and
# carries no docs/tracking/ROLLING_WORK_LEDGER.md at all. Claim-awareness is the
# entire safety difference between this pass and the two it retires, so a prune
# there stays blocked until that clone carries the ledger. That refusal is the
# design working, not a gap in it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_SRC="${SCRIPT_DIR}/onex-worktree-prune.service"
TIMER_SRC="${SCRIPT_DIR}/onex-worktree-prune.timer"
DISK_GC_SERVICE_SRC="${SCRIPT_DIR}/onex-disk-gc.service"

USER_UNIT_DIR="${HOME}/.config/systemd/user"
SERVICE_DST="${USER_UNIT_DIR}/onex-worktree-prune.service"
TIMER_DST="${USER_UNIT_DIR}/onex-worktree-prune.timer"
DISK_GC_SERVICE_DST="${USER_UNIT_DIR}/onex-disk-gc.service"

ENV_DIR="${HOME}/.config/onex"
ENV_FILE="${ENV_DIR}/worktree-prune.env"

# Fail-fast on the registry root — never a silent default that prunes the wrong
# tree (omni_home/CLAUDE.md rule 8).
OMNI_HOME="${OMNI_HOME:-${HOME}/Code/omni_home}"
OMNICLAUDE_DIR="${OMNI_HOME}/omniclaude"
PRUNE_SCRIPT="${OMNICLAUDE_DIR}/scripts/worktree_auto_prune.py"
WORKTREES_ROOT="${OMNI_HOME}/omni_worktrees"
LEDGER="${OMNI_HOME}/docs/tracking/ROLLING_WORK_LEDGER.md"

die() { echo "ERROR: $*" >&2; exit 1; }

# --- interpreter resolution -------------------------------------------------
# Ordered, and every candidate is PROVEN by importing the policy module rather
# than by existing. An interpreter that exists but cannot import the predicate
# would install a unit that fails every morning with a traceback nobody reads.
resolve_python() {
  local candidates=()
  [[ -n "${WORKTREE_PRUNE_PYTHON:-}" ]] && candidates+=("${WORKTREE_PRUNE_PYTHON}")
  candidates+=(
    "${OMNICLAUDE_DIR}/.venv/bin/python"
    "${HOME}/.local/share/onex/worktree-prune-venv/bin/python"
  )
  command -v uv >/dev/null 2>&1 && candidates+=("__uv__")
  candidates+=("$(command -v python3 || true)")

  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ "$candidate" == "__uv__" ]]; then
      if (cd "$OMNICLAUDE_DIR" && uv run python -c \
          'import omniclaude.hooks.lib.worktree_prune_policy' >/dev/null 2>&1); then
        echo "__uv__"; return 0
      fi
      continue
    fi
    [[ -x "$candidate" ]] || continue
    if PYTHONPATH="${OMNICLAUDE_DIR}/src" "$candidate" -c \
        'import omniclaude.hooks.lib.worktree_prune_policy' >/dev/null 2>&1; then
      echo "$candidate"; return 0
    fi
  done
  return 1
}

preflight() {
  local ok=0
  echo "Preflight for onex-worktree-prune (OMN-18688)"
  echo "  OMNI_HOME        = ${OMNI_HOME}"
  echo "  worktrees root   = ${WORKTREES_ROOT}"
  echo "  classifier       = ${PRUNE_SCRIPT}"
  echo "  ledger           = ${LEDGER}"
  echo ""

  [[ -d "$WORKTREES_ROOT" ]] \
    && echo "  OK    worktrees root exists ($(find "$WORKTREES_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') ticket dirs)" \
    || { echo "  FAIL  worktrees root is not a directory: ${WORKTREES_ROOT}"; ok=1; }

  [[ -f "$PRUNE_SCRIPT" ]] \
    && echo "  OK    classifier present" \
    || { echo "  FAIL  classifier missing: ${PRUNE_SCRIPT}"; ok=1; }

  # The ledger is the claim-awareness source. Without it the classifier refuses
  # to run at all, which is correct — a prune with no claim-awareness IS the
  # OMN-15551 hazard — but it should be caught here, not at 04:30.
  [[ -f "$LEDGER" ]] \
    && echo "  OK    ledger readable ($(grep -c '| CLAIM |' "$LEDGER" || true) CLAIM rows)" \
    || { echo "  FAIL  ledger missing: ${LEDGER} — the classifier refuses to run without it"; ok=1; }

  if resolved="$(resolve_python)"; then
    if [[ "$resolved" == "__uv__" ]]; then
      echo "  OK    interpreter: uv run, from ${OMNICLAUDE_DIR}"
    else
      echo "  OK    interpreter: ${resolved} (imports the policy module)"
    fi
  else
    ok=1
    cat >&2 <<'REMEDY'
  FAIL  no interpreter on this host can import omniclaude.hooks.lib.worktree_prune_policy

        This is a PREREQUISITE, not something this script guesses its way past.
        Installing the unit anyway would schedule a job that fails every morning
        with a traceback in a journal nobody reads, which is worse than not
        installing it: the host would look maintained and would not be.

        Remedy, in preference order:
          1. install uv for the runtime user, then re-run this script; or
          2. build a venv and point this script at it:
               python3 -m venv ~/.local/share/onex/worktree-prune-venv
               ~/.local/share/onex/worktree-prune-venv/bin/python -m pip install \
                   pydantic pydantic-settings pyyaml
             (uv is preferred per the shared standards; pip is the fallback only
              where uv is genuinely absent, and that is a deviation worth fixing)
REMEDY
  fi

  echo ""
  if [[ $ok -eq 0 ]]; then
    echo "Preflight PASSED. Safe to run: bash deploy/disk-gc/install-worktree-prune.sh"
  else
    echo "Preflight FAILED. Nothing was changed."
  fi
  return $ok
}

case "${1:-}" in
  --preflight)
    preflight
    exit $?
    ;;
  --status)
    systemctl --user list-timers onex-worktree-prune.timer --no-pager || true
    echo ""
    systemctl --user status onex-worktree-prune.timer onex-worktree-prune.service --no-pager || true
    echo ""
    echo "Latest report: ${HOME}/.local/log/onex/worktree-prune-latest.md"
    echo ""
    echo "Recent logs:"
    journalctl --user -u onex-worktree-prune.service -n 40 --no-pager || true
    exit 0
    ;;
  --uninstall)
    echo "Uninstalling onex-worktree-prune user timer..."
    systemctl --user stop onex-worktree-prune.timer onex-worktree-prune.service 2>/dev/null || true
    systemctl --user disable onex-worktree-prune.timer 2>/dev/null || true
    rm -f "$SERVICE_DST" "$TIMER_DST" "$ENV_FILE"
    systemctl --user daemon-reload
    echo "Done. onex-worktree-prune uninstalled."
    echo "NOTE: this does NOT put the retired reaper back. That is deliberate —"
    echo "      it was retired on safety grounds, not to make room. To restore it:"
    echo "      bash deploy/disk-gc/install-worktree-reaper.sh"
    exit 0
    ;;
  "")
    ;;
  *)
    die "unknown argument: $1 (expected --preflight, --status, --uninstall, or none)"
    ;;
esac

preflight || die "preflight failed; nothing was changed. Fix the items above and re-run."

RESOLVED_PYTHON="$(resolve_python)"
if [[ "$RESOLVED_PYTHON" == "__uv__" ]]; then
  PY_BIN="$(command -v uv) run --directory ${OMNICLAUDE_DIR} python"
  PY_PATH=""
else
  PY_BIN="$RESOLVED_PYTHON"
  PY_PATH="${OMNICLAUDE_DIR}/src"
fi

echo ""
echo "Installing onex-worktree-prune systemd USER timer..."

mkdir -p "$ENV_DIR" "$USER_UNIT_DIR"
cat > "$ENV_FILE" <<ENVEOF
# Generated by deploy/disk-gc/install-worktree-prune.sh (OMN-18688).
# Regenerate by re-running that script; do not hand-edit.
WORKTREE_PRUNE_PYTHON=${PY_BIN}
WORKTREE_PRUNE_SCRIPT=${PRUNE_SCRIPT}
PYTHONPATH=${PY_PATH}
ENVEOF
echo "  interpreter recorded in ${ENV_FILE}"

cp "$SERVICE_SRC" "$SERVICE_DST"
cp "$TIMER_SRC" "$TIMER_DST"

# Re-copy onex-disk-gc.service so the host picks up the removal of its
# worktree-gc.sh ExecStart line. Without this the retired pass keeps running
# beside the new one, which is the exact two-pruner state this change removes.
if [[ -f "$DISK_GC_SERVICE_DST" ]]; then
  cp "$DISK_GC_SERVICE_SRC" "$DISK_GC_SERVICE_DST"
  echo "  refreshed onex-disk-gc.service (worktree-gc.sh ExecStart line removed)"
fi

# Retire the reaper. Stopped, disabled and removed — not left disabled-but-present,
# because a unit file on disk is a unit somebody re-enables in six months without
# reading why it was turned off.
if systemctl --user list-unit-files 2>/dev/null | grep -q '^onex-worktree-reaper\.timer'; then
  echo "  retiring onex-worktree-reaper.timer (superseded: no ledger-claim gate)"
  systemctl --user stop onex-worktree-reaper.timer onex-worktree-reaper.service 2>/dev/null || true
  systemctl --user disable onex-worktree-reaper.timer 2>/dev/null || true
  rm -f "${USER_UNIT_DIR}/onex-worktree-reaper.timer" \
        "${USER_UNIT_DIR}/onex-worktree-reaper.service"
fi

systemctl --user daemon-reload
systemctl --user enable --now onex-worktree-prune.timer

echo ""
echo "Done. onex-worktree-prune installed and armed (user scope)."
echo "  Service: ${SERVICE_DST}"
echo "  Timer:   ${TIMER_DST}"
echo "  Env:     ${ENV_FILE}"
echo ""
echo "Check:      systemctl --user list-timers onex-worktree-prune.timer"
echo "Logs:       journalctl --user -u onex-worktree-prune.service -f"
echo "Report:     ${HOME}/.local/log/onex/worktree-prune-latest.md"
echo "Uninstall:  bash deploy/disk-gc/install-worktree-prune.sh --uninstall"
echo ""
echo "NOTE: if this host runs headless, enable lingering so the timer fires"
echo "without an active login session:  loginctl enable-linger \$USER"
