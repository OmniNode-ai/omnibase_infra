#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Render and install the durable lab alarm LaunchAgent (OMN-18867).
#
# The plist is rendered rather than committed because a plist cannot expand a
# variable: launchd needs literal absolute paths, and a committed literal path
# is what CLAUDE.md Operating Rule 6 forbids. Rendering at install time gives
# launchd what it needs without putting this machine's paths in the repo.
#
#   bash scripts/launchd/install-lab-alarm.sh            # install
#   bash scripts/launchd/install-lab-alarm.sh --uninstall
#   bash scripts/launchd/install-lab-alarm.sh --render-only <target>
#
# It installs ONLY. RunAtLoad is false in the template, so loading the agent
# never evaluates anything as a side effect. Take the first run through
# launchd itself once installed (see the readback block below), which also
# produces the fresh log entry the readback asks you to check.
#
# --render-only writes the substituted plist to <target> and exits before
# plutil/bootstrap -- it never touches launchd or the real
# ~/Library/LaunchAgents. It exists so the substitution step (this is what a
# real install does with OMNI_HOME/BREW_PYTHON/ONEX_INFRA_HOST/
# ONEX_RUNTIME_SSH_HOST) is provable from a fixture environment, including on
# a CI runner with no brew interpreter -- see tests/ci/test_lab_alarm_omn18867.py.
set -euo pipefail

LABEL="ai.omninode.lab-alarm"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${HERE}/com.omninode.lab-alarm.plist.template"
TARGET="${HOME}/Library/LaunchAgents/${LABEL}.plist"

if [[ "${1:-}" == "--uninstall" ]]; then
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
  rm -f "${TARGET}"
  echo "uninstalled ${LABEL}"
  exit 0
fi

RENDER_ONLY=0
if [[ "${1:-}" == "--render-only" ]]; then
  RENDER_ONLY=1
  TARGET="${2:?--render-only needs a target file path}"
fi

# Fail fast rather than defaulting: a silently wrong OMNI_HOME would install a
# timer reading a different tree's config and reporting on a lane nobody runs
# (Operating Rule 8).
: "${OMNI_HOME:?set OMNI_HOME to the omni_home registry root}"
[[ -d "${OMNI_HOME}/omnibase_infra" ]] || {
  echo "ERROR: ${OMNI_HOME} has no omnibase_infra/ — is that the registry root?" >&2
  exit 1
}
[[ -f "${OMNI_HOME}/omnibase_infra/config/lab_alarm.json" ]] || {
  echo "ERROR: no config/lab_alarm.json — the agent would fire and refuse" >&2
  exit 1
}

# The lab host and its ssh alias, read exactly like OMNI_HOME above and for
# the same reason (Operating Rule 8): config/lab_alarm.json expands
# ${ONEX_INFRA_HOST} and ${ONEX_RUNTIME_SSH_HOST} at load and RAISES rather
# than probe an empty address if either is unset. Both are normally exported
# from a shell profile (~/.zshrc), which launchd never sources -- an agent
# installed without these crashes at config load on every real fire rather
# than evaluating any condition (OMN-18867 AC1 defect, found live 2026-09-21:
# a manual shell run succeeded on the operator's inherited env while
# `launchctl kickstart` on the plist as shipped before this fix died before
# reaching the first condition).
: "${ONEX_INFRA_HOST:?set ONEX_INFRA_HOST to the lab host this alarm reads}"
: "${ONEX_RUNTIME_SSH_HOST:?set ONEX_RUNTIME_SSH_HOST to the ssh target the docker probe rewrites through}"

# The brew interpreter, by literal resolved path. launchd runs with a
# restricted PATH and no login shell, so neither 'python3' nor a
# $(brew --prefix) expansion resolves inside the agent (Operating Rule 11).
BREW_PYTHON=""
for candidate in /opt/homebrew/bin/python3.13 /usr/local/bin/python3.13; do
  [[ -x "${candidate}" ]] && { BREW_PYTHON="${candidate}"; break; }
done
if [[ -z "${BREW_PYTHON}" ]]; then
  if [[ "${RENDER_ONLY}" == "1" ]]; then
    # --render-only never executes the rendered plist, so a missing brew
    # interpreter (e.g. a Linux CI runner) is not fatal here -- only for a
    # real install, where the agent would be unable to run at all.
    BREW_PYTHON="/render-only/no-brew-python-resolved"
  else
    echo "ERROR: no brew python3.13 at /opt/homebrew/bin or /usr/local/bin" >&2
    exit 1
  fi
fi

mkdir -p "${OMNI_HOME}/.onex_state/lab-alarm" "$(dirname "${TARGET}")"

sed -e "s|@OMNI_HOME@|${OMNI_HOME}|g" \
    -e "s|@BREW_PYTHON@|${BREW_PYTHON}|g" \
    -e "s|@HOME@|${HOME}|g" \
    -e "s|@ONEX_INFRA_HOST@|${ONEX_INFRA_HOST}|g" \
    -e "s|@ONEX_RUNTIME_SSH_HOST@|${ONEX_RUNTIME_SSH_HOST}|g" \
    "${TEMPLATE}" > "${TARGET}"

if [[ "${RENDER_ONLY}" == "1" ]]; then
  echo "rendered ${LABEL} -> ${TARGET}"
  exit 0
fi

plutil -lint "${TARGET}" >/dev/null

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "${TARGET}"

echo "installed ${LABEL} -> ${TARGET}"
echo "interpreter: ${BREW_PYTHON}"
echo
echo "READBACK — both halves, per the OMN-17173 correction. A loaded-but-"
echo "bootout'd job LOOKS installed and does nothing, so neither line alone"
echo "proves this fires:"
echo
echo "  1. a non-zero PID column for the label:"
launchctl list | grep -F "${LABEL}" || echo "     (not listed — the bootstrap did not take)"
echo
echo "  2. a FRESH entry in its own stdout log, after the first run:"
echo "     ${OMNI_HOME}/.onex_state/lab-alarm/launchd.out.log"
echo
echo "Take that first run THROUGH LAUNCHD now, so there is something to read"
echo "back -- StandardOutPath/StandardErrorPath only capture a run launchd"
echo "itself executes; running the python invocation below directly from a"
echo "shell writes to your terminal instead and leaves the log empty:"
echo "  launchctl kickstart -k gui/\$(id -u)/${LABEL}"
echo
echo "To read the same run's output directly rather than through the log,"
echo "invoke it exactly as launchd will (same interpreter, same env):"
echo "  ${BREW_PYTHON} ${OMNI_HOME}/omnibase_infra/scripts/lab_alarm.py \\"
echo "    --config ${OMNI_HOME}/omnibase_infra/config/lab_alarm.json \\"
echo "    --state-dir ${OMNI_HOME}/.onex_state/lab-alarm \\"
echo "    --ledger ${OMNI_HOME}/docs/tracking/ROLLING_WORK_LEDGER.md"
