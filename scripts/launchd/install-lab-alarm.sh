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
#
# It installs ONLY. RunAtLoad is false in the template, so loading the agent
# never evaluates anything as a side effect. Take the first run by hand, which
# also produces the fresh log entry the readback below asks you to check.
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

# The brew interpreter, by literal resolved path. launchd runs with a
# restricted PATH and no login shell, so neither 'python3' nor a
# $(brew --prefix) expansion resolves inside the agent (Operating Rule 11).
BREW_PYTHON=""
for candidate in /opt/homebrew/bin/python3.13 /usr/local/bin/python3.13; do
  [[ -x "${candidate}" ]] && { BREW_PYTHON="${candidate}"; break; }
done
[[ -n "${BREW_PYTHON}" ]] || {
  echo "ERROR: no brew python3.13 at /opt/homebrew/bin or /usr/local/bin" >&2
  exit 1
}

mkdir -p "${OMNI_HOME}/.onex_state/lab-alarm" "${HOME}/Library/LaunchAgents"

sed -e "s|@OMNI_HOME@|${OMNI_HOME}|g" \
    -e "s|@BREW_PYTHON@|${BREW_PYTHON}|g" \
    -e "s|@HOME@|${HOME}|g" \
    "${TEMPLATE}" > "${TARGET}"

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
echo "Take that first run by hand now, so there is something to read back:"
echo "  ${BREW_PYTHON} ${OMNI_HOME}/omnibase_infra/scripts/lab_alarm.py \\"
echo "    --config ${OMNI_HOME}/omnibase_infra/config/lab_alarm.json \\"
echo "    --state-dir ${OMNI_HOME}/.onex_state/lab-alarm \\"
echo "    --ledger ${OMNI_HOME}/docs/tracking/ROLLING_WORK_LEDGER.md"
