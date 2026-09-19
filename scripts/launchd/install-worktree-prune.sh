#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Render and install the unattended worktree prune LaunchAgent (OMN-18832).
#
# The plist is rendered rather than committed because a plist cannot expand a
# variable: launchd needs literal absolute paths, and a committed literal path
# is what CLAUDE.md Operating Rule 6 forbids. Rendering at install time gives
# launchd what it needs without putting this machine's paths in the repo.
#
#   bash scripts/launchd/install-worktree-prune.sh            # install
#   bash scripts/launchd/install-worktree-prune.sh --uninstall
#
# It installs ONLY. It does not run a prune: RunAtLoad is false in the
# template, so loading the agent never removes anything as a side effect. Take
# the first run by hand.
set -euo pipefail

LABEL="ai.omninode.worktree-prune"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${HERE}/com.omninode.worktree-prune.plist.template"
TARGET="${HOME}/Library/LaunchAgents/${LABEL}.plist"

if [[ "${1:-}" == "--uninstall" ]]; then
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
  rm -f "${TARGET}"
  echo "uninstalled ${LABEL}"
  exit 0
fi

# Fail fast rather than defaulting: a silently wrong OMNI_HOME would install a
# timer that prunes the wrong tree (Operating Rule 8).
: "${OMNI_HOME:?set OMNI_HOME to the omni_home registry root}"
[[ -d "${OMNI_HOME}/omni_worktrees" ]] || {
  echo "ERROR: ${OMNI_HOME} has no omni_worktrees/ — is that really the registry root?" >&2
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

mkdir -p "${OMNI_HOME}/.onex_state/worktree-prune" "${HOME}/Library/LaunchAgents"

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
echo "READBACK — a loaded-but-bootout'd job looks installed and does nothing,"
echo "so check the label AND, after the first run, a fresh log entry:"
launchctl list | grep -F "${LABEL}" || echo "  (not listed — the bootstrap did not take)"
echo "  logs: ${OMNI_HOME}/.onex_state/worktree-prune/"
