#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# install-watchdog.sh — Install the CAIA watchdog launchd agent
#
# Usage:
#   bash scripts/install-watchdog.sh          # Install and load
#   bash scripts/install-watchdog.sh --uninstall  # Unload and remove

set -euo pipefail

LABEL="ai.omninode.caia-watchdog"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_SRC="${SCRIPT_DIR}/ai.omninode.caia-watchdog.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
CHECKPOINT_DIR="${HOME}/.onex_state/orchestrator"

if [[ "${1:-}" == "--uninstall" ]]; then
    echo "Uninstalling ${LABEL}..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "Done. Watchdog uninstalled."
    exit 0
fi

echo "Installing ${LABEL}..."

# Ensure checkpoint directory exists (WatchPaths needs it)
mkdir -p "$CHECKPOINT_DIR"

# Ensure LaunchAgents directory exists
mkdir -p "${HOME}/Library/LaunchAgents"

# Make watchdog script executable
chmod +x "${SCRIPT_DIR}/caia-watchdog.sh"

# Unload existing if present (safe to fail)
launchctl unload "$PLIST_DST" 2>/dev/null || true

# Copy plist
cp "$PLIST_SRC" "$PLIST_DST"

# Load the agent
launchctl load "$PLIST_DST"

echo "Done. Watchdog installed and running."
echo "  Plist:      ${PLIST_DST}"
echo "  Script:     ${SCRIPT_DIR}/caia-watchdog.sh"
echo "  Log:        ${CHECKPOINT_DIR}/watchdog.log"
echo "  Checkpoint: ${CHECKPOINT_DIR}/checkpoint.yaml"
echo ""
echo "To uninstall: bash ${SCRIPT_DIR}/install-watchdog.sh --uninstall"
