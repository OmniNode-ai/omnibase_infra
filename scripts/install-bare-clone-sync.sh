#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# install-bare-clone-sync.sh — Install the bare-clone-sync launchd agent
#
# This script installs a launchd user agent that runs pull-all.sh every
# 30 minutes to keep bare clones in omni_home fresh.
#
# Usage:
#   ./install-bare-clone-sync.sh           # Install and start
#   ./install-bare-clone-sync.sh uninstall # Stop and remove
#
# Uninstall (manual):
#   launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/ai.omninode.bare-clone-sync.plist
#   rm ~/Library/LaunchAgents/ai.omninode.bare-clone-sync.plist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_NAME="ai.omninode.bare-clone-sync.plist"
PLIST_SRC="${SCRIPT_DIR}/${PLIST_NAME}"
PLIST_DST="${HOME}/Library/LaunchAgents/${PLIST_NAME}"
DOMAIN_TARGET="gui/$(id -u)"
SERVICE_TARGET="${DOMAIN_TARGET}/${PLIST_NAME%.plist}"

# ── Uninstall ────────────────────────────────────────────────────────
if [[ "${1:-}" == "uninstall" ]]; then
    echo "Uninstalling ${PLIST_NAME}..."

    if launchctl print "${SERVICE_TARGET}" &>/dev/null; then
        launchctl bootout "${DOMAIN_TARGET}" "${PLIST_DST}" 2>/dev/null || true
        echo "  Service stopped."
    else
        echo "  Service not loaded (nothing to stop)."
    fi

    if [[ -f "${PLIST_DST}" ]]; then
        rm -f "${PLIST_DST}"
        echo "  Plist removed from ~/Library/LaunchAgents/."
    else
        echo "  Plist not found (already removed)."
    fi

    echo "Done. Bare-clone-sync uninstalled."
    exit 0
fi

# ── Install ──────────────────────────────────────────────────────────
echo "Installing ${PLIST_NAME}..."

# Verify source plist exists
if [[ ! -f "${PLIST_SRC}" ]]; then
    echo "ERROR: Plist not found at ${PLIST_SRC}" >&2
    exit 1
fi

# Verify pull-all.sh exists at the path referenced in the plist
PULL_ALL="${OMNI_HOME}/omnibase_infra/scripts/pull-all.sh"
if [[ ! -f "${PULL_ALL}" ]]; then
    echo "ERROR: pull-all.sh not found at ${PULL_ALL}" >&2
    echo "       The plist references this absolute path." >&2
    exit 1
fi

# Create LaunchAgents directory if needed
mkdir -p "${HOME}/Library/LaunchAgents"

# Unload existing service if loaded
if launchctl print "${SERVICE_TARGET}" &>/dev/null; then
    echo "  Stopping existing service..."
    launchctl bootout "${DOMAIN_TARGET}" "${PLIST_DST}" 2>/dev/null || true
fi

# Copy plist
cp "${PLIST_SRC}" "${PLIST_DST}"
echo "  Copied plist to ${PLIST_DST}"

# Load service
launchctl bootstrap "${DOMAIN_TARGET}" "${PLIST_DST}"
echo "  Service loaded."

# Validate service is running
sleep 1
if launchctl print "${SERVICE_TARGET}" &>/dev/null; then
    echo "  Service is running."
    echo ""
    echo "Done. Bare clones will sync every 30 minutes."
    echo "Logs: /tmp/bare-clone-sync.log"
    echo "Errors: /tmp/bare-clone-sync-error.log"
else
    echo "WARNING: Service loaded but may not be running yet." >&2
    echo "         Check: launchctl print ${SERVICE_TARGET}" >&2
    exit 1
fi
