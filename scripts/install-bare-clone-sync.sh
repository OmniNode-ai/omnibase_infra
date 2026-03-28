#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# install-bare-clone-sync.sh — Install the bare-clone-sync launchd agent
#
# Usage:
#   ./scripts/install-bare-clone-sync.sh           # install and load
#   ./scripts/install-bare-clone-sync.sh --uninstall  # unload and remove

set -euo pipefail

LABEL="ai.omninode.bare-clone-sync"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_SRC="$SCRIPT_DIR/ai.omninode.bare-clone-sync.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [[ "${1:-}" == "--uninstall" ]]; then
  echo "Unloading $LABEL..."
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST_DST"
  echo "Removed $PLIST_DST"
  exit 0
fi

if [[ ! -f "$PLIST_SRC" ]]; then
  echo "ERROR: plist not found at $PLIST_SRC" >&2
  exit 1
fi

# Ensure target directory exists
mkdir -p "$HOME/Library/LaunchAgents"

# Unload if already loaded
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true

# Copy and load
cp "$PLIST_SRC" "$PLIST_DST"
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"

echo "Installed and loaded $LABEL"
echo "  Plist: $PLIST_DST"
echo "  Interval: every 5 minutes"
echo "  Log: /tmp/bare-clone-sync.log"
