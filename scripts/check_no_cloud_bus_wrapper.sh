#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Wrapper for check_no_cloud_bus.sh
# Resolves the script relative to this repo first, then falls back to $OMNI_HOME.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer the co-located copy in this repo
CHECK_SCRIPT="$REPO_ROOT/scripts/validation/check_no_cloud_bus.sh"
if [[ -f "$CHECK_SCRIPT" ]]; then
  exec bash "$CHECK_SCRIPT" "$PWD"
fi

# Fall back to OMNI_HOME for backwards compat
if [[ -n "${OMNI_HOME:-}" ]]; then
  CHECK_SCRIPT="$OMNI_HOME/scripts/check_no_cloud_bus.sh"
  if [[ -f "$CHECK_SCRIPT" ]]; then
    exec bash "$CHECK_SCRIPT" "$PWD"
  fi
fi

# OMN-14667: fail loud when the underlying check cannot be resolved. Exiting 0
# here made an unrunnable gate byte-indistinguishable from a passing one (the
# DRIFT-2 false-green shape the precommit-fail-loud-meta-gate now rejects).
echo "ERROR: check_no_cloud_bus.sh not found (looked in \$REPO_ROOT/scripts/validation and \$OMNI_HOME/scripts) -- cannot run cloud-bus guard; failing closed." >&2
exit 1
