#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# Wrapper: run topic-naming-lint against both contract YAML files and Python
# source files. (OMN-3259, OMN-9573)
#
# Path resolution order for INFRA_ROOT (absolute path to omnibase_infra repo):
#   1. $OMNIBASE_INFRA_PATH (backward-compatible explicit override)
#   2. Two levels up from this script's directory (canonical: scripts/validation/)
#   3. $OMNI_HOME/omnibase_infra (cross-repo fallback)
#   4. Walk up from cwd looking for a sibling omnibase_infra/ directory
#   5. Warn and skip gracefully if none resolve
#
# Usage: invoked by pre-commit as a system-language hook.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINT="$SCRIPT_DIR/lint_topic_names.py"

# --- Resolve INFRA_ROOT ---
INFRA_ROOT=""

# 1. Backward-compatible explicit override.
if [[ -n "${OMNIBASE_INFRA_PATH:-}" && -d "$OMNIBASE_INFRA_PATH/src/omnibase_infra" ]]; then
    INFRA_ROOT="$OMNIBASE_INFRA_PATH"
fi

# 2. Canonical: this script lives at <infra_root>/scripts/validation/
_candidate="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -z "$INFRA_ROOT" && -d "$_candidate/src/omnibase_infra" ]]; then
    INFRA_ROOT="$_candidate"
fi

# 3. $OMNI_HOME/omnibase_infra
if [[ -z "$INFRA_ROOT" && -n "${OMNI_HOME:-}" && -d "$OMNI_HOME/omnibase_infra/src/omnibase_infra" ]]; then
    INFRA_ROOT="$OMNI_HOME/omnibase_infra"
fi

# 4. Walk up from cwd looking for a sibling omnibase_infra/
if [[ -z "$INFRA_ROOT" ]]; then
    _walk="$(pwd)"
    while [[ "$_walk" != "/" ]]; do
        if [[ -d "$_walk/omnibase_infra/src/omnibase_infra" ]]; then
            INFRA_ROOT="$_walk/omnibase_infra"
            break
        fi
        _walk="$(dirname "$_walk")"
    done
fi

if [[ -z "$INFRA_ROOT" ]]; then
    echo "topic-naming-lint: WARNING: could not locate omnibase_infra source tree; skipping scan." >&2
    exit 0
fi

RC=0
uv run --frozen python "$LINT" --scan-contracts "$INFRA_ROOT/src/omnibase_infra/nodes" || RC=$?
uv run --frozen python "$LINT" --scan-python "$INFRA_ROOT/src/omnibase_infra" || RC=$?

exit "$RC"
