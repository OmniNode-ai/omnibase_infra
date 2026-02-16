#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
#
# Demo Loop Assertion Gate
#
# Single entrypoint that validates the complete demo loop
# (canonical events -> projection -> dashboard) is wired correctly.
#
# Usage:
#   ./scripts/demo-loop-assert.sh              # Full check (live infra)
#   ./scripts/demo-loop-assert.sh --ci         # CI mode (skip live checks)
#   ./scripts/demo-loop-assert.sh --verbose    # Debug logging
#   ./scripts/demo-loop-assert.sh --help       # Show help
#
# Exit codes:
#   0  All assertions passed (demo loop ready)
#   1  One or more assertions failed (demo loop not ready)
#   2  Internal error
#
# Related: OMN-2297

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =============================================================================
# Source .env if present (provides KAFKA_BOOTSTRAP_SERVERS, etc.)
# =============================================================================

ENV_FILE="${PROJECT_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    # shellcheck disable=SC1090
    set -a
    source "${ENV_FILE}"
    set +a
fi

# =============================================================================
# Forward all arguments to the Python gate module
# =============================================================================

echo "=========================================="
echo "  Demo Loop Assertion Gate (OMN-2297)"
echo "=========================================="
echo ""

cd "${PROJECT_ROOT}"
exec uv run python -m omnibase_infra.validation.demo_loop_gate "$@"
