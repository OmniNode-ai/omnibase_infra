#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# check_version_source_sync.sh — Verify pyproject.toml version matches package __version__
#
# Usage:
#   ./scripts/check_version_source_sync.sh
#
# Exit codes:
#   0 — versions are in sync
#   1 — versions differ or could not be determined
#
# This script prevents version drift between the declared pyproject.toml version
# and the runtime __version__ string used by the package (OMN-4796).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYPROJECT="${REPO_ROOT}/pyproject.toml"
PKG_INIT="${REPO_ROOT}/src/omnibase_infra/__init__.py"

# ---------------------------------------------------------------------------
# Extract version from pyproject.toml
# ---------------------------------------------------------------------------
PYPROJECT_VERSION=$(python3 -c "
import tomllib, sys
with open('${PYPROJECT}', 'rb') as f:
    data = tomllib.load(f)
print(data.get('project', {}).get('version') or data.get('tool', {}).get('poetry', {}).get('version') or data['version'])
" 2>/dev/null || grep -E '^version' "${PYPROJECT}" | head -1 | tr -d ' ' | cut -d'"' -f2)

if [[ -z "${PYPROJECT_VERSION}" ]]; then
    echo "ERROR: Could not extract version from ${PYPROJECT}" >&2
    exit 1
fi

echo "pyproject.toml version: ${PYPROJECT_VERSION}"

# ---------------------------------------------------------------------------
# Extract __version__ from package __init__.py (fallback constant)
# ---------------------------------------------------------------------------
# The __init__.py uses importlib.metadata at runtime, with a static fallback:
#   __version__ = "0.0.0-dev"
# We also check for any hardcoded version string in the file.
# ---------------------------------------------------------------------------
INIT_FALLBACK=$(grep -E '__version__\s*=\s*"[0-9]' "${PKG_INIT}" | head -1 | sed 's/.*"\(.*\)".*/\1/' || true)

if [[ -n "${INIT_FALLBACK}" && "${INIT_FALLBACK}" != "0.0.0-dev" ]]; then
    echo "__init__.py fallback version: ${INIT_FALLBACK}"
    if [[ "${PYPROJECT_VERSION}" != "${INIT_FALLBACK}" ]]; then
        echo "" >&2
        echo "ERROR: Version mismatch detected!" >&2
        echo "  pyproject.toml : ${PYPROJECT_VERSION}" >&2
        echo "  __init__.py    : ${INIT_FALLBACK}" >&2
        echo "" >&2
        echo "Fix: update one of the above to match, then regenerate uv.lock." >&2
        exit 1
    fi
    echo "OK: pyproject.toml and __init__.py versions match (${PYPROJECT_VERSION})"
else
    # Package uses importlib.metadata — check via uv if available
    if command -v uv &>/dev/null; then
        RUNTIME_VERSION=$(cd "${REPO_ROOT}" && uv run python -c "from importlib.metadata import version; print(version('omnibase-infra'))" 2>/dev/null || echo "")
        if [[ -n "${RUNTIME_VERSION}" ]]; then
            echo "Runtime version (importlib.metadata): ${RUNTIME_VERSION}"
            if [[ "${PYPROJECT_VERSION}" != "${RUNTIME_VERSION}" ]]; then
                echo "" >&2
                echo "ERROR: Version mismatch detected!" >&2
                echo "  pyproject.toml  : ${PYPROJECT_VERSION}" >&2
                echo "  importlib.metadata: ${RUNTIME_VERSION}" >&2
                echo "" >&2
                echo "Fix: run 'uv build' after updating pyproject.toml version." >&2
                exit 1
            fi
            echo "OK: pyproject.toml and runtime versions match (${PYPROJECT_VERSION})"
        else
            echo "WARN: Could not determine runtime version (package not installed). Skipping runtime check."
        fi
    else
        echo "WARN: uv not found. Skipping runtime version check."
    fi
fi

exit 0
