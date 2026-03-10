#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# check-sibling-compat.sh — CI pre-check for sibling repo install compatibility
#
# Validates that sibling repos (omniintelligence, omnimemory) can be installed
# with their full transitive dependencies alongside the current omnibase_infra
# source using the --overrides strategy (not --no-deps).
#
# Exits 0 if compatible, 1 if not.
#
# Usage:
#   ./scripts/check-sibling-compat.sh
#
# Environment:
#   SIBLING_REPOS_DIR — parent directory containing sibling repo checkouts
#                       (default: parent of the script's repo root)
#
# This script is designed to run in the same environment as the
# Kafka Schema Handshake (OMN-3411) and Kafka Boundary Compat (OMN-3256)
# CI jobs, and replicates their install logic for early failure detection.
#
# Permanent prevention for the --no-deps cascade window issue (OMN-4315):
# The --no-deps flag was previously used to avoid version conflicts when sibling
# repos pinned a released omnibase-infra version lagging behind the current PR.
# However, --no-deps skips ALL transitive deps, causing ImportError at test
# collection time (e.g., ModuleNotFoundError: No module named 'adaptive_classifier').
# The correct fix is --overrides: install full deps, but force omnibase-infra
# to the current local editable version.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIBLING_REPOS_DIR="${SIBLING_REPOS_DIR:-$(cd "${REPO_ROOT}/.." && pwd)}"

SIBLINGS=("omniintelligence" "omnimemory")
MISSING_SIBLINGS=()

echo "================================================================"
echo "Sibling Repo Compatibility Check (OMN-4315 prevention)"
echo "================================================================"
echo "Repo root:     ${REPO_ROOT}"
echo "Siblings dir:  ${SIBLING_REPOS_DIR}"
echo ""

# Check sibling repos exist
for sibling in "${SIBLINGS[@]}"; do
    sibling_path="${SIBLING_REPOS_DIR}/${sibling}"
    if [[ ! -d "${sibling_path}" ]]; then
        MISSING_SIBLINGS+=("${sibling}")
        echo "SKIP: ${sibling} not found at ${sibling_path}"
    else
        echo "FOUND: ${sibling} at ${sibling_path}"
    fi
done

if [[ ${#MISSING_SIBLINGS[@]} -eq ${#SIBLINGS[@]} ]]; then
    echo ""
    echo "INFO: No sibling repos found — skipping compatibility check."
    echo "      (This is expected in isolated repo development.)"
    echo "      To run full check, ensure siblings are checked out at:"
    for sibling in "${SIBLINGS[@]}"; do
        echo "        ${SIBLING_REPOS_DIR}/${sibling}"
    done
    exit 0
fi

# Extract current omnibase_infra version
INFRA_VERSION=$(python3 -c "
import tomllib, pathlib
d = tomllib.load(open(pathlib.Path('${REPO_ROOT}/pyproject.toml'), 'rb'))
print(d['project']['version'])
")
echo ""
echo "Current omnibase-infra version: ${INFRA_VERSION}"

# Create override file
OVERRIDE_FILE=$(mktemp /tmp/sibling-compat-overrides.XXXXXX.txt)
echo "omnibase-infra==${INFRA_VERSION}" > "${OVERRIDE_FILE}"
echo "Override file: ${OVERRIDE_FILE}"
echo "Contents: $(cat "${OVERRIDE_FILE}")"
echo ""

# Build install args for available siblings
INSTALL_ARGS=()
for sibling in "${SIBLINGS[@]}"; do
    sibling_path="${SIBLING_REPOS_DIR}/${sibling}"
    if [[ -d "${sibling_path}" ]]; then
        INSTALL_ARGS+=("-e" "../${sibling}")
    fi
done

echo "Running dry-run compatibility check..."
echo "Command: uv pip install --overrides ${OVERRIDE_FILE} ${INSTALL_ARGS[*]} --dry-run"
echo ""

cd "${REPO_ROOT}"

# Run the actual install check (dry-run where possible, or verify import after install)
if uv pip install --overrides "${OVERRIDE_FILE}" "${INSTALL_ARGS[@]}" --dry-run 2>&1; then
    echo ""
    echo "================================================================"
    echo "PASS: Sibling repos are compatible with current omnibase_infra."
    echo "      Full transitive deps would install successfully."
    echo "================================================================"
    EXIT_CODE=0
else
    EXIT_CODE=$?
    echo ""
    echo "================================================================"
    echo "FAIL: Sibling repo install failed (exit ${EXIT_CODE})."
    echo ""
    echo "Diagnosis:"
    echo "  - A sibling repo may have an unresolvable dependency conflict"
    echo "    even after overriding omnibase-infra to ${INFRA_VERSION}"
    echo "  - Check if the sibling's pyproject.toml has other pins that"
    echo "    conflict with omnibase_infra's uv.lock"
    echo ""
    echo "Do NOT use --no-deps as a workaround — it hides transitive dep"
    echo "failures that will surface as ImportError at test collection time."
    echo ""
    echo "Resolution options:"
    echo "  1. Update sibling repo's pin for the conflicting package"
    echo "  2. Add additional overrides for the conflicting package"
    echo "  3. Open a cascade ticket to track the version bump"
    echo "================================================================"
    EXIT_CODE=1
fi

rm -f "${OVERRIDE_FILE}"
exit "${EXIT_CODE}"
