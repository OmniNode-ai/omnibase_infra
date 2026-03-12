#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# check_placeholder_topics.sh — Detect placeholder topic names in production code
#
# Placeholder topics in production docstrings/code cause confusion during
# topic auditing and can mask misconfigurations. This script enforces that
# only real topic names are used in production source files (OMN-4797).
#
# Usage:
#   ./scripts/check_placeholder_topics.sh [--scan-dir <dir>]
#
# Exit codes:
#   0 — no placeholder topics found
#   1 — one or more placeholder topics detected
#
# Note: Test files (tests/) are excluded — placeholder topics are acceptable
# in test fixtures and unit tests.

set -euo pipefail

SCAN_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src}"
EXCLUDE_PATTERN="tests/"

# Placeholder patterns to deny in production code
DENY_PATTERNS=(
    "your-topic-here"
    "your_topic_here"
    "placeholder-topic"
    "placeholder_topic"
    "example-topic"
    "example_topic"
    "TODO-topic"
    "TODO_topic"
    "PLACEHOLDER_TOPIC"
    "MY_TOPIC"
    "SAMPLE_TOPIC"
)

FOUND=0
RESULTS=()

for PATTERN in "${DENY_PATTERNS[@]}"; do
    while IFS= read -r line; do
        # Skip test files
        if echo "${line}" | grep -q "${EXCLUDE_PATTERN}"; then
            continue
        fi
        RESULTS+=("${line}")
        FOUND=1
    done < <(grep -rn "${PATTERN}" "${SCAN_DIR}" --include="*.py" 2>/dev/null || true)
done

if [[ ${FOUND} -eq 1 ]]; then
    echo "ERROR: Placeholder topic name(s) found in production source:" >&2
    for result in "${RESULTS[@]}"; do
        echo "  ${result}" >&2
    done
    echo "" >&2
    echo "Replace placeholder topic names with real topic names from the topic registry." >&2
    echo "See: src/omnibase_infra/topics/ for canonical topic constants." >&2
    exit 1
fi

echo "OK: No placeholder topic names found in ${SCAN_DIR}"
exit 0
