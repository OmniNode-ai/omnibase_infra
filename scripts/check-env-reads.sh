#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# CI gate: block new os.environ/os.getenv reads outside approved modules.
# Modes:
#   --staged  (pre-commit): check staged diff only
#   --base <ref>  (CI): check diff against base branch
set -euo pipefail

MODE="${1:---staged}"
BASE_REF="${2:-origin/main}"

APPROVED_PATTERNS=(
    "runtime/service_kernel.py"
    "runtime/overlay/"
    "runtime/config_discovery/config_prefetcher.py"
    "runtime/runtime_profile.py"
    "tests/"
    "scripts/"
)

ENV_READ_PATTERNS='os\.environ\[|os\.environ\.get|os\.getenv|from os import environ|from os import getenv'

get_changed_files() {
    case "$MODE" in
        --staged) git diff --cached --diff-filter=ACMR --name-only -- '*.py' ;;
        --base)   git diff "$BASE_REF"...HEAD --diff-filter=ACMR --name-only -- '*.py' ;;
        *)        echo "Unknown mode: $MODE" >&2; exit 1 ;;
    esac
}

get_diff() {
    local file="$1"
    case "$MODE" in
        --staged) git diff --cached -- "$file" ;;
        --base)   git diff "$BASE_REF"...HEAD -- "$file" ;;
    esac
}

FAILED=0
while IFS= read -r file; do
    [ -z "$file" ] && continue
    APPROVED=0
    for pattern in "${APPROVED_PATTERNS[@]}"; do
        [[ "$file" == *"$pattern"* ]] && APPROVED=1 && break
    done
    [ "$APPROVED" -eq 1 ] && continue

    if get_diff "$file" | grep -qE "^\+.*($ENV_READ_PATTERNS)"; then
        echo "BLOCKED: $file introduces new os.environ/os.getenv read"
        echo "  Use overlay-resolved config instead."
        echo "  Approved modules: ${APPROVED_PATTERNS[*]}"
        FAILED=1
    fi
done < <(get_changed_files)

exit $FAILED
