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

# Anchored path patterns — matched against /-separated path segments to prevent
# substring false-positives (e.g. "tests/" must not match "src/contests/").
# Prefix patterns: file must START with this value (top-level directories).
APPROVED_PREFIX_PATTERNS=(
    "tests/"
    "scripts/"
)
# Infix patterns: file must contain "/<pattern>" (path segment boundary).
APPROVED_INFIX_PATTERNS=(
    "/runtime/service_kernel.py"
    "/runtime/overlay/"
    "/runtime/config_discovery/config_prefetcher.py"
    "/runtime/runtime_profile.py"
    "/services/registry_api/registry_discovery.py"
    # OMN-13537: receipt-mode CLI is the config-resolution boundary between the
    # --state-root flag and the ArtifactStore's sole env-var contract
    # (ONEX_ARTIFACT_STORE_ROOT). The pinned omnibase_core store exposes no
    # injection seam, so the boundary must publish the resolved default to the
    # environment — same class of boundary as service_kernel/config_prefetcher.
    "/cli/receipt_mode.py"
)

ENV_READ_PATTERNS='os\.environ\[|os\.environ\.get|os\.getenv|from os import environ|from os import getenv'

is_approved() {
    local file="$1"
    for prefix in "${APPROVED_PREFIX_PATTERNS[@]}"; do
        [[ "$file" == "$prefix"* ]] && return 0
    done
    for infix in "${APPROVED_INFIX_PATTERNS[@]}"; do
        [[ "$file" == *"$infix"* ]] && return 0
    done
    return 1
}

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
    is_approved "$file" && continue

    if get_diff "$file" | grep -qE "^\+.*($ENV_READ_PATTERNS)"; then
        echo "BLOCKED: $file introduces new os.environ/os.getenv read"
        echo "  Use overlay-resolved config instead."
        echo "  Approved top-level dirs: ${APPROVED_PREFIX_PATTERNS[*]}"
        echo "  Approved path segments: ${APPROVED_INFIX_PATTERNS[*]}"
        FAILED=1
    fi
done < <(get_changed_files)

exit $FAILED
