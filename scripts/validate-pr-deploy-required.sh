#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Wave D CI gate: runtime-change PRs must cite a ticket with deploy evidence.
# Ticket: OMN-8912
# Root cause: OMN-8841 — Dockerfile.runtime changed, deploy never dispatched,
#             deploy-agent inactive 2 days undetected.
#
# Usage (GitHub Actions):
#   bash scripts/validate-pr-deploy-required.sh <PR_NUMBER> [contracts_dir]
#
# Environment:
#   GH_TOKEN  — required for gh pr diff
#   GITHUB_REPOSITORY — set automatically in Actions
#
# Exit codes:
#   0  - Gate passed
#   1  - Gate failed
#   2  - Usage error

set -euo pipefail

PR_NUMBER="${1:-}"
CONTRACTS_DIR="${2:-contracts}"

if [[ -z "$PR_NUMBER" ]]; then
    echo "::error::Usage: $0 <PR_NUMBER> [contracts_dir]" >&2
    exit 2
fi

# Get changed files for this PR
CHANGED_FILES=$(gh pr diff "$PR_NUMBER" --name-only 2>/dev/null || true)

if [[ -z "$CHANGED_FILES" ]]; then
    echo "::notice::No changed files detected for PR #$PR_NUMBER — gate skipped."
    exit 0
fi

# Get PR body
PR_BODY=$(gh pr view "$PR_NUMBER" --json body -q '.body' 2>/dev/null || echo "")

# Run Python validator
CHANGED_ONELINE=$(echo "$CHANGED_FILES" | tr '\n' ' ')

python scripts/validation/validate_pr_deploy_required.py \
    --changed-files "$CHANGED_ONELINE" \
    --pr-body "$PR_BODY" \
    --contracts-dir "$CONTRACTS_DIR"
