#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Blocks PRs that touch strict-gate files without the ordering checkbox filled.
# Runs as a pre-commit hook and in CI.
# OMN-9125

set -euo pipefail

GATE_FILES=(
  "src/omnibase_infra/runtime/auto_wiring/handler_wiring.py"
  "src/omnibase_infra/runtime/service_kernel.py"
)

TEMPLATE=".github/PULL_REQUEST_TEMPLATE.md"

# In CI, check via staged/changed files against base branch.
# In pre-commit, check staged files only.
if [[ "${CI:-}" == "true" ]]; then
  BASE="${GITHUB_BASE_REF:-main}"
  CHANGED=$(git diff --name-only "origin/${BASE}...HEAD" 2>/dev/null || true)
else
  CHANGED=$(git diff --cached --name-only 2>/dev/null || true)
fi

touches_gate=false
for f in "${GATE_FILES[@]}"; do
  if echo "$CHANGED" | grep -qF "$f"; then
    touches_gate=true
    break
  fi
done

if [[ "$touches_gate" == "false" ]]; then
  exit 0
fi

# PR touches a gate file — verify the ordering checkbox is present and checked.
# In CI this is enforced via PR body; in pre-commit we warn only (body not available).
if [[ "${CI:-}" == "true" ]]; then
  PR_BODY="${PR_BODY:-}"
  if [[ -z "$PR_BODY" ]]; then
    echo "WARNING: check-strict-gate-ordering: PR_BODY not set in CI — skipping body check." >&2
    exit 0
  fi

  # Must have at least one compliance PR link or feature-flag note.
  if echo "$PR_BODY" | grep -qE '\[x\].*Yes.*ALL downstream' || \
     echo "$PR_BODY" | grep -qE '\[x\].*Yes.*behind a feature flag'; then
    echo "check-strict-gate-ordering: strict-gate ordering checkbox confirmed." >&2
    exit 0
  fi

  echo "ERROR: check-strict-gate-ordering: This PR touches a strict-gate file." >&2
  echo "  Files: ${GATE_FILES[*]}" >&2
  echo "  You must check one of the 'Yes' boxes in the 'Strict-gate ordering' section" >&2
  echo "  of the PR template and link the compliance PRs or feature-flag config." >&2
  exit 1
fi

# Pre-commit: warn only — PR body not available locally.
echo "WARNING: check-strict-gate-ordering: staged changes touch a strict-gate file." >&2
echo "  Ensure the 'Strict-gate ordering' section of the PR template is completed." >&2
exit 0
