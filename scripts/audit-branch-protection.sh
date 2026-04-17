#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# audit-branch-protection.sh — audit branch protection settings across OmniNode-ai repos
#
# Usage:
#   bash scripts/audit-branch-protection.sh [OPTIONS]
#
# Options:
#   --owner ORG       GitHub org/owner (default: OmniNode-ai)
#   --repo REPO       Audit a single repo instead of all known repos
#   --fix             Remove required_pull_request_reviews via PUT (dangerous)
#   --dry-run         Report violations but do not mutate (default when --fix absent)
#   --help            Show this help
#
# Exit codes:
#   0  — no violations found
#   1  — one or more violations found (or --fix ran but could not resolve)
#
# Check A: required_approving_review_count > 0  → violation (blocks solo-dev workflow)
# Check B: each required_status_checks.contexts[] must match at least one check-run
#          name seen in the last 5 commits on the repo default branch
#
# This script is a thin wrapper around scripts/audit_branch_protection_lib.py
# so the Check A / Check B logic is unit-testable without spawning subprocesses.
#
# OMN-9034

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
OWNER="OmniNode-ai"
SINGLE_REPO=""
DO_FIX=0

REPOS=(
  omniclaude
  omnibase_core
  omnibase_infra
  omnibase_spi
  omnidash
  omniintelligence
  omnimemory
  omninode_infra
  omniweb
  onex_change_control
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner) OWNER="$2"; shift 2 ;;
    --repo)  SINGLE_REPO="$2"; shift 2 ;;
    --fix)   DO_FIX=1; shift ;;
    --dry-run) DO_FIX=0; shift ;;
    --help|-h) usage ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$SINGLE_REPO" ]]; then
  REPOS=("$SINGLE_REPO")
fi

violations=0

check_repo() {
  local repo="$1"
  local full="${OWNER}/${repo}"
  local result
  local status
  local message
  local protection

  # Delegate Check A + Check B to the lib module (uses real gh as the injected caller).
  result=$(python3 "${SCRIPT_DIR}/audit_branch_protection_lib_cli.py" audit \
    --owner "${OWNER}" --repo "${repo}")

  status=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  message=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['message'])")

  case "$status" in
    skip)
      echo "  [SKIP] ${message}"
      return 0
      ;;
    ok)
      echo "  [OK]   ${message}"
      return 0
      ;;
    violation)
      echo "  [FAIL] ${repo}: ${message}"
      violations=$((violations + 1))

      if [[ "$DO_FIX" -eq 1 ]]; then
        protection=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['protection_json'])")
        echo "  [FIX]  Removing required_pull_request_reviews on ${repo}..."
        local put_payload
        put_payload=$(python3 "${SCRIPT_DIR}/audit_branch_protection_lib_cli.py" fix-payload \
          --protection-json "$protection")
        if gh api --method PUT "repos/${full}/branches/main/protection" \
             --input <(echo "$put_payload") > /dev/null 2>&1; then
          echo "  [OK]   required_pull_request_reviews removed on ${repo}"
          violations=$((violations - 1))
        else
          echo "  [ERR]  Could not remove required_pull_request_reviews on ${repo}"
        fi
      fi
      ;;
  esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "=== branch-protection audit: owner=${OWNER}, repos=${#REPOS[@]}, fix=${DO_FIX} ==="
echo ""

for repo in "${REPOS[@]}"; do
  echo "--- ${repo}"
  check_repo "$repo"
done

echo ""
if [[ "$violations" -eq 0 ]]; then
  echo "PASS: no violations found."
  exit 0
else
  echo "FAIL: ${violations} violation(s) found."
  if [[ "$DO_FIX" -eq 0 ]]; then
    echo "Re-run with --fix to attempt automated remediation."
  fi
  exit 1
fi
