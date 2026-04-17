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
# OMN-9034

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
OWNER="OmniNode-ai"
SINGLE_REPO=""
DO_FIX=0
DRY_RUN=1  # default: report only

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
    --fix)   DO_FIX=1; DRY_RUN=0; shift ;;
    --dry-run) DRY_RUN=1; DO_FIX=0; shift ;;
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
  local repo_violations=0

  # Fetch branch protection for main
  local protection
  protection=$(gh api "repos/${full}/branches/main/protection" 2>/dev/null) || {
    echo "  [SKIP] ${repo}: branch protection not enabled or inaccessible"
    return 0
  }

  # ------------------------------------------------------------------
  # Check A: required_approving_review_count > 0 blocks solo-dev workflow
  # ------------------------------------------------------------------
  local rac
  rac=$(echo "$protection" | python3 -c "
import json, sys
d = json.load(sys.stdin)
prr = d.get('required_pull_request_reviews') or {}
print(prr.get('required_approving_review_count', 0))
" 2>/dev/null) || rac=0

  if [[ "$rac" -gt 0 ]]; then
    echo "  [FAIL] ${repo} Check A: required_approving_review_count=${rac} (must be 0)"
    repo_violations=$((repo_violations + 1))

    if [[ "$DO_FIX" -eq 1 ]]; then
      echo "  [FIX]  Removing required_pull_request_reviews on ${repo}..."
      # Build minimal protection PUT payload — preserve other settings, drop reviews
      local put_payload
      put_payload=$(echo "$protection" | python3 -c "
import json, sys
d = json.load(sys.stdin)
payload = {}
rsc = d.get('required_status_checks')
if rsc:
    payload['required_status_checks'] = {
        'strict': rsc.get('strict', False),
        'contexts': rsc.get('contexts', []),
    }
payload['enforce_admins'] = bool((d.get('enforce_admins') or {}).get('enabled', False))
payload['required_pull_request_reviews'] = None
payload['restrictions'] = None
print(json.dumps(payload))
")
      if gh api --method PUT "repos/${full}/branches/main/protection" \
           --input <(echo "$put_payload") > /dev/null 2>&1; then
        echo "  [OK]   required_pull_request_reviews removed on ${repo}"
        repo_violations=$((repo_violations - 1))
      else
        echo "  [ERR]  Could not remove required_pull_request_reviews on ${repo}"
      fi
    fi
  fi

  # ------------------------------------------------------------------
  # Check B: each required status check context must match a recent run
  # ------------------------------------------------------------------
  local contexts
  contexts=$(echo "$protection" | python3 -c "
import json, sys
d = json.load(sys.stdin)
rsc = d.get('required_status_checks') or {}
ctxs = rsc.get('contexts', [])
print('\n'.join(ctxs))
" 2>/dev/null) || contexts=""

  if [[ -z "$contexts" ]]; then
    : # no required checks — nothing to validate
  else
    # Collect check-run names from the last 5 commits
    local seen_names=""
    local commits
    commits=$(gh api "repos/${full}/commits?per_page=5" --jq '.[].sha' 2>/dev/null) || commits=""

    while IFS= read -r sha; do
      [[ -z "$sha" ]] && continue
      local runs
      runs=$(gh api "repos/${full}/commits/${sha}/check-runs?per_page=50" \
               --jq '.check_runs[].name' 2>/dev/null) || runs=""
      seen_names="${seen_names}${runs}"$'\n'
    done <<< "$commits"

    while IFS= read -r ctx; do
      [[ -z "$ctx" ]] && continue
      if ! echo "$seen_names" | grep -qxF "$ctx"; then
        echo "  [FAIL] ${repo} Check B: context '${ctx}' not found in last-5-commit check-runs (orphaned)"
        repo_violations=$((repo_violations + 1))
      fi
    done <<< "$contexts"
  fi

  if [[ "$repo_violations" -eq 0 ]]; then
    echo "  [OK]   ${repo}: clean"
  fi

  violations=$((violations + repo_violations))
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
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Re-run with --fix to attempt automated remediation."
  fi
  exit 1
fi
