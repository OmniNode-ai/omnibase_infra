#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# audit-merge-method-on-main.sh — flag 2-parent merge commits on main
#
# Detects merge-queue bypass after the fact: PRs merged via the queue produce
# single-parent (squash) commits; admin-bypass / direct merges produce
# 2-parent merge commits. This script walks the last N commits on main for
# every OmniNode-ai repo and exits non-zero if any 2-parent commit is found
# outside the documented queue-batching pattern (which would still produce a
# single squash commit per PR).
#
# OMN-8843 — proposed-fix item 3 (post-merge validator)
#
# Usage:
#   bash scripts/audit-merge-method-on-main.sh [OPTIONS]
#
# Options:
#   --owner ORG       GitHub org/owner (default: OmniNode-ai)
#   --repo REPO       Audit a single repo instead of all known repos
#   --depth N         How many commits to inspect on main (default: 50)
#   --since DATE      Only consider commits since DATE (e.g. "2026-04-15")
#   --help            Show this help
#
# Exit codes:
#   0  — no bypass commits found
#   1  — one or more 2-parent commits on main (admin-bypass suspected)
#   2  — usage error / GitHub API failure
#
# Token: uses CROSS_REPO_PAT if set, else GITHUB_TOKEN. With GITHUB_TOKEN
# scoped to a single repo, sibling repos will be skipped.

set -euo pipefail

OWNER="OmniNode-ai"
SINGLE_REPO=""
DEPTH=50
SINCE=""

REPOS=(
  omniclaude
  omnibase_core
  omnibase_infra
  omnibase_spi
  omnibase_compat
  omnidash
  omniintelligence
  omnimemory
  omnimarket
  onex_change_control
)

usage() {
  sed -n '2,30p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner) OWNER="$2"; shift 2 ;;
    --repo) SINGLE_REPO="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    --since) SINCE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$SINGLE_REPO" ]]; then
  REPOS=("$SINGLE_REPO")
fi

# Build the GraphQL filter argument list.
# `since` filters server-side; without it we walk the most recent $DEPTH commits.
build_query() {
  local repo="$1"
  local since_clause=""
  if [[ -n "$SINCE" ]]; then
    since_clause=", since: \"${SINCE}T00:00:00Z\""
  fi
  cat <<EOF
{
  repository(owner: "$OWNER", name: "$repo") {
    defaultBranchRef {
      target {
        ... on Commit {
          history(first: $DEPTH${since_clause}) {
            nodes {
              oid
              messageHeadline
              committedDate
              parents { totalCount }
              author { user { login } }
            }
          }
        }
      }
    }
  }
}
EOF
}

FAILED=0
TOTAL_BYPASS=0

for repo in "${REPOS[@]}"; do
  query=$(build_query "$repo")
  if ! response=$(gh api graphql -f query="$query" 2>/dev/null); then
    echo "SKIP  $repo: GraphQL query failed (token scope or repo missing)" >&2
    continue
  fi

  # Extract 2-parent commits.
  two_parent=$(echo "$response" \
    | jq -r '.data.repository.defaultBranchRef.target.history.nodes[]
             | select(.parents.totalCount > 1)
             | "\(.oid[0:8])  \(.committedDate)  \(.author.user.login // "unknown")  \(.messageHeadline)"')

  if [[ -z "$two_parent" ]]; then
    echo "OK    $repo: no 2-parent commits in last $DEPTH on main"
    continue
  fi

  # Filter out merge-queue batching commits if any (queue creates a squashed
  # single-parent commit per PR — true 2-parent commits should not appear).
  # Currently no allowlist is needed; any 2-parent commit on main is flagged.
  count=$(echo "$two_parent" | wc -l | tr -d ' ')
  TOTAL_BYPASS=$((TOTAL_BYPASS + count))
  FAILED=1
  echo "FAIL  $repo: $count 2-parent commit(s) detected (admin-bypass suspected):"
  echo "$two_parent" | sed 's/^/      /'
done

if [[ "$FAILED" -ne 0 ]]; then
  echo ""
  echo "ERROR: $TOTAL_BYPASS 2-parent merge commit(s) on main bypassed the merge queue."
  echo "       Direct merges via admin bypass produce 2-parent commits and skip the SQUASH ruleset."
  echo "       See OMN-8843. Verify rulesets with: bash scripts/verify-merge-queue-no-bypass.sh"
  exit 1
fi

echo ""
echo "All audited repos: no merge-queue bypass detected on main."
