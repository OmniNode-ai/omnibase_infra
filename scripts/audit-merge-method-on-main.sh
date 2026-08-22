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

# Known, approved one-time merge-commit exceptions (short SHA -> ticket
# reference). Do NOT add an entry here casually — it silences a real bypass
# signal for the life of the --depth window. Only add SHAs that were
# reviewed and approved as legitimate ancestry/migration exceptions; cite
# the approving ticket so the exemption itself is auditable.
#
# OMN-16107: without this allowlist, the two OMN-15028 ancestry-bridge
# commits below re-flagged this audit red on every ~4-6h scheduled run for
# 10+ days straight (2026-08-07 through at least 2026-08-17), which is the
# actual mechanism behind "the audit is always red so it gets tuned out" —
# not a missing/expired CROSS_REPO_PAT (verified present + functioning on
# every sampled run in that window).
declare -A ALLOWLISTED_TWO_PARENT_SHAS=(
  ["11a040da"]="OMN-15028 — ancestry bridge, one-time merge-commit exception"
  ["d6ef368d"]="OMN-15028 — record main ancestry for promotion"
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
AUDITED=0

GH_STDERR_FILE=$(mktemp)
trap 'rm -f "$GH_STDERR_FILE"' EXIT

for repo in "${REPOS[@]}"; do
  query=$(build_query "$repo")
  # OMN-16107: capture stderr instead of discarding it — a bare `2>/dev/null`
  # here turned every real cause (rate limit, transient network error, token
  # scope) into the same opaque "token scope or repo missing" guess, with no
  # way to tell which one actually happened on a given run.
  : >"$GH_STDERR_FILE"
  if ! response=$(gh api graphql -f query="$query" 2>"$GH_STDERR_FILE"); then
    gh_err=$(tr '\n' ' ' <"$GH_STDERR_FILE")
    echo "SKIP  $repo: gh api graphql failed: ${gh_err:-<no stderr captured>}" >&2
    continue
  fi

  # GraphQL may return HTTP 200 with an `errors` array; skip such repos.
  if jq -e '(.errors // []) | length > 0' >/dev/null <<<"$response"; then
    gql_errors=$(jq -c '.errors' <<<"$response")
    echo "SKIP  $repo: GraphQL returned errors: $gql_errors" >&2
    continue
  fi

  # Extract 2-parent commits safely even when nested objects are null/missing.
  if ! two_parent=$(jq -r '
      (.data.repository.defaultBranchRef.target.history.nodes // [])[]
      | select((.parents.totalCount // 0) > 1)
      | "\(.oid[0:8])  \(.committedDate // "unknown-date")  \(.author.user.login // "unknown")  \(.messageHeadline // "")"
    ' <<<"$response"); then
    echo "SKIP  $repo: malformed GraphQL payload" >&2
    continue
  fi

  AUDITED=$((AUDITED + 1))

  if [[ -z "$two_parent" ]]; then
    echo "OK    $repo: no 2-parent commits in last $DEPTH on main"
    continue
  fi

  # Split out approved one-time exceptions (ALLOWLISTED_TWO_PARENT_SHAS)
  # from genuine unexplained 2-parent commits. An allowlisted commit is
  # still reported (visibility), just not counted as a failure.
  unallowed=""
  allowed=""
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    sha="${line%%  *}"
    if [[ -n "${ALLOWLISTED_TWO_PARENT_SHAS[$sha]+set}" ]]; then
      allowed+="$line  [ALLOWLISTED: ${ALLOWLISTED_TWO_PARENT_SHAS[$sha]}]"$'\n'
    else
      unallowed+="$line"$'\n'
    fi
  done <<<"$two_parent"

  if [[ -n "$allowed" ]]; then
    echo "ALLOW $repo: approved one-time merge-commit exception(s):"
    printf '%s' "$allowed" | sed 's/^/      /'
  fi

  if [[ -z "$unallowed" ]]; then
    echo "OK    $repo: no unexplained 2-parent commits in last $DEPTH on main"
    continue
  fi

  count=$(printf '%s' "$unallowed" | sed '/^$/d' | wc -l | tr -d ' ')
  TOTAL_BYPASS=$((TOTAL_BYPASS + count))
  FAILED=1
  echo "FAIL  $repo: $count 2-parent commit(s) detected (admin-bypass suspected):"
  printf '%s' "$unallowed" | sed '/^$/d; s/^/      /'
done

# Fail if zero repos were auditable — this indicates a token scope problem
# that would produce a misleading green result.
if [[ "$AUDITED" -eq 0 ]]; then
  echo ""
  echo "ERROR: no repos were auditable. Check CROSS_REPO_PAT has org-wide read scope."
  echo "       CROSS_REPO_PAT is required for meaningful audit coverage (enforced at workflow level)."
  exit 2
fi

if [[ "$FAILED" -ne 0 ]]; then
  echo ""
  echo "ERROR: $TOTAL_BYPASS 2-parent merge commit(s) on main bypassed the merge queue."
  echo "       Direct merges via admin bypass produce 2-parent commits and skip the SQUASH ruleset."
  echo "       See OMN-8843. Verify rulesets with: bash scripts/verify-merge-queue-no-bypass.sh"
  exit 1
fi

echo ""
echo "All audited repos ($AUDITED): no merge-queue bypass detected on main."
