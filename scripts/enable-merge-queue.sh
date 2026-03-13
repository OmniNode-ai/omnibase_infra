#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# enable-merge-queue.sh — Enable GitHub Merge Queue on OmniNode repos via GraphQL rulesets
#
# Usage:
#   ./scripts/enable-merge-queue.sh                   # Enable on all 6 repos
#   ./scripts/enable-merge-queue.sh omnibase_core      # Enable on a single repo
#   ./scripts/enable-merge-queue.sh --validate         # Validate current config
#   ./scripts/enable-merge-queue.sh --dry-run          # Show what would be done
#
# Prerequisites:
#   - gh CLI authenticated with admin access to OmniNode-ai org
#   - Org plan must support merge queues (Team or Enterprise)
#
# Settings applied per repo:
#   Merge method:       squash
#   Group size:         min 1, max 5
#   Queue timeout:      60 minutes
#   Require up-to-date: YES (ALLGREEN strategy)
#   Direct merge:       admin-only (org admin bypass with ALWAYS mode)
#
# OMN-2818

set -euo pipefail

ORG="OmniNode-ai"
REPOS=(omniclaude omnibase_core omnibase_infra omnibase_spi omniintelligence omnimemory)

# Merge queue settings
MQ_MERGE_METHOD="SQUASH"
MQ_GROUPING_STRATEGY="ALLGREEN"
MQ_MIN_ENTRIES=1
MQ_MAX_ENTRIES_BUILD=5
MQ_MAX_ENTRIES_MERGE=5
MQ_TIMEOUT_MINUTES=60
MQ_MIN_WAIT_MINUTES=0
MQ_RULESET_NAME="Merge Queue"

DRY_RUN=false
VALIDATE_ONLY=false

# --- Helpers ---

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN: $*" >&2; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

usage() {
  head -18 "$0" | tail -16 | sed 's/^# //' | sed 's/^#//'
  exit 0
}

get_repo_id() {
  local repo="$1"
  gh api graphql -f query="
    { repository(owner: \"${ORG}\", name: \"${repo}\") { id } }
  " --jq '.data.repository.id'
}

check_existing_mq_ruleset() {
  local repo="$1"
  gh api graphql -f query="
    {
      repository(owner: \"${ORG}\", name: \"${repo}\") {
        rulesets(first: 20) {
          nodes {
            id
            name
            rules(first: 10) {
              nodes { type }
            }
          }
        }
      }
    }
  " --jq '.data.repository.rulesets.nodes[] | select(.rules.nodes[] | .type == "MERGE_QUEUE") | .id' 2>/dev/null || true
}

enable_merge_queue() {
  local repo="$1"
  local repo_id

  # Check for existing MQ ruleset
  local existing_id
  existing_id=$(check_existing_mq_ruleset "$repo")
  if [[ -n "$existing_id" ]]; then
    log "  ${repo}: Merge queue ruleset already exists (${existing_id}), skipping"
    return 0
  fi

  repo_id=$(get_repo_id "$repo")
  if [[ -z "$repo_id" || "$repo_id" == "null" ]]; then
    warn "${repo}: Could not resolve repository ID"
    return 1
  fi

  if $DRY_RUN; then
    log "  ${repo}: [DRY RUN] Would create merge queue ruleset (repo_id=${repo_id})"
    return 0
  fi

  local result
  if ! result=$(gh api graphql -f query="
    mutation {
      createRepositoryRuleset(input: {
        sourceId: \"${repo_id}\"
        name: \"${MQ_RULESET_NAME}\"
        target: BRANCH
        enforcement: ACTIVE
        conditions: {
          refName: {
            include: [\"~DEFAULT_BRANCH\"]
            exclude: []
          }
        }
        rules: [
          {
            type: MERGE_QUEUE
            parameters: {
              mergeQueue: {
                checkResponseTimeoutMinutes: ${MQ_TIMEOUT_MINUTES}
                groupingStrategy: ${MQ_GROUPING_STRATEGY}
                maxEntriesToBuild: ${MQ_MAX_ENTRIES_BUILD}
                maxEntriesToMerge: ${MQ_MAX_ENTRIES_MERGE}
                mergeMethod: ${MQ_MERGE_METHOD}
                minEntriesToMerge: ${MQ_MIN_ENTRIES}
                minEntriesToMergeWaitMinutes: ${MQ_MIN_WAIT_MINUTES}
              }
            }
          }
        ]
        bypassActors: [
          {
            organizationAdmin: true
            bypassMode: ALWAYS
          }
        ]
      }) {
        ruleset {
          id
          name
          enforcement
        }
      }
    }
  " 2>&1); then
    warn "${repo}: gh api graphql call failed"
    echo "$result" >&2
    return 1
  fi

  local ruleset_id
  ruleset_id=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['createRepositoryRuleset']['ruleset']['id'])" 2>/dev/null || true)

  if [[ -n "$ruleset_id" ]]; then
    log "  ${repo}: Merge queue enabled (ruleset_id=${ruleset_id})"
  else
    warn "${repo}: Failed to enable merge queue"
    echo "$result" >&2
    return 1
  fi
}

validate_repo() {
  local repo="$1"
  local result
  if ! result=$(gh api graphql -f query="
    {
      repository(owner: \"${ORG}\", name: \"${repo}\") {
        mergeQueue(branch: \"main\") {
          configuration {
            mergeMethod
            minimumEntriesToMerge
            maximumEntriesToBuild
            maximumEntriesToMerge
            mergingStrategy
            checkResponseTimeout
          }
        }
        rulesets(first: 10) {
          nodes {
            name
            enforcement
            bypassActors(first: 5) {
              nodes { bypassMode }
            }
          }
        }
      }
    }
  " 2>&1); then
    warn "${repo}: gh api graphql call failed"
    echo "$result" >&2
    return 1
  fi

  printf '%s' "$result" | python3 -c "
import json, sys
data = json.load(sys.stdin)['data']['repository']
mq = data['mergeQueue']
rulesets = data['rulesets']['nodes']
repo = sys.argv[1]

if mq is None:
    print(f'FAIL: {repo} - merge queue not configured')
    sys.exit(1)

cfg = mq['configuration']
issues = []
if cfg['mergeMethod'] != 'SQUASH':
    issues.append(f\"mergeMethod={cfg['mergeMethod']}\")
if cfg['minimumEntriesToMerge'] != 1:
    issues.append(f\"minEntries={cfg['minimumEntriesToMerge']}\")
if cfg['maximumEntriesToBuild'] != 5:
    issues.append(f\"maxBuild={cfg['maximumEntriesToBuild']}\")
if cfg['maximumEntriesToMerge'] != 5:
    issues.append(f\"maxMerge={cfg['maximumEntriesToMerge']}\")
if cfg['mergingStrategy'] != 'ALLGREEN':
    issues.append(f\"strategy={cfg['mergingStrategy']}\")
if cfg['checkResponseTimeout'] != 3600:
    issues.append(f\"timeout={cfg['checkResponseTimeout']}\")

has_bypass = any(
    a['bypassMode'] == 'ALWAYS'
    for r in rulesets
    if r.get('name') == 'Merge Queue'
    for a in r.get('bypassActors', {}).get('nodes', [])
)
if not has_bypass:
    issues.append('missing admin bypass')

if issues:
    print(f\"WARN: {repo} - {', '.join(issues)}\")
    sys.exit(1)
else:
    print(f\"OK:   {repo} - squash, group 1-5, 60min timeout, admin bypass\")
    sys.exit(0)
" "$repo"
}

# --- Main ---

# Parse arguments
targets=()
for arg in "$@"; do
  case "$arg" in
    --help|-h)    usage ;;
    --dry-run)    DRY_RUN=true ;;
    --validate)   VALIDATE_ONLY=true ;;
    *)            targets+=("$arg") ;;
  esac
done

# Default to all repos if no targets specified
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=("${REPOS[@]}")
fi

# Verify gh auth
if ! gh auth status &>/dev/null; then
  die "gh CLI not authenticated. Run 'gh auth login' first."
fi

if $VALIDATE_ONLY; then
  log "Validating merge queue configuration..."
  failures=0
  for repo in "${targets[@]}"; do
    if ! validate_repo "$repo"; then
      ((failures++))
    fi
  done
  echo ""
  if [[ $failures -eq 0 ]]; then
    log "All ${#targets[@]} repos: Merge Queue correctly configured"
  else
    die "${failures} repo(s) failed validation"
  fi
  exit 0
fi

log "Enabling merge queue on ${#targets[@]} repos..."
if $DRY_RUN; then
  log "(DRY RUN mode - no changes will be made)"
fi
echo ""

failures=0
for repo in "${targets[@]}"; do
  if ! enable_merge_queue "$repo"; then
    ((failures++))
  fi
done

echo ""
if [[ $failures -eq 0 ]]; then
  log "Done. Merge queue enabled on ${#targets[@]} repos."
  echo ""
  log "Validating..."
  validation_failures=0
  for repo in "${targets[@]}"; do
    if ! validate_repo "$repo"; then
      ((validation_failures++))
    fi
  done
  if [[ $validation_failures -ne 0 ]]; then
    die "${validation_failures} repo(s) failed post-enable validation"
  fi
else
  die "${failures} repo(s) failed"
fi
