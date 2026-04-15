#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# OMN-8843: Verify no merge queue ruleset has OrganizationAdmin bypass.
# Fails with exit code 1 if any bypass_actors entry is found.
set -euo pipefail

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

ORG="OmniNode-ai"
FAILED=0

for repo in "${REPOS[@]}"; do
  ruleset_id=$(gh api "repos/$ORG/$repo/rulesets" 2>/dev/null \
    | jq -r '.[] | select(.name | test("Merge Queue"; "i")) | .id' | head -1)

  if [[ -z "$ruleset_id" ]]; then
    echo "SKIP  $repo: no Merge Queue ruleset found"
    continue
  fi

  bypass=$(gh api "repos/$ORG/$repo/rulesets/$ruleset_id" 2>/dev/null | jq -c '.bypass_actors')

  if [[ "$bypass" == "[]" ]]; then
    echo "OK    $repo (ruleset $ruleset_id): bypass_actors=[]"
  else
    echo "FAIL  $repo (ruleset $ruleset_id): bypass_actors=$bypass"
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo ""
  echo "ERROR: One or more merge queue rulesets have admin bypass. (OMN-8843)"
  exit 1
fi

echo ""
echo "All merge queue rulesets have no bypass actors."
