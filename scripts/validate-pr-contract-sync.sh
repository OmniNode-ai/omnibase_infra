#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# validate-pr-contract-sync.sh — Wave C CI gate [OMN-8915]
#
# Fails any PR that touches a node handler without also touching its
# contract.yaml, or a skill file without touching its SKILL.md / contract.yaml.
#
# Usage (CI — reads from gh pr diff):
#   validate-pr-contract-sync.sh <PR_NUMBER>
#
# Usage (test harness — reads from env):
#   VALIDATE_PR_FILES="file1\nfile2" \
#   VALIDATE_PR_COMMIT_MESSAGES="msg1\nmsg2" \
#   validate-pr-contract-sync.sh --from-env
#
# Exemption:
#   Include [skip-contract-sync] in any commit message to bypass the gate.
#   This must be accompanied by a justification in the commit body.
#
# Exit codes:
#   0: gate passes
#   1: contract sync violation found

set -euo pipefail

# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

if [[ "${1:-}" == "--from-env" ]]; then
    changed_files="${VALIDATE_PR_FILES:-}"
    commit_messages="${VALIDATE_PR_COMMIT_MESSAGES:-}"
elif [[ -n "${1:-}" ]]; then
    pr_number="$1"
    changed_files="$(gh pr diff "$pr_number" --name-only 2>/dev/null || true)"
    commit_messages="$(gh pr view "$pr_number" --json commits --jq '.commits[].messageHeadline' 2>/dev/null || true)"
else
    echo "ERROR: usage: $0 <PR_NUMBER> | --from-env" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Exemption check
# ---------------------------------------------------------------------------

if echo "$commit_messages" | grep -qF '[skip-contract-sync]'; then
    echo "INFO: [skip-contract-sync] token found — gate bypassed."
    exit 0
fi

# ---------------------------------------------------------------------------
# Collect node names that have handler changes or contract changes
# (bash 3.2-compatible — uses temp files instead of associative arrays)
# ---------------------------------------------------------------------------

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    # Node handler pattern: .../nodes/<node>/handlers/*.py
    if echo "$file" | grep -qE '.*/nodes/([^/]+)/handlers/[^/]+\.py$'; then
        node="$(echo "$file" | sed -E 's|.*/nodes/([^/]+)/handlers/[^/]+\.py$|\1|')"
        touch "$tmpdir/node_handler_${node}"
        continue
    fi

    # Node contract pattern: .../nodes/<node>/contract.yaml
    if echo "$file" | grep -qE '.*/nodes/([^/]+)/contract\.yaml$'; then
        node="$(echo "$file" | sed -E 's|.*/nodes/([^/]+)/contract\.yaml$|\1|')"
        touch "$tmpdir/node_contract_${node}"
        continue
    fi

    # Skill manifest: plugins/onex/skills/<skill>/(SKILL.md|contract.yaml)
    if echo "$file" | grep -qE 'plugins/onex/skills/([^/]+)/SKILL\.md$'; then
        skill="$(echo "$file" | sed -E 's|plugins/onex/skills/([^/]+)/SKILL\.md$|\1|')"
        touch "$tmpdir/skill_manifest_${skill}"
        continue
    fi
    if echo "$file" | grep -qE 'plugins/onex/skills/([^/]+)/contract\.yaml$'; then
        skill="$(echo "$file" | sed -E 's|plugins/onex/skills/([^/]+)/contract\.yaml$|\1|')"
        touch "$tmpdir/skill_manifest_${skill}"
        continue
    fi

    # Skill code/prompt: plugins/onex/skills/<skill>/<anything else>
    if echo "$file" | grep -qE 'plugins/onex/skills/([^/]+)/[^/]+$'; then
        skill="$(echo "$file" | sed -E 's|plugins/onex/skills/([^/]+)/[^/]+$|\1|')"
        touch "$tmpdir/skill_code_${skill}"
        continue
    fi
done <<< "$changed_files"

# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

violations=()

# Nodes with handler changes but no contract change
for marker in "$tmpdir"/node_handler_*; do
    [[ -e "$marker" ]] || continue
    node="${marker#"$tmpdir/node_handler_"}"
    if [[ ! -e "$tmpdir/node_contract_${node}" ]]; then
        violations+=("NODE  $node: handler changed but contract.yaml not updated")
    fi
done

# Skills with code changes but no manifest change
for marker in "$tmpdir"/skill_code_*; do
    [[ -e "$marker" ]] || continue
    skill="${marker#"$tmpdir/skill_code_"}"
    if [[ ! -e "$tmpdir/skill_manifest_${skill}" ]]; then
        violations+=("SKILL $skill: skill file changed but SKILL.md / contract.yaml not updated")
    fi
done

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

if [[ ${#violations[@]} -eq 0 ]]; then
    echo "OK: contract-sync gate passed."
    exit 0
fi

echo "FAIL: contract-sync gate — the following nodes/skills were changed without updating their contracts:"
echo ""
for v in "${violations[@]}"; do
    echo "  $v"
done
echo ""
echo "Fix: update the contract.yaml (node) or SKILL.md/contract.yaml (skill) to reflect your changes."
echo "Bypass: add [skip-contract-sync] to a commit message with justification."
exit 1
