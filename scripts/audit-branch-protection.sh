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
#   --fix             Remove required_pull_request_reviews via PUT (MAIN ONLY; dangerous)
#   --dry-run         Report violations but do not mutate (default when --fix absent)
#   --help            Show this help
#
# Exit codes:
#   0  — no violations found
#   1  — one or more violations found (or --fix ran but could not resolve)
#
# Audits BOTH `main` (the release boundary) AND `dev` (the everyday merge target)
# per repo, with per-branch attribution (OMN-14696). `dev` drift was previously
# invisible because the audit was hardcoded to `main`. Rule #5: enforcement, not
# detection.
#
# Per-branch checks (run for main AND dev):
#   Check A (reviews): approving reviews must NOT be enforced (blocks solo-dev
#            merges). Judged via GraphQL `requiresApprovingReviews` (authoritative),
#            NOT the REST required_approving_review_count — REST reports a phantom
#            count of 1 for a protected-but-no-review branch, which false-fails dev.
#
# Main-only checks (never run for dev — these are main-scoped):
#   Check B (orphans): each required_status_checks.contexts[] must match a check-run
#            seen in the last 5 commits on the DEFAULT branch (a main-scoped read;
#            the `main-target-guard` PR-only allowlist is main-specific).
#
# SAFETY — the `dev` extension is strictly READ/AUDIT ONLY:
#   `--fix` PUTs to `branches/main/protection` and stays MAIN-ONLY. It is gated on
#   `branch == "main"`, the PUT URL is a hardcoded `main` literal (never $branch),
#   and the audit returns an empty protection_json off main — so a `dev` violation
#   can NEVER build or PUT a payload to a dev branch's protection.
#
# DEV-EXEMPT repos (audited on `main` only): omnistream (no dev branch), omniweb
#   (dev exists but intentionally unprotected) — mirrors onex_change_control#4281.
#
# This script is a thin wrapper around scripts/audit_branch_protection_lib.py
# so the Check A / Check B logic is unit-testable without spawning subprocesses.
#
# OMN-9034, OMN-14696 (dev coverage), OMN-14683

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

# Branches audited per repo. `main` is always audited; `dev` is audited too
# (OMN-14696) unless the repo is dev-exempt. Override with a space-separated env
# list (BRANCH_PROTECTION_AUDIT_BRANCHES) for targeted runs.
BRANCHES=(main dev)
if [[ -n "${BRANCH_PROTECTION_AUDIT_BRANCHES:-}" ]]; then
  IFS=' ' read -r -a BRANCHES <<< "${BRANCH_PROTECTION_AUDIT_BRANCHES}"
fi

# Repos with no protected `dev` branch — audited on `main` only. Mirrors
# onex_change_control#4281 / OMN-14683 DEV_EXEMPT_REPOS so a legitimately
# unprotected `dev` does not false-fail. (omnistream has no `dev`; omniweb's
# `dev` exists but is intentionally unprotected.)
DEV_EXEMPT_REPOS=(omnistream omniweb)

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

is_dev_exempt() {
  local repo="$1"
  local p
  for p in "${DEV_EXEMPT_REPOS[@]}"; do
    if [[ "$p" == "$repo" ]]; then
      return 0
    fi
  done
  return 1
}

# Audit ONE branch of ONE repo. Increments `violations` on a failure and, for
# `main` under --fix, attempts remediation. Attributes every line to [branch].
check_branch() {
  local repo="$1"
  local branch="$2"
  local full="${OWNER}/${repo}"
  local result status message protection put_payload

  # Delegate Check A (+ Check B for main) to the lib (real gh injected caller).
  result=$(python3 "${SCRIPT_DIR}/audit_branch_protection_lib_cli.py" audit \
    --owner "${OWNER}" --repo "${repo}" --branch "${branch}")

  status=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  message=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['message'])")

  case "$status" in
    skip)
      echo "    [${branch}] [SKIP] ${message}"
      return 0
      ;;
    ok)
      echo "    [${branch}] [OK]   ${message}"
      return 0
      ;;
    violation)
      echo "    [${branch}] [FAIL] ${repo}: ${message}"
      violations=$((violations + 1))

      # --fix is MAIN-ONLY and NEVER mutates a non-main branch's protection.
      # Triple safety: (1) this block is gated on branch == "main"; (2) the audit
      # returns an empty protection_json off main, so no payload exists to PUT;
      # (3) the PUT URL below is a hardcoded `main` literal, never $branch.
      if [[ "$DO_FIX" -eq 1 && "$branch" == "main" ]]; then
        protection=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin)['protection_json'])")
        echo "    [main] [FIX]  Removing required_pull_request_reviews on ${repo}..."
        put_payload=$(python3 "${SCRIPT_DIR}/audit_branch_protection_lib_cli.py" fix-payload \
          --protection-json "$protection")
        if gh api --method PUT "repos/${full}/branches/main/protection" \
             --input <(echo "$put_payload") > /dev/null 2>&1; then
          echo "    [main] [OK]   required_pull_request_reviews removed on ${repo}"
          violations=$((violations - 1))
        else
          echo "    [main] [ERR]  Could not remove required_pull_request_reviews on ${repo}"
        fi
      elif [[ "$DO_FIX" -eq 1 ]]; then
        # Reached only for a NON-main violation under --fix — report-only, no PUT.
        echo "    [${branch}] [NOTE] --fix is main-only; ${branch} violation is report-only (no mutation)"
      fi
      ;;
  esac
}

# Audit every configured branch of ONE repo (dev skipped for dev-exempt repos).
check_repo() {
  local repo="$1"
  local branch
  for branch in "${BRANCHES[@]}"; do
    if [[ "$branch" == "dev" ]] && is_dev_exempt "$repo"; then
      echo "    [dev] [SKIP] ${repo}: dev-exempt (no protected dev branch)"
      continue
    fi
    check_branch "$repo" "$branch"
  done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "=== branch-protection audit: owner=${OWNER}, repos=${#REPOS[@]}, branches=${BRANCHES[*]}, fix=${DO_FIX} (main-only) ==="
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
