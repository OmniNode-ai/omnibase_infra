#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
#
# preflight-check.sh -- Pre-flight gate for /redeploy skill
#
# Checks required env vars, bus tunnel reachability, and VirtioFS bind-mount
# conflicts before any deployment action runs.
#
# Usage:
#   bash scripts/preflight-check.sh
#   bash scripts/preflight-check.sh --skip-tunnel-check
#
# Exit codes:
#   0 -- All checks pass
#   1 -- One or more REQUIRED checks failed
#   2 -- Advisory warnings only (non-blocking)

set -euo pipefail

SKIP_TUNNEL_CHECK=false
MISSING=()
WARNINGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tunnel-check) SKIP_TUNNEL_CHECK=true; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ── Required env vars ──────────────────────────────────────────────────────

REQUIRED_VARS=(
  POSTGRES_PASSWORD
  KAFKA_BOOTSTRAP_SERVERS
  OMNI_HOME
  ENABLE_ENV_SYNC_PROBE
)

for var in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "PREFLIGHT MISSING: ${var}"
    MISSING+=("${var}")
  else
    echo "PREFLIGHT OK: ${var}=${!var}"
  fi
done

# ── Bus tunnel reachability ────────────────────────────────────────────────

if [[ "${SKIP_TUNNEL_CHECK}" == false ]]; then
  if nc -z localhost 29092 2>/dev/null; then  # cloud-bus-ok OMN-4922
    echo "PREFLIGHT OK: cloud bus tunnel reachable (localhost:29092)"  # cloud-bus-ok OMN-4922
  else
    echo "PREFLIGHT MISSING: cloud bus tunnel not reachable at localhost:29092"  # cloud-bus-ok OMN-4922
    MISSING+=("cloud-bus-tunnel")
  fi
fi

# ── VirtioFS bind-mount conflict detection ─────────────────────────────────
#
# Check if ../contracts/nodes would be a valid path from the deploy root.
# When run from a worktree, the relative paths in docker-compose are:
#   ../contracts:/app/contracts:ro
#   ../src/omnibase_infra/nodes:/app/contracts/nodes:ro
# If the worktree lacks a contracts/ sibling directory, the second mount
# overwrites the first's subdirectory with an empty or missing directory.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "${REPO_ROOT}")"

if [[ ! -d "${PARENT_DIR}/contracts" ]]; then
  echo "PREFLIGHT WARNING: ../contracts not found relative to repo root (${PARENT_DIR}/contracts)"
  echo "  The bind-mount ../contracts:/app/contracts:ro will fail or mount empty dir."
  echo "  Ensure deploy-runtime.sh rsync has run before containers start."
  WARNINGS+=("VirtioFS-contracts-missing")
fi

# ── Summary ───────────────────────────────────────────────────────────────

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo ""
  echo "PREFLIGHT FAILED: ${#MISSING[@]} required check(s) failed:"
  for m in "${MISSING[@]}"; do
    echo "  - ${m}"
  done
  echo ""
  echo "Fix required before deploy can proceed."
  exit 1
fi

if [[ ${#WARNINGS[@]} -gt 0 ]]; then
  echo ""
  echo "PREFLIGHT WARNINGS: ${#WARNINGS[@]} advisory issue(s):"
  for w in "${WARNINGS[@]}"; do
    echo "  - ${w}"
  done
  exit 2
fi

echo ""
echo "PREFLIGHT OK: all checks passed"
exit 0
