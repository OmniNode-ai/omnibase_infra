#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
#
# test_preflight.sh -- Unit tests for preflight-check.sh
#
# Run from the omnibase_infra worktree root:
#   bash tests/redeploy/test_preflight.sh

set -euo pipefail

# Test: PREFLIGHT fails when required vars are missing
unset ENABLE_ENV_SYNC_PROBE
unset KAFKA_BOOTSTRAP_SERVERS
unset POSTGRES_PASSWORD
unset OMNI_HOME

result=$(bash scripts/preflight-check.sh 2>&1; echo "EXIT:$?") || true
echo "$result" | grep -q "MISSING.*ENABLE_ENV_SYNC_PROBE" || {
  echo "FAIL: expected MISSING ENABLE_ENV_SYNC_PROBE, got: $result"; exit 1
}
echo "PASS: preflight correctly reports missing vars"

# Test: PREFLIGHT passes (exit 0 or 2/advisory only) when all required vars set and tunnel check skipped
# Note: exit 2 is advisory (non-blocking VirtioFS warning) — treated as pass for this test.
# pipefail would cause a pipe with exit 2 on the left side to fail, so we capture separately.
pass_output=$(OMNI_HOME="/tmp" \
KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-${TEST_KAFKA_HOST:-localhost}:19092}" \
POSTGRES_PASSWORD="test" \
ENABLE_ENV_SYNC_PROBE="true" \
  bash scripts/preflight-check.sh --skip-tunnel-check 2>&1) || pass_exit=$?
pass_exit="${pass_exit:-0}"
# exit 0 = all clear; exit 2 = advisory warnings only (non-blocking) — both are acceptable
if [[ "${pass_exit}" -eq 1 ]]; then
  echo "FAIL: expected exit 0 or 2 with --skip-tunnel-check, got exit 1. Output: ${pass_output}"
  exit 1
fi
echo "${pass_output}" | grep -q "OK" || {
  echo "FAIL: expected OK in output with --skip-tunnel-check, got: ${pass_output}"; exit 1
}
echo "PASS: preflight skips tunnel check when flag set (exit ${pass_exit})"
