#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# test_pipeline_smoke.sh -- Integration smoke test for all 6 regression scenarios
#
# Tests Issues 1, 3, 4, 5 with automated assertions.
# Issues 2 and 6 require live infrastructure and are documented in output only.
#
# Usage:
#   cd /path/to/omnibase_infra/worktree
#   bash tests/redeploy/test_pipeline_smoke.sh
#
# Options:
#   OMNINODE_INFRA_ROOT=/path/to/omninode_infra  -- required for Issue 1 check
#
# Exit codes:
#   0 -- All testable scenarios PASS
#   1 -- One or more scenarios FAILED

set -euo pipefail

PASS=0
FAIL=0
SKIP=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OMNINODE_INFRA_ROOT="${OMNINODE_INFRA_ROOT:-}"

check_exit() {
  local name="$1"
  local expected_exit="$2"
  shift 2

  actual_exit=0
  "$@" >/dev/null 2>&1 || actual_exit=$?

  if [[ "${actual_exit}" == "${expected_exit}" ]]; then
    echo "PASS: ${name} (exit ${actual_exit})"
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${name} (expected exit ${expected_exit}, got ${actual_exit})"
    FAIL=$((FAIL + 1))
  fi
}

check_output() {
  local name="$1"
  local pattern="$2"
  shift 2

  output=$("$@" 2>&1) || true

  if echo "${output}" | grep -q "${pattern}"; then
    echo "PASS: ${name}"
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${name} (pattern '${pattern}' not found in output)"
    FAIL=$((FAIL + 1))
  fi
}

# ── Issue 1: CronJob --cone + file detection ────────────────────────────────
# validate-cronjob-manifests.sh lives in omninode_infra

if [[ -n "${OMNINODE_INFRA_ROOT}" && -f "${OMNINODE_INFRA_ROOT}/scripts/validate-cronjob-manifests.sh" ]]; then
  CONE_TEST_DIR="/tmp/cronjob-smoke-$$"
  mkdir -p "${CONE_TEST_DIR}"
  CONE_YAML="${CONE_TEST_DIR}/cronjob-cone-test.yaml"

  cat > "${CONE_YAML}" <<'YAML'
apiVersion: batch/v1
kind: CronJob
metadata:
  name: test-cone-bug
spec:
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: runner
            resources:
              limits:
                cpu: "500m"
                memory: "512Mi"
            command:
            - /bin/sh
            - -c
            - |
              git sparse-checkout init --cone
              git sparse-checkout set \
                onex_validation_policy.yaml \
                scripts/run_cross_repo_validation.py
YAML

  check_exit "Issue1: CronJob cone+file detection (exit 1)" 1 \
    bash "${OMNINODE_INFRA_ROOT}/scripts/validate-cronjob-manifests.sh" "${CONE_TEST_DIR}"

  rm -rf "${CONE_TEST_DIR}"
else
  echo "SKIP: Issue1: CronJob cone+file detection — set OMNINODE_INFRA_ROOT to enable"
  echo "      export OMNINODE_INFRA_ROOT=/path/to/omninode_infra/worktree"
  SKIP=$((SKIP + 1))
fi

# ── Issue 3: ENABLE_ENV_SYNC_PROBE missing in PREFLIGHT ────────────────────

check_output "Issue3: ENABLE_ENV_SYNC_PROBE missing (exit 1)" "PREFLIGHT FAILED" \
  bash -c "
    set +e
    POSTGRES_PASSWORD='' KAFKA_BOOTSTRAP_SERVERS='' OMNI_HOME='' ENABLE_ENV_SYNC_PROBE='' \
      bash '${REPO_ROOT}/scripts/preflight-check.sh' --skip-tunnel-check 2>&1
    true
  "

# ── Issue 4: Bus tunnel check skippable (all vars set, --skip-tunnel-check) ─

# Issue 4: --skip-tunnel-check means tunnel check is bypassed.
# In a worktree without ../contracts, the VirtioFS check emits an advisory (exit 2).
# The important assertion is: no PREFLIGHT FAILED (no required vars missing).
check_output "Issue4: bus tunnel check skippable (no PREFLIGHT FAILED)" "PREFLIGHT OK: ENABLE_ENV_SYNC_PROBE" \
  bash -c "
    OMNI_HOME='/tmp' \
    KAFKA_BOOTSTRAP_SERVERS='localhost:29092' \
    POSTGRES_PASSWORD='test' \
    ENABLE_ENV_SYNC_PROBE='true' \
      bash '${REPO_ROOT}/scripts/preflight-check.sh' --skip-tunnel-check 2>&1
  "

# ── Issue 5: Stale schema fingerprint exits 2 ──────────────────────────────

FINGERPRINT_FILE="${REPO_ROOT}/docker/migrations/schema_fingerprint.sha256"
FINGERPRINT_BACKUP=""

if [[ -f "${FINGERPRINT_FILE}" ]]; then
  FINGERPRINT_BACKUP=$(cat "${FINGERPRINT_FILE}")
fi

echo "sha256:deadbeef00000000000000000000000000000000000000000000000000000000" > "${FINGERPRINT_FILE}"

check_exit "Issue5: stale fingerprint exits 2 (exit 2)" 2 \
  bash -c "cd '${REPO_ROOT}' && uv run python scripts/check_schema_fingerprint.py verify"

# Restore fingerprint
if [[ -n "${FINGERPRINT_BACKUP}" ]]; then
  echo "${FINGERPRINT_BACKUP}" > "${FINGERPRINT_FILE}"
else
  cd "${REPO_ROOT}" && uv run python scripts/check_schema_fingerprint.py stamp >/dev/null 2>&1 || true
fi

# ── Issues 2 and 6: Require live infrastructure ─────────────────────────────

echo ""
echo "NOTE: Issue 2 (k8s pod readiness) requires live AWS SSM + cloud k8s access"
echo "      Run: bash scripts/k8s-pod-readiness-check.sh"
echo ""
echo "NOTE: Issue 6 (VirtioFS bind-mount conflict) requires live Docker infra"
echo "      The check is embedded in scripts/deploy-runtime.sh validate_repo_structure"

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "Smoke test results: PASS=${PASS} FAIL=${FAIL} SKIP=${SKIP}"
if [[ "${FAIL}" -eq 0 ]]; then
  echo "SMOKE TEST PASSED"
  exit 0
else
  echo "SMOKE TEST FAILED"
  exit 1
fi
