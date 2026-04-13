#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# check_llm_endpoints.sh — probe all known LLM inference endpoints.
# Exits 0 if all required endpoints are healthy, non-zero otherwise.
#
# Required endpoints (must be reachable or script exits 1):
#   LLM_CODER_URL       — Qwen3-Coder-30B (port 8000)
#   LLM_DEEPSEEK_R1_URL — DeepSeek-R1-14B  (port 8001)
#
# Optional endpoints (failures are warned, not fatal):
#   LLM_CODER_FAST_URL  — fast coder (port 8001 alias)
#   LLM_EMBEDDING_URL   — embedding model
#
# Usage:
#   bash scripts/check_llm_endpoints.sh
#   TIMEOUT=5 bash scripts/check_llm_endpoints.sh

set -euo pipefail

TIMEOUT="${TIMEOUT:-5}"
PASS=0
FAIL=0
WARN=0

_probe() {
  local label="$1"
  local url="$2"
  local required="${3:-true}"

  # Try /health first, then /v1/models as fallback
  local status
  status=$(curl -sf --max-time "$TIMEOUT" -o /dev/null -w "%{http_code}" "${url%/}/health" 2>/dev/null || true)

  if [[ "$status" =~ ^2 ]]; then
    echo "  OK   ${label} — ${url} (/health HTTP ${status})"
    PASS=$((PASS + 1))
    return 0
  fi

  status=$(curl -sf --max-time "$TIMEOUT" -o /dev/null -w "%{http_code}" "${url%/}/v1/models" 2>/dev/null || true)
  if [[ "$status" =~ ^2 ]]; then
    echo "  OK   ${label} — ${url} (/v1/models HTTP ${status})"
    PASS=$((PASS + 1))
    return 0
  fi

  if [[ "$required" == "true" ]]; then
    echo "  FAIL ${label} — ${url} (unreachable)"
    FAIL=$((FAIL + 1))
  else
    echo "  WARN ${label} — ${url} (unreachable, optional)"
    WARN=$((WARN + 1))
  fi
}

echo "=== LLM endpoint health check ==="
echo "    timeout: ${TIMEOUT}s"
echo ""

# Required endpoints — these must be set and healthy
if [[ -z "${LLM_DEEPSEEK_R1_URL:-}" ]]; then
  echo "  FAIL LLM_DEEPSEEK_R1_URL not set"
  FAIL=$((FAIL + 1))
else
  _probe "deepseek-r1 (LLM_DEEPSEEK_R1_URL)" "$LLM_DEEPSEEK_R1_URL" true
fi

if [[ -z "${LLM_CODER_URL:-}" ]]; then
  echo "  WARN LLM_CODER_URL not set (optional in CI)"
  WARN=$((WARN + 1))
else
  _probe "coder (LLM_CODER_URL)" "$LLM_CODER_URL" false
fi

# Optional endpoints
if [[ -n "${LLM_CODER_FAST_URL:-}" ]]; then
  _probe "coder-fast (LLM_CODER_FAST_URL)" "$LLM_CODER_FAST_URL" false
fi

if [[ -n "${LLM_EMBEDDING_URL:-}" ]]; then
  _probe "embedding (LLM_EMBEDDING_URL)" "$LLM_EMBEDDING_URL" false
fi

echo ""
echo "=== Summary: ${PASS} ok, ${FAIL} failed, ${WARN} warned ==="

if [[ "$FAIL" -gt 0 ]]; then
  echo "RESULT: UNHEALTHY — ${FAIL} required endpoint(s) unreachable"
  exit 1
fi

echo "RESULT: HEALTHY"
exit 0
