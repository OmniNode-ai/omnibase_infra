#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# verify-omnidash-health.sh -- OMNIDASH_VERIFY phase: data-source health gate
#
# Checks that omnidash is running and has sufficient live data sources.
# A live data source count below the threshold indicates the runtime is not
# producing events or is connected to the wrong bus.
#
# Note: This script proves omnidash responsiveness and live data-source health.
# It does NOT directly verify bus correctness — live-source count is a proxy
# indicator that events are flowing.
#
# Usage:
#   bash scripts/verify-omnidash-health.sh
#   OMNIDASH_URL=http://localhost:3001 bash scripts/verify-omnidash-health.sh
#   EXPECTED_LIVE=2 bash scripts/verify-omnidash-health.sh
#
# Exit codes:
#   0 -- omnidash running and >= EXPECTED_LIVE sources live
#   1 -- omnidash running but insufficient live sources
#   2 -- omnidash not running (advisory — does not block deploy)

set -euo pipefail

OMNIDASH_URL="${OMNIDASH_URL:-http://localhost:3000}"
EXPECTED_LIVE="${EXPECTED_LIVE:-3}"

# ── Check if omnidash is running ──────────────────────────────────────────

if ! curl -sf --connect-timeout 3 --max-time 5 "${OMNIDASH_URL}/api/health" >/dev/null 2>&1; then
  echo "OMNIDASH_VERIFY WARNING: omnidash not reachable at ${OMNIDASH_URL}/api/health (advisory)"
  echo "  omnidash may not be running locally — skipping data-source check"
  exit 2
fi

echo "OMNIDASH_VERIFY: omnidash reachable at ${OMNIDASH_URL}"

# ── Fetch data-source health ───────────────────────────────────────────────

HEALTH_RESPONSE=$(curl -sf --connect-timeout 3 --max-time 10 \
  "${OMNIDASH_URL}/api/health/data-sources" 2>&1) || {
    echo "OMNIDASH_VERIFY WARNING: could not fetch /api/health/data-sources (advisory)"
    exit 2
  }

# Parse counts using python3 (available on all OmniNode machines)
PARSE_RESULT=$(echo "${HEALTH_RESPONSE}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    sources = data.get('sources', data.get('dataSources', []))
    live = sum(1 for s in sources if s.get('status') == 'live')
    mock = sum(1 for s in sources if s.get('status') == 'mock')
    offline = sum(1 for s in sources if s.get('status') == 'offline')
    probe_disabled = sum(1 for s in sources if s.get('status') == 'probe_disabled')
    total = len(sources)
    print(f'live={live} mock={mock} offline={offline} probe_disabled={probe_disabled} total={total}')
except Exception as e:
    print(f'live=0 mock=0 offline=0 probe_disabled=0 total=0 error={e}')
" 2>/dev/null || echo "live=0 mock=0 offline=0 probe_disabled=0 total=0")

echo "OMNIDASH_VERIFY: data-source counts: ${PARSE_RESULT}"

LIVE_COUNT=$(echo "${PARSE_RESULT}" | grep -oE 'live=[0-9]+' | cut -d= -f2 || echo "0")
PROBE_DISABLED=$(echo "${PARSE_RESULT}" | grep -oE 'probe_disabled=[0-9]+' | cut -d= -f2 || echo "0")

# ── Advisory: probe_disabled signals missing ENABLE_ENV_SYNC_PROBE ─────────

if [[ "${PROBE_DISABLED}" -gt 0 ]]; then
  echo "OMNIDASH_VERIFY WARNING: ${PROBE_DISABLED} source(s) have probe_disabled status"
  echo "  This signals ENABLE_ENV_SYNC_PROBE is missing from the omnidash process env"
  echo "  Fix: ensure ENABLE_ENV_SYNC_PROBE=true is set before starting omnidash"
fi

# ── Gate: require sufficient live sources ─────────────────────────────────

if [[ "${LIVE_COUNT}" -ge "${EXPECTED_LIVE}" ]]; then
  echo ""
  echo "OMNIDASH_VERIFY OK: ${LIVE_COUNT} live sources >= threshold ${EXPECTED_LIVE}"
  exit 0
else
  echo ""
  echo "OMNIDASH_VERIFY FAILED: only ${LIVE_COUNT} live sources (threshold: ${EXPECTED_LIVE})"
  echo "  Runtime may be offline or events not flowing on the connected bus"
  exit 1
fi
