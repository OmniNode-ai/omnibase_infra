#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# OMN-3554: Reject hardcoded Kafka broker address fallbacks
# Detects os.getenv("KAFKA_*", non-empty) and private-IP Kafka broker ports
#
# Suppressions:
#   # kafka-fallback-ok         — intentional test fixture default
#   # noqa                      — general suppression
#   # onex-allow-internal-ip    — R2 only: intentional private-IP reference

set -euo pipefail

FAILED=0

# R1: os.getenv("KAFKA_*", non-empty) pattern
MATCHES=$(grep -rn --include="*.py" \
    --exclude-dir=".venv" \
    --exclude-dir="node_modules" \
    -E "os\.getenv\([[:space:]]*[\"']KAFKA_[^\"']+[\"'][[:space:]]*,[[:space:]]*[\"'][^\"']+[\"']" \
    . 2>/dev/null | \
    grep -v "# kafka-fallback-ok" | \
    grep -v "# noqa" || true)

if [ -n "$MATCHES" ]; then
    echo "ERROR: Hardcoded Kafka bootstrap fallback detected:"
    echo "$MATCHES"
    echo ""
    echo "FIX: Replace os.getenv(\"KAFKA_...\", \"fallback\") with:"
    echo "  os.environ[\"KAFKA_BOOTSTRAP_SERVERS\"]  # fails loudly when unset"
    echo "  os.getenv(\"KAFKA_BOOTSTRAP_SERVERS\")   # returns None when unset"
    echo "  If intentional (test fixture): add # kafka-fallback-ok"
    FAILED=1
fi

# R2: Private-IP Kafka broker addresses (Kafka-specific ports only)
IP_MATCHES=$(grep -rn --include="*.py" \
    --exclude-dir=".venv" \
    --exclude-dir="node_modules" \
    -E "192\.168\.[0-9]+\.[0-9]+:(9092|19092|29092|29093)" \
    . 2>/dev/null | \
    grep -v "# kafka-fallback-ok" | \
    grep -v "# noqa" | \
    grep -v "# onex-allow-internal-ip" || true)

if [ -n "$IP_MATCHES" ]; then
    echo "ERROR: Hardcoded private-IP Kafka broker address in Python file:"
    echo "$IP_MATCHES"
    echo ""
    echo "FIX: Use KAFKA_BOOTSTRAP_SERVERS env var."
    echo "  If intentional (test fixture): add # kafka-fallback-ok"
    FAILED=1
fi

exit $FAILED
