#!/bin/bash
# ============================================================================
# Remote Integration Test Runner
# ============================================================================
# Runs integration tests against remote production-like infrastructure
# Realistic network conditions, validates actual deployment setup
# ============================================================================

set -e

echo "🌐 Configuring for REMOTE test environment..."
source .env.test.remote

echo ""
echo "📋 Configuration:"
echo "  - Kafka: ${KAFKA_BOOTSTRAP_SERVERS} (via /etc/hosts → 192.168.86.200)"
echo "  - PostgreSQL: ${POSTGRES_HOST}:${POSTGRES_PORT}"
echo "  - Consul: ${CONSUL_HOST}:${CONSUL_PORT}"
echo "  - Latency Threshold: ${TEST_KAFKA_LATENCY_MS}ms (relaxed for network)"
echo "  - Throughput Threshold: ${TEST_KAFKA_THROUGHPUT} msg/s"
echo ""

# Verify /etc/hosts configuration
echo "🔍 Verifying /etc/hosts configuration..."
if grep -q "omninode-bridge-redpanda" /etc/hosts; then
    RESOLVED_IP=$(grep "omninode-bridge-redpanda" /etc/hosts | awk '{print $1}' | head -1)
    echo "✅ omninode-bridge-redpanda → ${RESOLVED_IP}"
else
    echo "⚠️  WARNING: omninode-bridge-redpanda not in /etc/hosts"
    echo "   Tests may use local Docker container instead of remote"
fi

echo ""
echo "🧪 Running integration tests against REMOTE infrastructure..."
poetry run pytest tests/integration/ "$@"
