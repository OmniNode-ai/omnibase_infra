#!/bin/bash
# ============================================================================
# Local Integration Test Runner
# ============================================================================
# Runs integration tests against local Docker infrastructure
# Fast, isolated, optimized for development iteration
# ============================================================================

set -e

echo "🔧 Configuring for LOCAL test environment..."
source .env.test.local

echo ""
echo "📋 Configuration:"
echo "  - Kafka: ${KAFKA_BOOTSTRAP_SERVERS}"
echo "  - PostgreSQL: ${POSTGRES_HOST}:${POSTGRES_PORT}"
echo "  - Consul: ${CONSUL_HOST}:${CONSUL_PORT}"
echo "  - Latency Threshold: ${TEST_KAFKA_LATENCY_MS}ms"
echo "  - Throughput Threshold: ${TEST_KAFKA_THROUGHPUT} msg/s"
echo ""

# Ensure local containers are running
echo "🐳 Checking local Docker containers..."
if ! docker ps | grep -q "omninode-bridge-redpanda"; then
    echo "⚠️  Local Redpanda not running. Starting containers..."
    docker compose -f deployment/docker-compose.yml up -d omninode-bridge-redpanda
    echo "⏳ Waiting 10s for Redpanda to be ready..."
    sleep 10
fi

echo "✅ Local infrastructure ready"
echo ""
echo "🧪 Running integration tests..."
poetry run pytest tests/integration/ "$@"
