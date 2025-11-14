#!/bin/bash
set -e

echo "🚀 Starting OmniBase Infrastructure Services..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo "⚠️  Please edit .env and update with your values before continuing"
    echo ""
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

echo "📦 Building Docker images..."
docker-compose -f docker-compose.infrastructure.yml build

echo ""
echo "🔧 Starting infrastructure services..."
docker-compose -f docker-compose.infrastructure.yml up -d consul postgres redpanda

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

echo ""
echo "🎯 Creating RedPanda topics..."
docker-compose -f docker-compose.infrastructure.yml up -d redpanda-topic-manager

echo ""
echo "🌐 Starting adapters..."
docker-compose -f docker-compose.infrastructure.yml up -d postgres-adapter consul-adapter

echo ""
echo "✅ Infrastructure startup complete!"
echo ""
echo "📊 Service Status:"
docker-compose -f docker-compose.infrastructure.yml ps

echo ""
echo "🔗 Access Points:"
echo "  - Consul UI:        http://localhost:${CONSUL_PORT:-8500}"
echo "  - PostgreSQL:       localhost:5435"
echo "  - RedPanda:         localhost:${REDPANDA_PORT:-9092}"
echo "  - RedPanda UI:      http://localhost:${REDPANDA_UI_PORT:-8080}"
echo "  - PostgreSQL Adapter: http://localhost:${POSTGRES_ADAPTER_PORT:-8081}/health"
echo "  - Consul Adapter:   http://localhost:${CONSUL_ADAPTER_PORT:-8082}/health"
echo ""
echo "📝 Logs:"
echo "  docker-compose -f docker-compose.infrastructure.yml logs -f [service-name]"
echo ""
echo "🛑 Stop:"
echo "  docker-compose -f docker-compose.infrastructure.yml down"
