#!/bin/bash
# migrate-to-remote.sh - Single script to migrate OmniNode Bridge to remote system
# Run this on your LOCAL machine

set -e

# Configuration
REMOTE_HOST="192.168.86.200"
REMOTE_USER="jonah"  # CHANGE THIS TO YOUR ACTUAL USERNAME
REMOTE_PATH="/Users/$REMOTE_USER/omninode_bridge"

echo "🚀 OmniNode Bridge Migration to $REMOTE_HOST"
echo "=============================================="

# Check if remote host is reachable
echo "📡 Checking connectivity to $REMOTE_HOST..."
if ! ping -c 1 -W 3000 $REMOTE_HOST > /dev/null 2>&1; then
    echo "❌ Cannot reach $REMOTE_HOST. Please check network connectivity."
    exit 1
fi
echo "✅ Remote host is reachable"

# Setup SSH key if needed
echo "🔑 Setting up SSH access..."
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating SSH key..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" -q
fi

# Copy SSH key to remote (will prompt for password)
echo "Copying SSH key to remote system (you'll need to enter password)..."
ssh-copy-id "$REMOTE_USER@$REMOTE_HOST" 2>/dev/null || {
    echo "❌ SSH key setup failed. Please ensure you can SSH to $REMOTE_HOST"
    exit 1
}

# Test SSH connection
echo "Testing SSH connection..."
ssh -o BatchMode=yes "$REMOTE_USER@$REMOTE_HOST" "echo 'SSH connection successful'" || {
    echo "❌ SSH connection failed. Please check credentials."
    exit 1
}

# Export Docker images
echo "📦 Exporting Docker images..."
mkdir -p ./docker-exports
rm -f ./docker-exports/*.tar.gz

# Export all omninode_bridge images
echo "Exporting omninode_bridge images..."
docker images | grep omninode_bridge | awk '{print $1":"$2}' | while read image; do
    echo "  - $image"
    docker save "$image" | gzip > "./docker-exports/${image//\//_}.tar.gz"
done

# Export infrastructure images
echo "Exporting infrastructure images..."
docker save redpandadata/redpanda:v24.2.7 | gzip > "./docker-exports/redpanda_v24.2.7.tar.gz"
docker save pgvector/pgvector:pg15 | gzip > "./docker-exports/pgvector_pg15.tar.gz"
docker save hashicorp/consul:1.17 | gzip > "./docker-exports/consul_1.17.tar.gz"
docker save hashicorp/vault:1.15 | gzip > "./docker-exports/vault_1.15.tar.gz"
docker save provectuslabs/kafka-ui:latest | gzip > "./docker-exports/kafka-ui_latest.tar.gz"

echo "✅ Images exported to ./docker-exports/"

# Create remote environment file
echo "📝 Creating remote environment configuration..."
cat > ./remote.env << 'EOF'
# Remote Environment Configuration for OmniNode Bridge
ENVIRONMENT=production
SERVICE_VERSION=0.1.0
LOG_LEVEL=info

# Database Configuration
POSTGRES_HOST=omninode-bridge-postgres
POSTGRES_PORT=5432
POSTGRES_DATABASE=omninode_bridge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=omninode_remote_2024_secure

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=omninode-bridge-redpanda:9092

# Kafka Advertised Listeners for Remote Access
KAFKA_ADVERTISED_HOST=192.168.86.200
KAFKA_ADVERTISED_PORT=29102

# Service Ports
HOOK_RECEIVER_PORT=8001
MODEL_METRICS_PORT=8005
ORCHESTRATOR_PORT=8060
REDUCER_PORT=8061
REGISTRY_PORT=8062
METADATA_STAMPING_PORT=8057
ONEXTREE_PORT=8058

# Infrastructure Ports
POSTGRES_PORT=5436
REDPANDA_PORT=29092
REDPANDA_EXTERNAL_PORT=29102
REDPANDA_ADMIN_PORT=29654
REDPANDA_PROXY_PORT=28092
CONSUL_PORT=28500
VAULT_PORT=8200

# AI Lab Configuration
AI_LAB_MAC_STUDIO=192.168.86.200
AI_LAB_MAC_MINI=192.168.86.101
AI_LAB_AI_PC=192.168.86.201
AI_LAB_MACBOOK_AIR=192.168.86.105
AI_LAB_OLLAMA_PORT=11434

# Security
API_KEY=omninode-bridge-remote-key-2024
SECURITY_CORS_ALLOWED_ORIGINS=http://192.168.86.200:3000,http://192.168.86.200:8000

# RedPanda Configuration
REDPANDA_LOG_LEVEL=info
REDPANDA_DEFAULT_PARTITIONS=3
REDPANDA_DEFAULT_REPLICATION_FACTOR=1
REDPANDA_LOG_RETENTION_MS=604800000
REDPANDA_SEGMENT_MS=3600000

# Consul Configuration
CONSUL_LOG_LEVEL=INFO
CONSUL_DNS_PORT=28600

# Service Configuration
ORCHESTRATOR_WORKERS=4
REDUCER_WORKERS=4
HOOK_RECEIVER_WORKERS=1
MODEL_METRICS_WORKERS=1
EOF

# Create remote docker-compose override
echo "📝 Creating remote docker-compose configuration..."
cat > ./docker-compose.remote.yml << 'EOF'
version: '3.8'

services:
  # Override port mappings for remote access
  postgres:
    ports:
      - "5436:5432"

  redpanda:
    ports:
      - "29092:9092"
      - "29102:29092"
      - "29654:9644"
      - "28092:8082"

  consul:
    ports:
      - "28500:8500"
      - "28600:8600/udp"

  vault:
    ports:
      - "8200:8200"

  hook-receiver:
    ports:
      - "8001:8001"

  model-metrics:
    ports:
      - "8005:8005"

  orchestrator:
    ports:
      - "8060:8060"
      - "9094:9091"

  reducer:
    ports:
      - "8061:8061"
      - "9092:9092"

  registry:
    ports:
      - "8062:8062"
      - "9095:9093"

  metadata-stamping:
    ports:
      - "8057:8053"
      - "9091:9090"

  onextree:
    ports:
      - "8058:8058"

  redpanda-ui:
    ports:
      - "8080:8080"
EOF

# Transfer everything to remote system
echo "📤 Transferring files to remote system..."
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH"
scp -r ./docker-exports "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"
scp -r ./deployment "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"
scp ./remote.env "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"
scp ./docker-compose.remote.yml "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"
scp ./docker-compose.remote-override.yml "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"

# Transfer the remote setup script
echo "📤 Transferring remote setup script..."
scp ./setup-remote.sh "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"

# Execute remote setup
echo "🚀 Executing remote setup..."
ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && chmod +x setup-remote.sh && ./setup-remote.sh"

# Wait a moment for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Verify deployment
echo "✅ Verifying deployment..."
echo "Checking service health..."

check_service() {
    local service=$1
    local port=$2
    local url="http://$REMOTE_HOST:$port/health"

    if curl -s -f "$url" > /dev/null 2>&1; then
        echo "  ✅ $service (port $port): Healthy"
        return 0
    else
        echo "  ❌ $service (port $port): Unhealthy"
        return 1
    fi
}

# Check all services
check_service "Hook Receiver" "8001"
check_service "Model Metrics" "8005"
check_service "Orchestrator" "8060"
check_service "Reducer" "8061"
check_service "Metadata Stamping" "8057"
check_service "OnexTree" "8058"

# Get container status
echo -e "\n📋 Container Status:"
ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml ps"

echo -e "\n🎉 Migration Complete!"
echo "=============================================="
echo "🌐 Services available at:"
echo "  - Hook Receiver: http://$REMOTE_HOST:8001"
echo "  - Model Metrics: http://$REMOTE_HOST:8005"
echo "  - Orchestrator: http://$REMOTE_HOST:8060"
echo "  - Reducer: http://$REMOTE_HOST:8061"
echo "  - Metadata Stamping: http://$REMOTE_HOST:8057"
echo "  - OnexTree: http://$REMOTE_HOST:8058"
echo "  - Consul UI: http://$REMOTE_HOST:28500"
echo "  - Vault UI: http://$REMOTE_HOST:8200"
echo "  - RedPanda UI: http://$REMOTE_HOST:8080"
echo ""
echo "🔧 Management commands:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST 'cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml ps'"
echo "  ssh $REMOTE_USER@$REMOTE_HOST 'cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml logs orchestrator'"
echo "  ssh $REMOTE_USER@$REMOTE_HOST 'cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml restart orchestrator'"

# Cleanup local exports
echo "🧹 Cleaning up local exports..."
rm -rf ./docker-exports
# Note: Keep remote.env and docker-compose files for reference

echo "✅ Migration complete and local cleanup done!"
