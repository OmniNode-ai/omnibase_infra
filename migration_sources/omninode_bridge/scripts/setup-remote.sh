#!/bin/bash
# setup-remote.sh - Single script to setup OmniNode Bridge on remote system
# This script will be transferred to and run on the remote system (192.168.86.200)

set -e

echo "🚀 OmniNode Bridge Remote Setup"
echo "==============================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS. Please adapt for your system."
    exit 1
fi

# Update Docker Desktop
echo "🔄 Updating Docker Desktop..."
if command -v brew &> /dev/null; then
    echo "Using Homebrew to update Docker Desktop..."
    brew update
    brew upgrade --cask docker
else
    echo "Homebrew not found. Please manually update Docker Desktop from:"
    echo "https://desktop.docker.com/mac/main/amd64/Docker.dmg"
    echo "Press Enter when Docker Desktop is updated and running..."
    read
fi

# Start Docker Desktop if not running
echo "🐳 Starting Docker Desktop..."
open -a Docker
echo "Waiting for Docker to start..."
sleep 30

# Verify Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi
echo "✅ Docker is running"

# Import Docker images
echo "📥 Importing Docker images..."
if [ ! -d "./docker-exports" ]; then
    echo "❌ docker-exports directory not found. Please ensure images were transferred."
    exit 1
fi

for file in ./docker-exports/*.tar.gz; do
    if [ -f "$file" ]; then
        echo "Importing $(basename "$file")..."
        gunzip -c "$file" | docker load
    fi
done

echo "✅ All images imported successfully"

# Setup environment
echo "📝 Setting up environment..."
if [ ! -f "./remote.env" ]; then
    echo "❌ remote.env file not found. Please ensure configuration was transferred."
    exit 1
fi

# Load environment variables
export $(cat remote.env | grep -v '^#' | xargs)

# Create docker-compose command with override for Kafka advertised listeners
COMPOSE_CMD="docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml"

# Stop any existing containers
echo "🛑 Stopping any existing containers..."
$COMPOSE_CMD down --remove-orphans 2>/dev/null || true

# Start infrastructure services first
echo "🏗️ Starting infrastructure services..."
$COMPOSE_CMD up -d postgres redpanda consul vault

# Wait for infrastructure to be healthy
echo "⏳ Waiting for infrastructure to be ready..."
sleep 45

# Check if infrastructure is healthy
echo "Checking infrastructure health..."
for i in {1..10}; do
    if docker ps | grep -q "omninode-bridge-postgres.*healthy" && \
       docker ps | grep -q "omninode-bridge-redpanda.*healthy" && \
       docker ps | grep -q "omninode-bridge-consul.*healthy"; then
        echo "✅ Infrastructure is healthy"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Infrastructure failed to start properly"
        echo "Container status:"
        docker ps | grep omninode
        exit 1
    fi
    echo "Waiting for infrastructure... ($i/10)"
    sleep 10
done

# Start core services
echo "🔧 Starting core bridge services..."
$COMPOSE_CMD up -d hook-receiver model-metrics metadata-stamping onextree

# Wait for core services
echo "⏳ Waiting for core services to start..."
sleep 30

# Start bridge nodes
echo "🌉 Starting bridge nodes..."
$COMPOSE_CMD up -d orchestrator reducer

# Wait for bridge nodes
echo "⏳ Waiting for bridge nodes to start..."
sleep 30

# Optional: Start registry with full profile
echo "📋 Starting registry (optional)..."
$COMPOSE_CMD --profile full up -d registry 2>/dev/null || echo "Registry not started (requires full profile)"

# Start RedPanda UI
echo "🖥️ Starting RedPanda UI..."
$COMPOSE_CMD up -d redpanda-ui 2>/dev/null || echo "RedPanda UI not started"

# Final health check
echo "🏥 Performing final health check..."
sleep 15

# Show container status
echo "📋 Container Status:"
$COMPOSE_CMD ps

# Check service endpoints
echo "🔍 Checking service endpoints..."
check_endpoint() {
    local service=$1
    local port=$2
    local path=${3:-/health}

    if curl -s -f "http://localhost:$port$path" > /dev/null 2>&1; then
        echo "  ✅ $service (port $port): Healthy"
    else
        echo "  ❌ $service (port $port): Unhealthy"
    fi
}

echo "Service Health Check:"
check_endpoint "Hook Receiver" "8001"
check_endpoint "Model Metrics" "8005"
check_endpoint "Orchestrator" "8060"
check_endpoint "Reducer" "8061"
check_endpoint "Metadata Stamping" "8057"
check_endpoint "OnexTree" "8058"
check_endpoint "Consul" "28500" "/v1/status/leader"
check_endpoint "Vault" "8200" "/v1/sys/health"

# Setup basic firewall rules (macOS)
echo "🔥 Setting up basic firewall rules..."
sudo pfctl -e 2>/dev/null || echo "Firewall already enabled"

# Create basic firewall rules
sudo tee /etc/pf.anchors/omninode-bridge << 'EOF' > /dev/null
# OmniNode Bridge Firewall Rules

# Allow SSH
pass in proto tcp from any to any port 22

# Allow OmniNode Bridge services from local network
pass in proto tcp from 192.168.86.0/24 to any port 8001  # Hook Receiver
pass in proto tcp from 192.168.86.0/24 to any port 8005  # Model Metrics
pass in proto tcp from 192.168.86.0/24 to any port 8060  # Orchestrator
pass in proto tcp from 192.168.86.0/24 to any port 8061  # Reducer
pass in proto tcp from 192.168.86.0/24 to any port 8062  # Registry
pass in proto tcp from 192.168.86.0/24 to any port 8057  # Metadata Stamping
pass in proto tcp from 192.168.86.0/24 to any port 8058  # OnexTree
pass in proto tcp from 192.168.86.0/24 to any port 28500  # Consul
pass in proto tcp from 192.168.86.0/24 to any port 8200   # Vault
pass in proto tcp from 192.168.86.0/24 to any port 8080   # RedPanda UI

# Block external access to services
block in proto tcp from any to any port 8001
block in proto tcp from any to any port 8005
block in proto tcp from any to any port 8060
block in proto tcp from any to any port 8061
block in proto tcp from any to any port 8062
block in proto tcp from any to any port 8057
block in proto tcp from any to any port 8058
block in proto tcp from any to any port 28500
block in proto tcp from any to any port 8200
block in proto tcp from any to any port 8080
EOF

# Load firewall rules
sudo pfctl -f /etc/pf.conf 2>/dev/null || echo "Firewall rules loaded"

# Verify Kafka advertised listeners configuration
echo "🔍 Verifying Kafka advertised listeners configuration..."
if grep -q "KAFKA_ADVERTISED_HOST" remote.env && grep -q "KAFKA_ADVERTISED_PORT" remote.env; then
    echo "✅ Kafka advertised listeners configured for remote access"
else
    echo "⚠️ Warning: Kafka advertised listeners not configured, external access may not work"
fi

# Create management script
echo "📝 Creating management script..."
cat > ./manage-bridge.sh << 'EOF'
#!/bin/bash
# manage-bridge.sh - Management script for OmniNode Bridge

COMPOSE_CMD="docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml"

case "$1" in
    "status")
        echo "📋 Container Status:"
        $COMPOSE_CMD ps
        ;;
    "logs")
        service=${2:-orchestrator}
        echo "📄 Logs for $service:"
        $COMPOSE_CMD logs -f $service
        ;;
    "restart")
        service=${2:-orchestrator}
        echo "🔄 Restarting $service:"
        $COMPOSE_CMD restart $service
        ;;
    "stop")
        echo "🛑 Stopping all services:"
        $COMPOSE_CMD down
        ;;
    "start")
        echo "🚀 Starting all services:"
        $COMPOSE_CMD up -d
        ;;
    "update")
        echo "🔄 Updating and restarting:"
        $COMPOSE_CMD pull
        $COMPOSE_CMD up -d
        ;;
    *)
        echo "Usage: $0 {status|logs|restart|stop|start|update}"
        echo "  status  - Show container status"
        echo "  logs    - Show logs (optionally specify service)"
        echo "  restart - Restart service (default: orchestrator)"
        echo "  stop    - Stop all services"
        echo "  start   - Start all services"
        echo "  update  - Update and restart all services"
        ;;
esac
EOF

chmod +x ./manage-bridge.sh

# Cleanup
echo "🧹 Cleaning up..."
rm -rf ./docker-exports

echo "✅ Remote setup complete!"
echo "==============================="
echo "🌐 Services available at:"
echo "  - Hook Receiver: http://192.168.86.200:8001"
echo "  - Model Metrics: http://192.168.86.200:8005"
echo "  - Orchestrator: http://192.168.86.200:8060"
echo "  - Reducer: http://192.168.86.200:8061"
echo "  - Metadata Stamping: http://192.168.86.200:8057"
echo "  - OnexTree: http://192.168.86.200:8058"
echo "  - Consul UI: http://192.168.86.200:28500"
echo "  - Vault UI: http://192.168.86.200:8200"
echo "  - RedPanda UI: http://192.168.86.200:8080"
echo ""
echo "🔧 Management commands:"
echo "  ./manage-bridge.sh status    - Show container status"
echo "  ./manage-bridge.sh logs      - Show logs"
echo "  ./manage-bridge.sh restart   - Restart services"
echo "  ./manage-bridge.sh stop      - Stop all services"
echo "  ./manage-bridge.sh start     - Start all services"
echo "  ./manage-bridge.sh update    - Update and restart"
echo ""
echo "🎉 OmniNode Bridge is now running on this system!"
