#!/bin/bash
# rebuild-service.sh - Rebuild and redeploy a specific OmniNode Bridge service
# Usage: ./rebuild-service.sh <service-name> [remote-host]

set -e

# Configuration
REMOTE_HOST="${2:-192.168.86.200}"
REMOTE_USER="jonah"
REMOTE_PATH="/Users/$REMOTE_USER/omninode_bridge"

# Get service name from first argument
SERVICE_NAME="$1"

if [ -z "$SERVICE_NAME" ]; then
    echo "❌ Usage: $0 <service-name> [remote-host]"
    echo ""
    echo "Available services:"
    echo "  - orchestrator"
    echo "  - reducer"
    echo "  - hook-receiver"
    echo "  - model-metrics"
    echo "  - metadata-stamping"
    echo "  - onextree"
    echo "  - registry"
    echo "  - deployment-receiver"
    echo ""
    echo "Examples:"
    echo "  $0 orchestrator"
    echo "  $0 reducer 192.168.86.200"
    exit 1
fi

echo "🔄 Rebuilding and redeploying $SERVICE_NAME to $REMOTE_HOST"
echo "=================================================="

# Check if service exists in docker-compose
if ! grep -q "^  $SERVICE_NAME:" deployment/docker-compose.yml; then
    echo "❌ Service '$SERVICE_NAME' not found in docker-compose.yml"
    echo "Available services:"
    grep "^  [a-zA-Z-]*:" deployment/docker-compose.yml | sed 's/^  /  - /' | sed 's/:$//'
    exit 1
fi

# Check if remote host is reachable
echo "📡 Checking connectivity to $REMOTE_HOST..."
if ! ping -c 1 -W 3000 $REMOTE_HOST > /dev/null 2>&1; then
    echo "❌ Cannot reach $REMOTE_HOST. Please check network connectivity."
    exit 1
fi

# Build the specific service locally
echo "🔨 Building $SERVICE_NAME locally..."
if [ "$SERVICE_NAME" = "orchestrator" ] || [ "$SERVICE_NAME" = "reducer" ] || [ "$SERVICE_NAME" = "registry" ]; then
    # These services use the bridge-nodes dockerfile
    docker build -f docker/bridge-nodes/Dockerfile.$SERVICE_NAME -t omninode_bridge-$SERVICE_NAME:latest .
elif [ "$SERVICE_NAME" = "hook-receiver" ]; then
    docker build -f docker/hook-receiver/Dockerfile -t omninode_bridge-hook-receiver:latest .
elif [ "$SERVICE_NAME" = "model-metrics" ]; then
    docker build -f docker/model-metrics/Dockerfile -t omninode_bridge-model-metrics:latest .
elif [ "$SERVICE_NAME" = "metadata-stamping" ]; then
    docker build -f docker/metadata-stamping/Dockerfile -t omninode_bridge-metadata-stamping:latest .
elif [ "$SERVICE_NAME" = "onextree" ]; then
    docker build -f docker/onextree/Dockerfile -t omninode_bridge-onextree:latest .
elif [ "$SERVICE_NAME" = "deployment-receiver" ]; then
    docker build -f docker/deployment-receiver/Dockerfile -t omninode_bridge-deployment-receiver:latest .
else
    echo "❌ Unknown service: $SERVICE_NAME"
    exit 1
fi

echo "✅ Service built successfully"

# Export the specific image
echo "📦 Exporting $SERVICE_NAME image..."
mkdir -p ./docker-exports
docker save "omninode_bridge-$SERVICE_NAME:latest" | gzip > "./docker-exports/omninode_bridge-$SERVICE_NAME.tar.gz"

# Transfer image to remote system
echo "📤 Transferring image to remote system..."
scp "./docker-exports/omninode_bridge-$SERVICE_NAME.tar.gz" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/docker-exports/"

# Import image on remote system
echo "📥 Importing image on remote system..."
ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && gunzip -c docker-exports/omninode_bridge-$SERVICE_NAME.tar.gz | docker load"

# Stop the specific service on remote system
echo "🛑 Stopping $SERVICE_NAME on remote system..."
ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml stop $SERVICE_NAME"

# Remove the old container
echo "🗑️ Removing old $SERVICE_NAME container..."
ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml rm -f $SERVICE_NAME"

# Start the service with new image
echo "🚀 Starting $SERVICE_NAME with new image..."
ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml up -d $SERVICE_NAME"

# Wait for service to start
echo "⏳ Waiting for $SERVICE_NAME to start..."
sleep 15

# Check service health
echo "🏥 Checking $SERVICE_NAME health..."
get_service_port() {
    case "$1" in
        "orchestrator") echo "8060" ;;
        "reducer") echo "8061" ;;
        "hook-receiver") echo "8001" ;;
        "model-metrics") echo "8005" ;;
        "metadata-stamping") echo "8057" ;;
        "onextree") echo "8058" ;;
        "registry") echo "8062" ;;
        "deployment-receiver") echo "8001" ;;
        *) echo "unknown" ;;
    esac
}

SERVICE_PORT=$(get_service_port "$SERVICE_NAME")

if [ "$SERVICE_PORT" != "unknown" ]; then
    if curl -s -f "http://$REMOTE_HOST:$SERVICE_PORT/health" > /dev/null 2>&1; then
        echo "✅ $SERVICE_NAME is healthy on port $SERVICE_PORT"
    else
        echo "❌ $SERVICE_NAME health check failed on port $SERVICE_PORT"
        echo "Checking container status..."
        ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml ps $SERVICE_NAME"
        echo "Checking logs..."
        ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml logs --tail=20 $SERVICE_NAME"
    fi
else
    echo "⚠️ Cannot determine port for $SERVICE_NAME, checking container status..."
    ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml ps $SERVICE_NAME"
fi

# Show final status
echo "📋 Final container status:"
ssh "$REMOTE_USER@$REMOTE_HOST" "cd $REMOTE_PATH && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml ps $SERVICE_NAME"

# Cleanup local exports
echo "🧹 Cleaning up local exports..."
rm -f "./docker-exports/omninode_bridge-$SERVICE_NAME.tar.gz"

echo "✅ Rebuild and redeploy of $SERVICE_NAME complete!"
echo "🌐 Service available at: http://$REMOTE_HOST:$SERVICE_PORT"
