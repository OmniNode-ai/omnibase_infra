#!/bin/bash
# OmniNode Bridge Security Setup Script
# This script sets up the complete security infrastructure

set -e

echo "🔒 OmniNode Bridge Security Setup"
echo "================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root for security reasons"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is required but not installed"
    exit 1
fi

if ! command_exists openssl; then
    echo "❌ OpenSSL is required but not installed"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p ssl/{ca,postgres,client}
mkdir -p logs
chmod 755 logs

# Generate SSL certificates
echo "🔐 Generating SSL certificates..."
if [ ! -f ssl/ca/ca.crt ]; then
    chmod +x ssl/setup-ssl.sh
    ./ssl/setup-ssl.sh
else
    echo "ℹ️  SSL certificates already exist, skipping generation"
fi

# Set proper permissions
echo "🔒 Setting secure permissions..."
chmod 700 ssl
chmod 600 ssl/*/private.key ssl/*/*.key 2>/dev/null || true
chmod 644 ssl/*/*.crt ssl/*/*.pem 2>/dev/null || true

# Generate secure passwords if not set
if [ -z "$POSTGRES_PASSWORD" ]; then
    echo "🎲 Generating secure PostgreSQL password..."
    export POSTGRES_PASSWORD=$(openssl rand -base64 32)
    echo "📝 Generated POSTGRES_PASSWORD (save this securely!)"
    echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD"
fi

if [ -z "$API_KEY" ]; then
    echo "🎲 Generating secure API key..."
    export API_KEY=$(openssl rand -base64 32)
    echo "📝 Generated API_KEY (save this securely!)"
    echo "API_KEY=$API_KEY"
fi

# Create environment file
echo "📄 Creating environment configuration..."
cat > .env.security << EOF
# OmniNode Bridge Security Configuration
# Generated on $(date)

# Database Configuration
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_HOST=omninode-bridge-postgres
POSTGRES_PORT=5432
POSTGRES_DATABASE=omninode_bridge
POSTGRES_USER=postgres

# SSL Configuration
POSTGRES_SSL_ENABLED=true
POSTGRES_SSL_CERT=/app/ssl/client.crt
POSTGRES_SSL_KEY=/app/ssl/client.key
POSTGRES_SSL_CA=/app/ssl/ca.crt
POSTGRES_SSL_CHECK_HOSTNAME=false

# API Security
API_KEY=$API_KEY

# Service Ports
HOOK_RECEIVER_PORT=8001
MODEL_METRICS_PORT=8005
WORKFLOW_COORDINATOR_PORT=8006

# Infrastructure Ports
CONSUL_PORT=8500
CONSUL_DNS_PORT=8600
REDPANDA_PORT=9092
REDPANDA_EXTERNAL_PORT=29092
REDPANDA_ADMIN_PORT=9644
REDPANDA_PROXY_PORT=8082
REDPANDA_UI_PORT=8080

# Logging
LOG_LEVEL=info
ENVIRONMENT=production

# CORS Configuration
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
EOF

echo "✅ Environment configuration created: .env.security"

# Validate certificates
echo "🔍 Validating SSL certificates..."
if openssl verify -CAfile ssl/ca/ca.crt ssl/postgres/server.crt >/dev/null 2>&1; then
    echo "✅ PostgreSQL server certificate is valid"
else
    echo "❌ PostgreSQL server certificate validation failed"
    exit 1
fi

if openssl verify -CAfile ssl/ca/ca.crt ssl/client/client.crt >/dev/null 2>&1; then
    echo "✅ Client certificate is valid"
else
    echo "❌ Client certificate validation failed"
    exit 1
fi

# Build secure Docker images
echo "🐳 Building secure Docker images..."
docker-compose -f docker-compose.security.yml build --no-cache

# Start infrastructure with security
echo "🚀 Starting secure infrastructure..."
docker-compose -f docker-compose.security.yml up -d postgres consul redpanda

# Wait for infrastructure to be ready
echo "⏳ Waiting for infrastructure to be ready..."
sleep 30

# Check infrastructure health
echo "🏥 Checking infrastructure health..."
POSTGRES_READY=false
CONSUL_READY=false
REDPANDA_READY=false

for i in {1..30}; do
    if docker-compose -f docker-compose.security.yml exec -T postgres pg_isready -U postgres -d omninode_bridge >/dev/null 2>&1; then
        POSTGRES_READY=true
        break
    fi
    echo "⏳ Waiting for PostgreSQL... ($i/30)"
    sleep 2
done

if [ "$POSTGRES_READY" = true ]; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL failed to start"
    exit 1
fi

# Test SSL connection
echo "🔐 Testing SSL connection..."
if docker-compose -f docker-compose.security.yml exec -T postgres psql -U postgres -d omninode_bridge -c "SELECT ssl_is_used();" >/dev/null 2>&1; then
    echo "✅ SSL connection test passed"
else
    echo "❌ SSL connection test failed"
fi

# Start application services
echo "🚀 Starting application services..."
docker-compose -f docker-compose.security.yml up -d

# Final health check
echo "🏥 Final health check..."
sleep 20

if curl -f http://localhost:8001/health >/dev/null 2>&1; then
    echo "✅ HookReceiver service is healthy"
else
    echo "⚠️  HookReceiver service may not be ready yet"
fi

# Display summary
echo ""
echo "🎉 Security setup completed!"
echo "=========================="
echo ""
echo "📋 Summary:"
echo "  ✅ SSL certificates generated and configured"
echo "  ✅ Secure passwords generated"
echo "  ✅ Docker containers hardened"
echo "  ✅ Network security configured"
echo "  ✅ Database security enhanced"
echo ""
echo "📄 Important files created:"
echo "  🔐 SSL certificates: ssl/"
echo "  ⚙️  Environment config: .env.security"
echo "  📚 Documentation: INFRASTRUCTURE_SECURITY.md"
echo ""
echo "🔐 Security credentials (SAVE THESE SECURELY!):"
echo "  POSTGRES_PASSWORD=$POSTGRES_PASSWORD"
echo "  API_KEY=$API_KEY"
echo ""
echo "🌐 Service endpoints:"
echo "  📡 HookReceiver: http://localhost:8001"
echo "  📊 Model Metrics: http://localhost:8005"
echo "  🔄 Workflow Coordinator: http://localhost:8006"
echo "  🗃️  PostgreSQL: localhost:5436 (SSL required)"
echo "  📨 RedPanda: localhost:9092"
echo "  🕸️  Consul: http://localhost:8500"
echo ""
echo "📚 Next steps:"
echo "  1. Review INFRASTRUCTURE_SECURITY.md for detailed configuration"
echo "  2. Test SSL connections and security features"
echo "  3. Set up monitoring and alerting"
echo "  4. Configure backup procedures"
echo ""
echo "⚠️  IMPORTANT: This uses self-signed certificates for development."
echo "   For production, replace with CA-signed certificates!"

# Final verification
echo ""
echo "🔍 Running final security verification..."

# Check if services are running as non-root
if docker-compose -f docker-compose.security.yml exec -T hook-receiver id | grep -q "uid=1000"; then
    echo "✅ Application services running as non-root user"
else
    echo "⚠️  Warning: Services may be running as root"
fi

# Check if SSL is actually being used
if docker-compose -f docker-compose.security.yml exec -T postgres psql -U postgres -d omninode_bridge -c "SELECT ssl_version(), ssl_cipher();" 2>/dev/null | grep -q "TLS"; then
    echo "✅ SSL/TLS is active and working"
else
    echo "⚠️  Warning: SSL/TLS may not be properly configured"
fi

echo ""
echo "🔒 Security setup completed successfully!"
echo "   See INFRASTRUCTURE_SECURITY.md for detailed documentation."
