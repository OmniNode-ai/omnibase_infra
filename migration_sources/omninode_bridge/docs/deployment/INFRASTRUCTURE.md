# Infrastructure Setup Guide

## Overview

This guide provides step-by-step instructions for deploying the core infrastructure components of the OmniNode Bridge: RedPanda event streaming, Consul service discovery, PostgreSQL storage, and Redis caching. This infrastructure forms the foundation for intelligent service coordination.

## Prerequisites

### System Requirements
```yaml
Minimum Requirements:
  RAM: 16GB
  CPU: 8 cores
  Storage: 100GB SSD
  Network: 1Gbps

Recommended Requirements:
  RAM: 32GB
  CPU: 16 cores
  Storage: 500GB NVMe SSD
  Network: 10Gbps

Production Requirements:
  RAM: 64GB per node
  CPU: 32 cores per node
  Storage: 1TB NVMe SSD per node
  Network: 10Gbps with redundancy
```

### Software Prerequisites
```bash
# Required software
docker --version          # >= 24.0.0
docker-compose --version  # >= 2.20.0
curl --version           # For health checks
jq --version             # For JSON processing

# Optional but recommended
kubectl --version        # For Kubernetes deployment
helm --version          # For Helm charts
```

## Development Environment Setup

### 1. Clone and Setup Project
```bash
# Clone the repository
git clone https://github.com/OmniNode-ai/omninode_bridge.git
cd omninode_bridge

# Create environment files
cp .env.example .env.development
cp .env.example .env.production

# Create necessary directories
mkdir -p data/{redpanda,consul,postgres,redis}
mkdir -p logs/{redpanda,consul,postgres,redis,services}
mkdir -p config/{redpanda,consul,postgres,redis}
```

### 2. Environment Configuration
```bash
# .env.development
export ENVIRONMENT="development"
export LOG_LEVEL="DEBUG"

# Infrastructure
export REDPANDA_BROKERS="localhost:9092"
export CONSUL_URL="http://localhost:8500"
export POSTGRES_URL="postgresql://omninode:dev_password@localhost:5432/omninode_bridge_dev"
export REDIS_URL="redis://localhost:6379/0"

# Service Ports
export HOOKS_RECEIVER_PORT="8080"
export TOOL_PROXY_PORT="8090"
export MCP_REGISTRY_PORT="8100"

# Performance (reduced for development)
export MAX_CONNECTIONS="100"
export CONNECTION_POOL_SIZE="5"
export CACHE_TTL_DEFAULT="300"
```

### 3. Docker Compose Development Deployment
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # RedPanda Event Streaming
  redpanda:
    image: redpandadata/redpanda:latest
    container_name: omninode-redpanda-dev
    command:
      - redpanda
      - start
      - --smp
      - '2'
      - --reserve-memory
      - '1G'
      - --overprovisioned
      - --node-id
      - '0'
      - --kafka-addr
      - 'PLAINTEXT://0.0.0.0:29092,OUTSIDE://0.0.0.0:9092'
      - --advertise-kafka-addr
      - 'PLAINTEXT://redpanda:29092,OUTSIDE://localhost:9092'
      - --pandaproxy-addr
      - 'PLAINTEXT://0.0.0.0:8082,OUTSIDE://0.0.0.0:8082'
      - --advertise-pandaproxy-addr
      - 'PLAINTEXT://redpanda:8082,OUTSIDE://localhost:8082'
    ports:
      - "9092:9092"
      - "8082:8082"
      - "9644:9644"
    volumes:
      - redpanda_data:/var/lib/redpanda/data
      - ./config/redpanda/redpanda.yaml:/etc/redpanda/redpanda.yaml
    healthcheck:
      test: ["CMD-SHELL", "rpk cluster health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  # Consul Service Discovery
  consul:
    image: consul:1.16
    container_name: omninode-consul-dev
    ports:
      - "8500:8500"
      - "8600:8600/udp"
    volumes:
      - consul_data:/consul/data
      - ./config/consul/consul.hcl:/consul/config/consul.hcl
    environment:
      - CONSUL_BIND_INTERFACE=eth0
    command: consul agent -config-file=/consul/config/consul.hcl
    healthcheck:
      test: ["CMD-SHELL", "consul members"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  # PostgreSQL Storage
  postgres:
    image: postgres:15
    container_name: omninode-postgres-dev
    environment:
      POSTGRES_DB: omninode_bridge_dev
      POSTGRES_USER: omninode
      POSTGRES_PASSWORD: dev_password
      PGDATA: /var/lib/postgresql/data/pgdata
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgres/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./config/postgres/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U omninode -d omninode_bridge_dev"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: omninode-redis-dev
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./config/redis/redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD-SHELL", "redis-cli ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  # Jaeger Tracing (Optional)
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: omninode-jaeger-dev
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true

volumes:
  redpanda_data:
    driver: local
  consul_data:
    driver: local
  postgres_data:
    driver: local
  redis_data:
    driver: local

networks:
  default:
    name: omninode_dev_network
    driver: bridge
```

### 4. Start Development Infrastructure
```bash
# Start all infrastructure services
docker-compose -f docker-compose.dev.yml up -d

# Verify services are healthy
docker-compose -f docker-compose.dev.yml ps

# Check logs if needed
docker-compose -f docker-compose.dev.yml logs -f redpanda
docker-compose -f docker-compose.dev.yml logs -f consul
docker-compose -f docker-compose.dev.yml logs -f postgres
docker-compose -f docker-compose.dev.yml logs -f redis

# Test connectivity
curl http://localhost:8500/v1/status/leader  # Consul
curl http://localhost:9644/v1/status/ready   # RedPanda
redis-cli -h localhost ping                   # Redis
psql -h localhost -U omninode -d omninode_bridge_dev -c "SELECT 1"  # PostgreSQL
```

## Configuration Files

### RedPanda Configuration
```yaml
# config/redpanda/redpanda.yaml
redpanda:
  data_directory: "/var/lib/redpanda/data"
  node_id: 0
  rpc_server:
    address: "0.0.0.0"
    port: 33145
  kafka_api:
    - address: "0.0.0.0"
      port: 9092
  admin:
    - address: "0.0.0.0"
      port: 9644

  # Performance tuning for development
  log_segment_size: 268435456      # 256MB
  log_retention_ms: 86400000       # 1 day
  log_retention_bytes: 10737418240 # 10GB

  # Replication (single node for dev)
  default_topic_replications: 1
  transaction_coordinator_replication: 1

  # Compression
  compression_type: "gzip"

  # Batching
  batch_max_bytes: 1048576
  linger_ms: 50

# Topic auto-creation
auto_create_topics_enabled: true

# Schema registry (optional)
schema_registry:
  schema_registry_api:
    - address: "0.0.0.0"
      port: 8081
```

### Consul Configuration
```hcl
# config/consul/consul.hcl
datacenter = "dc1"
data_dir = "/consul/data"
log_level = "INFO"
node_name = "consul-dev-node"
bind_addr = "0.0.0.0"
client_addr = "0.0.0.0"

# Development server configuration
server = true
bootstrap_expect = 1
ui_config {
  enabled = true
}

# Ports
ports {
  grpc = 8502
  http = 8500
  dns = 8600
}

# Connect service mesh (disabled for development)
connect {
  enabled = false
}

# Performance settings for development
performance {
  raft_multiplier = 1
}

# Logging
log_rotate_duration = "24h"
log_rotate_max_files = 3

# Development services registration
services = [
  {
    name = "consul"
    tags = ["consul", "service-discovery"]
    port = 8500
    check = {
      http = "http://localhost:8500/v1/status/leader"
      interval = "10s"
    }
  }
]
```

### PostgreSQL Configuration
```sql
-- config/postgres/init-db.sql
-- Initialize OmniNode Bridge database schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS intelligence;
CREATE SCHEMA IF NOT EXISTS metrics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Intelligence patterns table
CREATE TABLE intelligence.patterns (
    pattern_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    frequency INTEGER DEFAULT 1,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    success_rate DECIMAL(3,2) DEFAULT 0.0 CHECK (success_rate >= 0 AND success_rate <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Service execution intelligence
CREATE TABLE intelligence.executions (
    execution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID NOT NULL,
    service_name VARCHAR(255) NOT NULL,
    operation VARCHAR(255) NOT NULL,
    execution_time_ms INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    intelligence_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Communication intelligence
CREATE TABLE intelligence.communications (
    communication_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID NOT NULL,
    source_service VARCHAR(255),
    target_service VARCHAR(255),
    request_data JSONB NOT NULL,
    response_data JSONB NOT NULL,
    performance_metrics JSONB NOT NULL,
    intelligence_insights JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Service ecosystem snapshots
CREATE TABLE intelligence.ecosystem_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    services_active JSONB NOT NULL,
    capability_matrix JSONB NOT NULL,
    dependency_graph JSONB NOT NULL,
    performance_metrics JSONB NOT NULL,
    snapshot_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optimization recommendations
CREATE TABLE intelligence.optimization_recommendations (
    recommendation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recommendation_type VARCHAR(100) NOT NULL,
    target_service VARCHAR(255),
    recommendation_data JSONB NOT NULL,
    expected_impact JSONB NOT NULL,
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMP WITH TIME ZONE,
    effectiveness_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Metrics tables
CREATE TABLE metrics.service_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE metrics.performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation_id VARCHAR(255) NOT NULL,
    response_time_ms INTEGER NOT NULL,
    throughput_rps DECIMAL(10,2),
    error_rate DECIMAL(5,4),
    resource_usage JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit tables
CREATE TABLE audit.events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    service_name VARCHAR(255),
    user_id VARCHAR(255),
    event_data JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_patterns_type ON intelligence.patterns(pattern_type);
CREATE INDEX idx_patterns_confidence ON intelligence.patterns(confidence DESC);
CREATE INDEX idx_patterns_frequency ON intelligence.patterns(frequency DESC);
CREATE INDEX idx_executions_service ON intelligence.executions(service_name, operation);
CREATE INDEX idx_executions_correlation ON intelligence.executions(correlation_id);
CREATE INDEX idx_executions_time ON intelligence.executions(created_at DESC);
CREATE INDEX idx_communications_correlation ON intelligence.communications(correlation_id);
CREATE INDEX idx_communications_services ON intelligence.communications(source_service, target_service);
CREATE INDEX idx_recommendations_service ON intelligence.optimization_recommendations(target_service);
CREATE INDEX idx_recommendations_applied ON intelligence.optimization_recommendations(applied, created_at DESC);
CREATE INDEX idx_service_metrics_name_type ON metrics.service_metrics(service_name, metric_type);
CREATE INDEX idx_service_metrics_timestamp ON metrics.service_metrics(timestamp DESC);
CREATE INDEX idx_performance_metrics_operation ON metrics.performance_metrics(operation_id);
CREATE INDEX idx_performance_metrics_timestamp ON metrics.performance_metrics(timestamp DESC);
CREATE INDEX idx_audit_events_type ON audit.events(event_type);
CREATE INDEX idx_audit_events_service ON audit.events(service_name);
CREATE INDEX idx_audit_events_timestamp ON audit.events(timestamp DESC);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for timestamp updates
CREATE TRIGGER update_patterns_updated_at BEFORE UPDATE ON intelligence.patterns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE VIEW intelligence.pattern_summary AS
SELECT
    pattern_type,
    COUNT(*) as pattern_count,
    AVG(confidence) as avg_confidence,
    AVG(success_rate) as avg_success_rate,
    MAX(last_seen) as last_activity
FROM intelligence.patterns
GROUP BY pattern_type;

CREATE VIEW intelligence.service_performance AS
SELECT
    service_name,
    operation,
    COUNT(*) as execution_count,
    AVG(execution_time_ms) as avg_execution_time,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as success_rate,
    MAX(created_at) as last_execution
FROM intelligence.executions
GROUP BY service_name, operation;

-- Grant permissions
GRANT USAGE ON SCHEMA intelligence TO omninode;
GRANT USAGE ON SCHEMA metrics TO omninode;
GRANT USAGE ON SCHEMA audit TO omninode;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA intelligence TO omninode;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO omninode;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO omninode;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA intelligence TO omninode;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO omninode;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO omninode;
```

```conf
# config/postgres/postgresql.conf
# PostgreSQL configuration for OmniNode Bridge

# Connection settings
listen_addresses = '*'
port = 5432
max_connections = 200
superuser_reserved_connections = 3

# Memory settings
shared_buffers = 256MB           # 25% of RAM for development
effective_cache_size = 1GB       # 75% of available RAM
work_mem = 4MB                   # For complex queries
maintenance_work_mem = 64MB      # For maintenance operations

# Write ahead log settings
wal_level = replica
wal_buffers = 16MB
checkpoint_completion_target = 0.9
max_wal_size = 2GB
min_wal_size = 80MB

# Query planner settings
default_statistics_target = 100
random_page_cost = 1.1          # For SSD storage

# Logging settings
log_destination = 'stderr'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 1000    # Log slow queries (>1s)
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_statement = 'all'               # For development only

# Performance monitoring
shared_preload_libraries = 'pg_stat_statements'
track_activities = on
track_counts = on
track_io_timing = on
track_functions = pl

# Locale settings
lc_messages = 'en_US.utf8'
lc_monetary = 'en_US.utf8'
lc_numeric = 'en_US.utf8'
lc_time = 'en_US.utf8'
default_text_search_config = 'pg_catalog.english'

# Time zone
timezone = 'UTC'
```

### Redis Configuration
```conf
# config/redis/redis.conf
# Redis configuration for OmniNode Bridge

# Network
bind 0.0.0.0
port 6379
timeout 300
keepalive 300

# General
daemonize no
supervised no
pidfile /var/run/redis_6379.pid
loglevel notice
logfile ""

# Snapshotting
save 900 1      # Save if at least 1 key changed in 900 seconds
save 300 10     # Save if at least 10 keys changed in 300 seconds
save 60 10000   # Save if at least 10000 keys changed in 60 seconds
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data

# Replication
replica-serve-stale-data yes
replica-read-only yes

# Security (development settings)
# requirepass your_redis_password_here

# Memory management
maxmemory 512mb
maxmemory-policy allkeys-lru

# Append only file
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Lua scripting
lua-time-limit 5000

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Event notification
notify-keyspace-events Ex

# Advanced config
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
hll-sparse-max-bytes 3000
stream-node-max-bytes 4096
stream-node-max-entries 100

# Client output buffer limits
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60

# TCP settings
tcp-keepalive 300
tcp-backlog 511
```

## Initial Topic and Service Setup

### 1. Create RedPanda Topics
```bash
# Create required topics for intelligence system
docker exec omninode-redpanda-dev rpk topic create hooks.service_lifecycle \
  --partitions 6 --replicas 1

docker exec omninode-redpanda-dev rpk topic create hooks.execution_intelligence \
  --partitions 12 --replicas 1

docker exec omninode-redpanda-dev rpk topic create proxy.communication_intelligence \
  --partitions 8 --replicas 1

docker exec omninode-redpanda-dev rpk topic create tools.registration_events \
  --partitions 4 --replicas 1

docker exec omninode-redpanda-dev rpk topic create hooks.user_interaction \
  --partitions 4 --replicas 1

docker exec omninode-redpanda-dev rpk topic create hooks.git_intelligence \
  --partitions 4 --replicas 1

# Verify topics creation
docker exec omninode-redpanda-dev rpk topic list
```

### 2. Register Initial Services in Consul
```bash
# Register infrastructure services
curl -X PUT http://localhost:8500/v1/agent/service/register \
  -H "Content-Type: application/json" \
  -d '{
    "ID": "redpanda-dev",
    "Name": "redpanda",
    "Tags": ["event-streaming", "kafka", "infrastructure"],
    "Address": "redpanda",
    "Port": 9092,
    "Meta": {
      "version": "latest",
      "environment": "development",
      "capabilities": "[\"event_streaming\", \"schema_registry\"]"
    },
    "Check": {
      "HTTP": "http://redpanda:9644/v1/status/ready",
      "Interval": "30s",
      "Timeout": "10s"
    }
  }'

curl -X PUT http://localhost:8500/v1/agent/service/register \
  -H "Content-Type: application/json" \
  -d '{
    "ID": "postgres-dev",
    "Name": "postgres",
    "Tags": ["database", "storage", "infrastructure"],
    "Address": "postgres",
    "Port": 5432,
    "Meta": {
      "version": "15",
      "environment": "development",
      "capabilities": "[\"relational_storage\", \"jsonb_support\"]"
    },
    "Check": {
      "TCP": "postgres:5432",
      "Interval": "30s",
      "Timeout": "10s"
    }
  }'

curl -X PUT http://localhost:8500/v1/agent/service/register \
  -H "Content-Type: application/json" \
  -d '{
    "ID": "redis-dev",
    "Name": "redis",
    "Tags": ["cache", "storage", "infrastructure"],
    "Address": "redis",
    "Port": 6379,
    "Meta": {
      "version": "7",
      "environment": "development",
      "capabilities": "[\"caching\", \"session_storage\", \"pub_sub\"]"
    },
    "Check": {
      "TCP": "redis:6379",
      "Interval": "30s",
      "Timeout": "10s"
    }
  }'

# Verify service registration
curl http://localhost:8500/v1/catalog/services | jq
```

### 3. Test Infrastructure Connectivity
```bash
# Test RedPanda
echo "Testing RedPanda..."
docker exec omninode-redpanda-dev rpk topic produce test-topic --num 1
docker exec omninode-redpanda-dev rpk topic consume test-topic --num 1

# Test PostgreSQL
echo "Testing PostgreSQL..."
docker exec omninode-postgres-dev psql -U omninode -d omninode_bridge_dev -c "
  INSERT INTO intelligence.patterns (pattern_type, pattern_data, confidence)
  VALUES ('test_pattern', '{\"test\": true}', 0.9);
  SELECT * FROM intelligence.patterns WHERE pattern_type = 'test_pattern';"

# Test Redis
echo "Testing Redis..."
docker exec omninode-redis-dev redis-cli set test_key "test_value"
docker exec omninode-redis-dev redis-cli get test_key

# Test Consul
echo "Testing Consul..."
curl -s http://localhost:8500/v1/catalog/services | jq 'keys'

echo "All infrastructure tests completed!"
```

## Health Monitoring Setup

### 1. Health Check Script
```bash
#!/bin/bash
# scripts/health-check.sh

set -e

echo "=== OmniNode Bridge Infrastructure Health Check ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local service_name=$1
    local health_check=$2

    echo -n "Checking $service_name... "

    if eval "$health_check" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Unhealthy${NC}"
        return 1
    fi
}

# Check Docker containers
echo "=== Docker Container Status ==="
docker-compose -f docker-compose.dev.yml ps

echo ""
echo "=== Service Health Checks ==="

# Check RedPanda
check_service "RedPanda" "curl -s http://localhost:9644/v1/status/ready"

# Check Consul
check_service "Consul" "curl -s http://localhost:8500/v1/status/leader"

# Check PostgreSQL
check_service "PostgreSQL" "docker exec omninode-postgres-dev pg_isready -U omninode"

# Check Redis
check_service "Redis" "docker exec omninode-redis-dev redis-cli ping"

echo ""
echo "=== Topic Verification ==="
topics=$(docker exec omninode-redpanda-dev rpk topic list 2>/dev/null | grep -c "hooks\|tools\|proxy" || echo "0")
echo "Intelligence topics created: $topics"

echo ""
echo "=== Service Registry ==="
services=$(curl -s http://localhost:8500/v1/catalog/services | jq 'keys | length')
echo "Registered services: $services"

echo ""
echo "=== Database Schema ==="
tables=$(docker exec omninode-postgres-dev psql -U omninode -d omninode_bridge_dev -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema IN ('intelligence', 'metrics', 'audit');" 2>/dev/null | xargs)
echo "Intelligence tables created: $tables"

echo ""
echo "Health check completed!"
```

### 2. Monitoring Dashboard URLs
```bash
# Access URLs for development environment
echo "=== Development Environment URLs ==="
echo "Consul UI:      http://localhost:8500"
echo "RedPanda Admin: http://localhost:9644"
echo "Jaeger UI:      http://localhost:16686"
echo ""
echo "=== API Endpoints ==="
echo "Consul API:     http://localhost:8500/v1/"
echo "RedPanda API:   http://localhost:8082/"
echo ""
echo "=== Database Connections ==="
echo "PostgreSQL:     postgresql://omninode:dev_password@localhost:5432/omninode_bridge_dev"
echo "Redis:          redis://localhost:6379/0"
```

## Production Environment Considerations

### 1. High Availability Setup
```yaml
# Production considerations
High Availability:
  RedPanda:
    - 3+ node cluster
    - Replication factor: 3
    - Cross-AZ deployment

  Consul:
    - 3 or 5 node cluster
    - Cross-AZ deployment
    - TLS encryption

  PostgreSQL:
    - Primary/replica setup
    - Synchronous replication
    - Automated failover

  Redis:
    - Redis Cluster or Sentinel
    - 3+ node setup
    - Cross-AZ deployment

Security:
  - TLS/SSL for all communications
  - Network segmentation
  - Service mesh (Consul Connect)
  - Secret management (Vault)
  - Regular security updates

Monitoring:
  - Prometheus metrics
  - Grafana dashboards
  - AlertManager notifications
  - Centralized logging (ELK stack)
  - Distributed tracing (Jaeger)

Backup:
  - Automated database backups
  - Cross-region backup storage
  - Disaster recovery procedures
  - Regular backup testing
```

### 2. Performance Tuning
```yaml
Performance Optimizations:
  RedPanda:
    - Dedicated NVMe storage
    - Memory-mapped files
    - Optimized batch sizes
    - Compression settings

  PostgreSQL:
    - Connection pooling (PgBouncer)
    - Read replicas
    - Partitioning for large tables
    - Query optimization

  Redis:
    - Memory optimization
    - Persistence settings
    - Connection pooling
    - Cluster sharding

  Network:
    - Dedicated network segments
    - Load balancers
    - CDN for static content
    - Network optimization
```

This infrastructure setup provides the robust foundation needed for the OmniNode Bridge's intelligent service coordination capabilities.
