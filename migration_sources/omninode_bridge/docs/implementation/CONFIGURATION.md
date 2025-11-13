# Configuration Guide

## Overview

This guide provides comprehensive configuration for all OmniNode Bridge components, including infrastructure services, intelligence systems, and service coordination. Configuration follows a layered approach supporting development, testing, and production environments.

## Environment Configuration

### Base Environment Variables
```bash
# Core Infrastructure
export REDPANDA_BROKERS="localhost:9092"
export CONSUL_URL="http://localhost:8500"
export POSTGRES_URL="postgresql://omninode:password@localhost:5432/omninode_bridge"
export REDIS_URL="redis://localhost:6379"

# Intelligence Services
export HOOKS_RECEIVER_PORT="8080"
export TOOL_PROXY_PORT="8090"
export MCP_REGISTRY_PORT="8100"

# Service Discovery
export SERVICE_NAME="omninode-bridge"
export SERVICE_VERSION="0.1.0"
export CONSUL_DATACENTER="dc1"
export CONSUL_ENCRYPTION_KEY=""

# Intelligence Configuration
export INTELLIGENCE_PROCESSOR_WORKERS="4"
export PATTERN_ANALYSIS_ENABLED="true"
export OPTIMIZATION_ENGINE_ENABLED="true"
export LEARNING_CYCLE_INTERVAL="300"  # 5 minutes

# Security
export JWT_SECRET_KEY="your-jwt-secret-key-here"
export API_KEY_SALT="your-api-key-salt-here"
export ENCRYPTION_KEY="your-32-byte-encryption-key-here"

# Performance
export MAX_CONNECTIONS="1000"
export CONNECTION_POOL_SIZE="20"
export CACHE_TTL_DEFAULT="3600"
export BATCH_SIZE="100"

# Monitoring
export METRICS_ENABLED="true"
export HEALTH_CHECK_INTERVAL="30"
export LOG_LEVEL="INFO"
export JAEGER_ENDPOINT="http://jaeger:14268/api/traces"
```

### Development Environment (.env.development)
```bash
# Development-specific overrides
LOG_LEVEL="DEBUG"
METRICS_ENABLED="true"
PATTERN_ANALYSIS_ENABLED="true"
OPTIMIZATION_ENGINE_ENABLED="false"  # Disable auto-optimization in dev

# Reduced resource usage for development
MAX_CONNECTIONS="100"
CONNECTION_POOL_SIZE="5"
INTELLIGENCE_PROCESSOR_WORKERS="2"

# Development database
POSTGRES_URL="postgresql://dev_user:dev_pass@localhost:5432/omninode_bridge_dev"

# Local Redis
REDIS_URL="redis://localhost:6379/0"

# Development Consul
CONSUL_URL="http://localhost:8500"
CONSUL_DATACENTER="dev"
```

### Production Environment (.env.production)
```bash
# Production-specific settings
LOG_LEVEL="INFO"
METRICS_ENABLED="true"
PATTERN_ANALYSIS_ENABLED="true"
OPTIMIZATION_ENGINE_ENABLED="true"

# Production performance settings
MAX_CONNECTIONS="5000"
CONNECTION_POOL_SIZE="50"
INTELLIGENCE_PROCESSOR_WORKERS="16"
BATCH_SIZE="500"

# Production database with SSL
POSTGRES_URL="postgresql://prod_user:secure_pass@db-cluster:5432/omninode_bridge?sslmode=require"

# Production Redis cluster
REDIS_URL="redis://redis-cluster:6379/0"

# Production Consul cluster
CONSUL_URL="https://consul-cluster:8501"
CONSUL_DATACENTER="prod"
CONSUL_ENCRYPTION_KEY="your-consul-encryption-key"

# Enhanced security
JWT_SECRET_KEY="production-jwt-secret-very-long-and-secure"
API_KEY_SALT="production-salt-very-secure"
ENCRYPTION_KEY="32-byte-production-encryption-key"

# Production monitoring
JAEGER_ENDPOINT="https://jaeger-prod:14268/api/traces"
PROMETHEUS_ENDPOINT="http://prometheus:9090"
```

## Service Configuration Files

### HookReceiver Service Configuration
```yaml
# config/hook-receiver.yaml
service:
  name: "hook-receiver"
  port: 8080
  host: "0.0.0.0"
  workers: 4

kafka:
  brokers:
    - "redpanda:9092"
  topics:
    service_lifecycle: "hooks.service_lifecycle"
    execution_intelligence: "hooks.execution_intelligence"
    user_interaction: "hooks.user_interaction"
    git_intelligence: "hooks.git_intelligence"
  producer:
    batch_size: 100
    linger_ms: 50
    compression_type: "gzip"
    max_request_size: 1048576
  consumer:
    group_id: "hook-receiver-group"
    auto_offset_reset: "latest"
    max_poll_records: 500

database:
  url: "${POSTGRES_URL}"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600

intelligence:
  pattern_analysis:
    enabled: true
    confidence_threshold: 0.7
    pattern_cache_size: 1000
    analysis_workers: 2

  session_management:
    enabled: true
    session_timeout: 3600
    cleanup_interval: 300

  metrics_collection:
    enabled: true
    collection_interval: 60
    retention_days: 90

logging:
  level: "${LOG_LEVEL:-INFO}"
  format: "json"
  handlers:
    - type: "console"
    - type: "file"
      filename: "logs/hook-receiver.log"
      max_size: "100MB"
      backup_count: 5

health_checks:
  enabled: true
  interval: 30
  timeout: 5
  endpoints:
    - "/health"
    - "/metrics"
    - "/ready"
```

### ToolCapture Proxy Configuration
```yaml
# config/tool-capture-proxy.yaml
service:
  name: "tool-capture-proxy"
  port: 8090
  host: "0.0.0.0"
  workers: 6

proxy:
  timeout: 30
  max_retries: 3
  retry_delay: 1
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 30
    half_open_max_calls: 10

kafka:
  brokers:
    - "redpanda:9092"
  topics:
    communication_intelligence: "proxy.communication_intelligence"
    performance_metrics: "proxy.performance_metrics"
    optimization_events: "proxy.optimization_events"

redis:
  url: "${REDIS_URL}"
  pool_size: 50
  socket_timeout: 5
  socket_connect_timeout: 5
  cache:
    default_ttl: 3600
    max_cache_size: "500MB"
    eviction_policy: "allkeys-lru"

consul:
  url: "${CONSUL_URL}"
  datacenter: "${CONSUL_DATACENTER}"
  service_discovery:
    enabled: true
    health_check_interval: 10
    deregister_critical_service_after: 60

routing:
  algorithms:
    - "performance_based"
    - "load_balancing"
    - "geographic_proximity"
  default_algorithm: "performance_based"

  performance_weights:
    response_time: 0.4
    success_rate: 0.3
    current_load: 0.2
    availability: 0.1

caching:
  enabled: true
  strategies:
    - "content_based"
    - "user_context"
    - "temporal_patterns"

  cache_rules:
    - pattern: "GET /api/v1/users/*"
      ttl: 300
      vary_by: ["Authorization"]

    - pattern: "GET /api/v1/config/*"
      ttl: 3600
      vary_by: []

    - pattern: "POST /api/v1/query"
      ttl: 60
      vary_by: ["payload_hash"]

intelligence:
  pattern_detection:
    enabled: true
    analysis_depth: "deep"
    real_time_analysis: true

  performance_monitoring:
    enabled: true
    sample_rate: 1.0
    anomaly_detection: true

  optimization:
    auto_apply: true
    confidence_threshold: 0.8
    max_optimizations_per_hour: 10

logging:
  level: "${LOG_LEVEL:-INFO}"
  correlation_tracking: true
  sensitive_data_filtering: true
```

### MCP Registry Configuration
```yaml
# config/mcp-registry.yaml
service:
  name: "mcp-registry"
  port: 8100
  host: "0.0.0.0"

consul:
  url: "${CONSUL_URL}"
  datacenter: "${CONSUL_DATACENTER}"
  watch_interval: 10
  metadata_keys:
    - "mcp_tools"
    - "capabilities"
    - "tool_signatures"

tool_discovery:
  enabled: true
  discovery_methods:
    - "consul_metadata"
    - "service_introspection"
    - "endpoint_scanning"

  validation:
    enabled: true
    signature_validation: true
    capability_verification: true

  registration:
    auto_register: true
    conflict_resolution: "newest_wins"
    deregistration_timeout: 300

hooks:
  installation:
    auto_install: true
    hook_types:
      - "pre_execution"
      - "post_execution"
      - "error_handling"

  monitoring:
    enabled: true
    performance_tracking: true
    usage_analytics: true

security:
  tool_validation:
    enabled: true
    signature_checking: true
    capability_sandboxing: true

  access_control:
    enabled: true
    rbac: true
    api_key_required: true

database:
  url: "${POSTGRES_URL}"
  tables:
    tools: "mcp_tools"
    hooks: "mcp_hooks"
    registrations: "mcp_registrations"
    usage_stats: "mcp_usage_stats"
```

## Infrastructure Configuration

### RedPanda Configuration
```yaml
# config/redpanda.yaml
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

  # Performance tuning
  log_segment_size: 1073741824  # 1GB
  log_retention_ms: 604800000   # 7 days
  log_retention_bytes: 107374182400  # 100GB

  # Replication
  default_topic_replications: 3
  transaction_coordinator_replication: 3

  # Compression
  compression_type: "gzip"

  # Batching
  batch_max_bytes: 1048576
  linger_ms: 50

rpk:
  kafka_api:
    brokers:
      - "localhost:9092"
  admin_api:
    addresses:
      - "localhost:9644"

topics:
  - name: "hooks.service_lifecycle"
    partitions: 6
    replication_factor: 3
    config:
      cleanup.policy: "delete"
      retention.ms: 604800000  # 7 days
      segment.ms: 86400000     # 1 day

  - name: "hooks.execution_intelligence"
    partitions: 12
    replication_factor: 3
    config:
      cleanup.policy: "delete"
      retention.ms: 259200000  # 3 days
      compression.type: "gzip"

  - name: "proxy.communication_intelligence"
    partitions: 8
    replication_factor: 3
    config:
      cleanup.policy: "delete"
      retention.ms: 86400000   # 1 day
```

### Consul Configuration
```hcl
# config/consul.hcl
datacenter = "dc1"
data_dir = "/opt/consul/data"
log_level = "INFO"
node_name = "consul-node-1"
bind_addr = "0.0.0.0"
client_addr = "0.0.0.0"

# Server configuration
server = true
bootstrap_expect = 1

# UI
ui_config {
  enabled = true
}

# API and ports
ports {
  grpc = 8502
  http = 8500
  https = 8501
  dns = 8600
}

# Connect (service mesh)
connect {
  enabled = true
}

# Service discovery
services {
  name = "consul"
  tags = ["consul", "service-discovery"]
  port = 8500
  check {
    http = "http://localhost:8500/v1/status/leader"
    interval = "10s"
  }
}

# ACL (production)
acl = {
  enabled = true
  default_policy = "deny"
  enable_token_persistence = true
}

# Encryption (production)
encrypt = "your-consul-encryption-key-here"
verify_incoming = true
verify_outgoing = true
verify_server_hostname = true

# CA certificates (production)
ca_file = "/etc/consul.d/consul-agent-ca.pem"
cert_file = "/etc/consul.d/consul-agent.pem"
key_file = "/etc/consul.d/consul-agent-key.pem"
```

### PostgreSQL Configuration
```yaml
# config/postgresql.conf
# Connection settings
listen_addresses = '*'
port = 5432
max_connections = 200
superuser_reserved_connections = 3

# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_min_duration_statement = 1000  # Log slow queries

# Replication (production)
wal_level = replica
max_wal_senders = 5
max_replication_slots = 5
hot_standby = on

# Performance monitoring
shared_preload_libraries = 'pg_stat_statements'
track_activities = on
track_counts = on
track_io_timing = on
track_functions = pl
```

## Docker Compose Configuration

### Development Docker Compose
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  redpanda:
    image: redpandadata/redpanda:latest
    command:
      - redpanda
      - start
      - --smp
      - '1'
      - --reserve-memory
      - '0M'
      - --overprovisioned
      - --node-id
      - '0'
      - --kafka-addr
      - 'PLAINTEXT://0.0.0.0:29092,OUTSIDE://0.0.0.0:9092'
      - --advertise-kafka-addr
      - 'PLAINTEXT://redpanda:29092,OUTSIDE://localhost:9092'
    ports:
      - "9092:9092"
      - "29092:29092"
    volumes:
      - redpanda_data:/var/lib/redpanda/data

  consul:
    image: consul:latest
    ports:
      - "8500:8500"
      - "8600:8600/udp"
    volumes:
      - consul_data:/consul/data
      - ./config/consul.hcl:/consul/config/consul.hcl
    command: consul agent -config-file=/consul/config/consul.hcl

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: omninode_bridge_dev
      POSTGRES_USER: dev_user
      POSTGRES_PASSWORD: dev_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgresql.conf:/etc/postgresql/postgresql.conf

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  hook-receiver:
    build:
      context: .
      dockerfile: services/hook-receiver/Dockerfile
    ports:
      - "8080:8080"
    environment:
      - LOG_LEVEL=DEBUG
      - KAFKA_BROKERS=redpanda:29092
      - POSTGRES_URL=postgresql://dev_user:dev_pass@postgres:5432/omninode_bridge_dev
      - CONSUL_URL=http://consul:8500
    depends_on:
      - redpanda
      - postgres
      - consul
    volumes:
      - ./config/hook-receiver.yaml:/app/config/hook-receiver.yaml

  tool-capture-proxy:
    build:
      context: .
      dockerfile: services/tool-capture-proxy/Dockerfile
    ports:
      - "8090:8090"
    environment:
      - LOG_LEVEL=DEBUG
      - KAFKA_BROKERS=redpanda:29092
      - REDIS_URL=redis://redis:6379
      - CONSUL_URL=http://consul:8500
    depends_on:
      - redpanda
      - redis
      - consul
    volumes:
      - ./config/tool-capture-proxy.yaml:/app/config/tool-capture-proxy.yaml

volumes:
  redpanda_data:
  consul_data:
  postgres_data:
  redis_data:
```

### Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  redpanda:
    image: redpandadata/redpanda:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2'
    environment:
      - REDPANDA_ENVIRONMENT=production
    volumes:
      - redpanda_data:/var/lib/redpanda/data
      - ./config/redpanda.yaml:/etc/redpanda/redpanda.yaml
    networks:
      - omninode_network

  consul:
    image: consul:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '1'
    environment:
      - CONSUL_BIND_INTERFACE=eth0
    volumes:
      - consul_data:/consul/data
      - ./config/consul.hcl:/consul/config/consul.hcl
      - ./certs:/consul/certs
    networks:
      - omninode_network

  postgres:
    image: postgres:15
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    environment:
      POSTGRES_DB: omninode_bridge
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgresql.conf:/etc/postgresql/postgresql.conf
    networks:
      - omninode_network

  redis:
    image: redis:7-alpine
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1'
    command: redis-server --appendonly yes --cluster-enabled yes
    volumes:
      - redis_data:/data
    networks:
      - omninode_network

networks:
  omninode_network:
    driver: overlay
    attachable: true

volumes:
  redpanda_data:
  consul_data:
  postgres_data:
  redis_data:
```

## Security Configuration

### JWT Configuration
```yaml
# config/jwt.yaml
jwt:
  secret_key: "${JWT_SECRET_KEY}"
  algorithm: "HS256"
  access_token_expire_minutes: 60
  refresh_token_expire_days: 30

  issuer: "omninode-bridge"
  audience: "omninode-services"

  claims:
    - "service_name"
    - "capabilities"
    - "user_context"
    - "session_id"

# Service-to-service authentication
service_auth:
  enabled: true
  mtls:
    enabled: true
    ca_cert: "/certs/ca.pem"
    cert_file: "/certs/service.pem"
    key_file: "/certs/service-key.pem"
    verify_client_cert: true

  api_keys:
    enabled: true
    key_length: 32
    salt: "${API_KEY_SALT}"
    rotation_days: 90
```

### TLS Configuration
```yaml
# config/tls.yaml
tls:
  enabled: true
  cert_file: "/certs/server.crt"
  key_file: "/certs/server.key"
  ca_file: "/certs/ca.crt"

  min_version: "1.2"
  cipher_suites:
    - "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
    - "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305"
    - "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"

  client_auth: "RequireAndVerifyClientCert"

  certificate_rotation:
    enabled: true
    check_interval: "24h"
    renewal_threshold: "720h"  # 30 days
```

This comprehensive configuration guide ensures all OmniNode Bridge components can be properly configured for development, testing, and production environments with appropriate security, performance, and intelligence settings.
