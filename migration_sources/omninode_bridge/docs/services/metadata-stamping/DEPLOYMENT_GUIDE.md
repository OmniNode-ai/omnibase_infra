# Metadata Stamping Service Deployment Guide

## Prerequisites Validation

### 1. Infrastructure Health Check
```bash
# Verify all dependencies are healthy
docker ps --filter "name=omninode-bridge" --format "table {{.Names}}\t{{.Status}}"

# Confirm required services are running:
# - omninode-bridge-postgres (healthy)
# - omninode-bridge-consul (healthy)
# - omninode-bridge-redpanda (healthy)
# - omninode-bridge-vault (healthy)
```

### 2. Port Availability Verification
```bash
# Verify port 8057 is available (metadata-stamping external port)
netstat -tulpn | grep :8057 || echo "Port 8057 is available"

# Verify port 9091 is available (metadata-stamping metrics)
netstat -tulpn | grep :9091 || echo "Port 9091 is available"

# Confirm port 8053 conflict (should be used by archon-intelligence)
netstat -tulpn | grep :8053 && echo "Port 8053 conflict confirmed - using 8057 instead"
```

## Deployment Steps

### Phase 1: Configuration Updates

#### 1. Update Main Docker Compose
```bash
# Navigate to project root
cd omninode_bridge  # or your repository directory

# Backup existing configuration
cp docker-compose.yml docker-compose.yml.backup

# Apply metadata-stamping service integration
# Add the service definition from docker-compose-metadata-patch.yml to docker-compose.yml
# Insert after the workflow-coordinator service and before redpanda-ui
```

#### 2. Update Kafka Topic Manager
```bash
# Update redpanda-topic-manager command section in docker-compose.yml
# Add metadata stamping topics from docker-compose-metadata-patch.yml
# This includes:
# - Event topics (stamp-created, stamp-validated, batch-processed)
# - Command topics (hash-generate, stamp-validate)
# - Retry/DLT topics
# - Audit/Metrics topics
```

#### 3. Environment Configuration
```bash
# Merge metadata stamping variables into main .env file
cat .env.metadata-stamping >> .env

# Verify critical environment variables
grep "METADATA_STAMPING" .env | head -10
```

### Phase 2: Database Schema Preparation

#### 1. Create Metadata Stamping Tables
```bash
# Run metadata stamping migrations
poetry run alembic upgrade head

# Verify tables were created
docker exec -it omninode-bridge-postgres psql -U postgres -d omninode_bridge -c "\\dt metadata*"
```

#### 2. Database Performance Tuning
```sql
-- Connect to PostgreSQL and optimize for metadata stamping
-- These commands enhance performance for BLAKE3 hash lookups

-- Create performance indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metadata_stamps_file_hash_btree
ON metadata_stamps USING btree(file_hash);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metadata_stamps_namespace_created
ON metadata_stamps(namespace, created_at DESC);

-- Analyze tables for query optimization
ANALYZE metadata_stamps;
ANALYZE hash_metrics;
```

### Phase 3: Service Deployment

#### 1. Build and Deploy Service
```bash
# Build the metadata stamping service image
docker-compose build metadata-stamping

# Start dependencies first (should already be running)
docker-compose up -d postgres consul redpanda

# Wait for dependencies to be healthy
docker-compose ps | grep "healthy"

# Deploy metadata stamping service
docker-compose up -d metadata-stamping

# Monitor startup logs
docker-compose logs -f metadata-stamping
```

#### 2. Service Registration Verification
```bash
# Verify service registered with Consul
curl -s http://localhost:28500/v1/agent/services | jq '.["metadata-stamping-service"]'

# Check service health in Consul
curl -s http://localhost:28500/v1/health/service/metadata-stamping-service | jq '.[].Checks[].Status'
```

### Phase 4: Validation and Testing

#### 1. Health Check Validation
```bash
# Direct health check
curl -f http://localhost:8057/api/v1/metadata-stamping/health

# Expected response:
# {
#   "status": "healthy",
#   "components": {
#     "database": "healthy",
#     "kafka": "healthy",
#     "hash_generator": "healthy"
#   },
#   "timestamp": "2024-09-29T..."
# }
```

#### 2. Performance Validation - Sub-2ms BLAKE3
```bash
# Test BLAKE3 hash generation performance
curl -X POST http://localhost:8057/api/v1/metadata-stamping/hash \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test-file.txt"

# Verify execution_time_ms < 2.0 in response
# {
#   "status": "success",
#   "data": {
#     "hash": "blake3_hash_here",
#     "execution_time_ms": 0.8,
#     "performance_grade": "A"
#   }
# }
```

#### 3. O.N.E. v0.1 Protocol Compliance Test
```bash
# Test stamp creation with O.N.E. protocol
curl -X POST http://localhost:8057/api/v1/metadata-stamping/stamp \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "test content",
    "namespace": "omninode.services.metadata",
    "metadata": {
      "author": "test-user",
      "description": "O.N.E. protocol test"
    }
  }'

# Verify O.N.E. v0.1 fields in response:
# - op_id (UUID)
# - namespace ("omninode.services.metadata")
# - version (1)
# - metadata_version ("0.1")
```

#### 4. Kafka Event Publishing Validation
```bash
# Monitor Kafka topics for metadata events
docker exec -it omninode-bridge-redpanda rpk topic consume dev.omninode_bridge.onex.evt.metadata-stamp-created.v1 --num 1

# Expected event format:
# {
#   "envelope": {
#     "event_id": "uuid",
#     "event_type": "metadata_stamp_created",
#     "namespace": "omninode.services.metadata",
#     "actor_id": "metadata-stamping-service"
#   },
#   "payload": {
#     "stamp_id": "uuid",
#     "file_hash": "blake3_hash",
#     "op_id": "uuid"
#   }
# }
```

### Phase 5: Monitoring and Observability

#### 1. Prometheus Metrics Verification
```bash
# Check metadata stamping metrics
curl -s http://localhost:9091/metrics | grep metadata_stamping | head -10

# Key metrics to monitor:
# - metadata_stamping_hash_generation_duration_seconds
# - metadata_stamping_api_requests_total
# - metadata_stamping_database_connections_active
# - metadata_stamping_kafka_events_published_total
```

#### 2. Performance Dashboard Setup
```bash
# Import metadata stamping dashboard to Grafana (if available)
# Dashboard should include:
# - BLAKE3 hash generation latency (target: <2ms)
# - API response times (target: <10ms)
# - Database connection pool utilization
# - Kafka event publishing rate
# - Service uptime and health status
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Port Conflict (8053 already in use)
```bash
# Verify archon-intelligence is using port 8053
docker ps | grep archon-intelligence

# Solution: Use port mapping 8057:8053 (already configured)
# Access service via http://localhost:8057 instead of 8053
```

#### 2. Database Connection Failures
```bash
# Check PostgreSQL connectivity
docker exec -it omninode-bridge-metadata-stamping pg_isready -h omninode-bridge-postgres -p 5432

# Verify database permissions
docker exec -it omninode-bridge-postgres psql -U postgres -d omninode_bridge -c "SELECT current_user, current_database();"
```

#### 3. Kafka Connection Issues
```bash
# Test Kafka connectivity
docker exec -it omninode-bridge-metadata-stamping curl -f http://omninode-bridge-redpanda:9092

# Verify topics exist
docker exec -it omninode-bridge-redpanda rpk topic list | grep metadata
```

#### 4. Performance Issues (>2ms BLAKE3)
```bash
# Check container resource usage
docker stats omninode-bridge-metadata-stamping

# Increase hash generator pool size if needed
# Set METADATA_STAMPING_HASH_GENERATOR_POOL_SIZE=200
# Set METADATA_STAMPING_HASH_GENERATOR_MAX_WORKERS=6
```

## Success Criteria Checklist

- [ ] Service starts successfully and shows "healthy" status
- [ ] Registers with Consul service discovery
- [ ] BLAKE3 hash generation consistently <2ms (p99)
- [ ] API responses consistently <10ms (p95)
- [ ] Database connection pool efficiency >90%
- [ ] Kafka events published successfully
- [ ] O.N.E. v0.1 protocol compliance validated
- [ ] Prometheus metrics exposed and collecting
- [ ] Health checks passing across all dependencies
- [ ] No port conflicts with existing services

## Performance Benchmarks

### Expected Performance Metrics
```
BLAKE3 Hash Generation:
- P99: <2.0ms
- P95: <1.5ms
- Average: <1.0ms

API Response Times:
- P95: <10ms
- P99: <25ms
- Average: <5ms

Database Operations:
- P95: <5ms
- Connection Pool Efficiency: >90%

Service Availability:
- Uptime: >99.9%
- Health Check Success Rate: >95%
```

## Post-Deployment Monitoring

### Daily Health Checks
```bash
# Automated health check script
#!/bin/bash
echo "=== Metadata Stamping Service Health Check ==="
echo "Service Status: $(curl -s http://localhost:8057/api/v1/metadata-stamping/health | jq -r '.status')"
echo "Container Status: $(docker inspect omninode-bridge-metadata-stamping --format '{{.State.Health.Status}}')"
echo "Consul Registration: $(curl -s http://localhost:28500/v1/health/service/metadata-stamping-service | jq -r '.[0].Checks[0].Status')"
echo "Performance Grade: $(curl -s -X POST http://localhost:8057/api/v1/metadata-stamping/hash -F 'file=@/dev/urandom' | jq -r '.data.performance_grade')"
```

### Weekly Performance Review
- Review Prometheus metrics for performance trends
- Analyze database query performance
- Check Kafka event publishing rates
- Validate O.N.E. protocol compliance
- Review error logs and alert history

This deployment guide ensures successful integration of the metadata stamping service with the existing omninode-bridge infrastructure while maintaining O.N.E. v0.1 protocol compliance and sub-2ms performance targets.
