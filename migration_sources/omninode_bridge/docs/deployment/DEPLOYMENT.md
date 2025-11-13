# OmniNode Bridge - Deployment Guide

Complete deployment guide for ONEX-compliant bridge nodes (orchestrator and reducer).

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment Modes](#deployment-modes)
- [Scaling](#scaling)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Production Best Practices](#production-best-practices)

## Overview

```text
The OmniNode Bridge consists of two ONEX-compliant nodes:

1. **NodeBridgeOrchestrator** (Port 8060)
   - Workflow coordination and FSM management
   - Routes requests to MetadataStampingService and OnexTree
   - Publishes events to Kafka
   - Manages workflow state transitions

2. **NodeBridgeReducer** (Port 8061)
   - Metadata aggregation and state persistence
   - Streams data from orchestrator workflows
   - Persists aggregated state to PostgreSQL
   - Computes aggregation statistics
```

### Infrastructure Components

```text
- **PostgreSQL 15**: Shared database for state persistence
- **RedPanda**: Kafka-compatible event streaming
- **Consul** (optional): Service discovery
```

## Prerequisites

### Required

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### Optional

- Kubernetes 1.24+ (for Kubernetes deployment)
- Helm 3.0+ (for Kubernetes deployment)

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd omninode_bridge

# Copy environment template
cp .env.example .env

# Edit configuration (required)
vim .env  # Update POSTGRES_PASSWORD and other secrets
```

### 2. Development Mode

```bash
# Start all services with hot reload
docker-compose -f docker-compose.bridge.yml up

# Or build and start in background
docker-compose -f docker-compose.bridge.yml up -d --build

# View logs
docker-compose -f docker-compose.bridge.yml logs -f orchestrator
docker-compose -f docker-compose.bridge.yml logs -f reducer
```

### 3. Access Services

- **Orchestrator API**: http://localhost:8060
- **Reducer API**: http://localhost:8061
- **Orchestrator Metrics**: http://localhost:9091/metrics
- **Reducer Metrics**: http://localhost:9092/metrics
- **PostgreSQL**: localhost:5436
- **RedPanda Kafka**: localhost:29092
- **RedPanda Admin**: http://localhost:29654

### 4. Health Checks

```bash
# Orchestrator health
curl http://localhost:8060/health

# Reducer health
curl http://localhost:8061/health

# Database health
docker exec omninode-bridge-postgres pg_isready -U postgres

# Kafka health
docker exec omninode-bridge-redpanda rpk cluster info
```

### 5. Stop Services

```bash
# Stop services
docker-compose -f docker-compose.bridge.yml down

# Stop and remove volumes (data loss!)
docker-compose -f docker-compose.bridge.yml down -v
```

## Configuration

### Environment Variables

Key configuration variables in `.env`:

#### Bridge Nodes

```bash
# Orchestrator
ORCHESTRATOR_PORT=8060
ORCHESTRATOR_WORKERS=4
ORCHESTRATOR_CONCURRENCY=10
MAX_WORKFLOW_QUEUE_SIZE=1000

# Reducer
REDUCER_PORT=8061
REDUCER_WORKERS=4
REDUCER_CONCURRENCY=10
MAX_AGGREGATION_QUEUE_SIZE=5000

# Aggregation settings
AGGREGATION_WINDOW_SIZE_MS=5000
AGGREGATION_BATCH_SIZE=100
```

#### Database

```bash
POSTGRES_HOST=omninode-bridge-postgres
POSTGRES_PORT=5432
POSTGRES_DATABASE=omninode_bridge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-secure-password  # CHANGE THIS!
```

#### Kafka/RedPanda

```bash
KAFKA_BOOTSTRAP_SERVERS=omninode-bridge-redpanda:9092
KAFKA_WORKFLOW_TOPIC=dev.omninode_bridge.onex.workflows.v1
```

#### Service URLs

```bash
METADATA_STAMPING_SERVICE_URL=http://metadata-stamping:8053
ONEXTREE_SERVICE_URL=http://onextree:8080
DEFAULT_NAMESPACE=omninode.bridge
```

### Build Configuration

```bash
# Build target: development or production
BUILD_TARGET=production

# Volume mount mode
MOUNT_MODE=ro  # Use 'rw' for development hot reload
```

## Deployment Modes

### Development Mode

Full development environment with hot reload:

```bash
# Set environment
export BUILD_TARGET=development
export MOUNT_MODE=rw
export LOG_LEVEL=debug

# Start services
docker-compose -f docker-compose.bridge.yml up --build
```

Features:
- Hot reload on code changes
- Debug logging enabled
- Development dependencies included
- Source code mounted as volumes

### Production Mode

Optimized production deployment:

```bash
# Set environment
export BUILD_TARGET=production
export MOUNT_MODE=ro
export LOG_LEVEL=info
export ENVIRONMENT=production

# Start services
docker-compose -f docker-compose.bridge.yml up -d --build

# Run database migrations
docker-compose -f docker-compose.bridge.yml --profile migrations up
```

Features:
- Multi-worker uvicorn processes
- Production dependencies only
- Read-only source mounts
- Health checks enabled
- Resource limits enforced

### With Service Discovery (Consul)

Include Consul for service discovery:

```bash
# Start with Consul
docker-compose -f docker-compose.bridge.yml --profile full up -d

# Verify Consul
curl http://localhost:28500/ui/
```

## Scaling

### Horizontal Scaling

Scale orchestrator and reducer independently:

```bash
# Scale orchestrator to 3 instances
docker-compose -f docker-compose.bridge.yml up -d --scale orchestrator=3

# Scale reducer to 2 instances
docker-compose -f docker-compose.bridge.yml up -d --scale reducer=2

# Scale both
docker-compose -f docker-compose.bridge.yml up -d \
  --scale orchestrator=3 \
  --scale reducer=2
```

**Note**: When scaling:
- Remove `container_name` from docker-compose for scaled services
- Use service discovery (Consul) for dynamic instance tracking
- Configure load balancing appropriately

### Vertical Scaling

Adjust resource limits in `docker-compose.bridge.yml`:

```yaml
services:
  orchestrator:
    deploy:
      resources:
        limits:
          cpus: '4.0'      # Increase from 2.0
          memory: 2G       # Increase from 1G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Performance Tuning

#### Orchestrator Tuning

```bash
# Increase workflow capacity
ORCHESTRATOR_WORKERS=8
ORCHESTRATOR_CONCURRENCY=20
MAX_WORKFLOW_QUEUE_SIZE=5000

# Adjust timeouts
ORCHESTRATOR_WORKFLOW_TIMEOUT_MS=600000
ORCHESTRATOR_STEP_TIMEOUT_MS=120000
```

#### Reducer Tuning

```bash
# Increase aggregation capacity
REDUCER_WORKERS=8
REDUCER_CONCURRENCY=20
MAX_AGGREGATION_QUEUE_SIZE=10000

# Optimize aggregation
AGGREGATION_WINDOW_SIZE_MS=2000  # Smaller windows for faster processing
AGGREGATION_BATCH_SIZE=500       # Larger batches for efficiency
```

#### Database Tuning

```bash
# Connection pool
POSTGRES_POOL_MIN_SIZE=10
POSTGRES_POOL_MAX_SIZE=50

# Query performance
POSTGRES_QUERY_TIMEOUT=60
POSTGRES_ACQUIRE_TIMEOUT=15
```

## Monitoring

### Prometheus Metrics

Both nodes expose Prometheus-compatible metrics:

```bash
# Orchestrator metrics
curl http://localhost:9091/metrics

# Reducer metrics
curl http://localhost:9092/metrics
```

Key metrics to monitor:

**Orchestrator**:
- `workflow_orchestration_duration_ms` - Workflow processing time
- `workflow_orchestration_total` - Total workflows processed
- `workflow_orchestration_errors_total` - Failed workflows
- `workflow_queue_size` - Current queue depth

**Reducer**:
- `aggregation_duration_ms` - Aggregation processing time
- `aggregation_items_total` - Total items aggregated
- `aggregation_errors_total` - Failed aggregations
- `aggregation_queue_size` - Current queue depth

### Health Monitoring

```bash
# Automated health check script
#!/bin/bash
services=("orchestrator:8060" "reducer:8061" "postgres:5432")

for service in "${services[@]}"; do
    name="${service%:*}"
    port="${service#*:}"

    if curl -sf "http://localhost:${port}/health" > /dev/null; then
        echo "✓ $name is healthy"
    else
        echo "✗ $name is unhealthy"
    fi
done
```

### Log Aggregation

View logs from all services:

```bash
# All services
docker-compose -f docker-compose.bridge.yml logs -f

# Specific service
docker-compose -f docker-compose.bridge.yml logs -f orchestrator

# With timestamps
docker-compose -f docker-compose.bridge.yml logs -f -t

# Last 100 lines
docker-compose -f docker-compose.bridge.yml logs --tail=100
```

## Troubleshooting

### Common Issues

#### 1. Orchestrator Won't Start

**Symptoms**: Orchestrator container exits immediately

**Diagnosis**:
```bash
# Check logs
docker-compose -f docker-compose.bridge.yml logs orchestrator

# Check database connectivity
docker exec omninode-bridge-orchestrator \
  nc -zv postgres 5432
```

**Solutions**:
- Verify PostgreSQL is healthy: `docker-compose -f docker-compose.bridge.yml ps`
- Check database credentials in `.env`
- Ensure database migrations ran: `docker-compose -f docker-compose.bridge.yml --profile migrations up`

#### 2. Reducer Aggregation Slow

**Symptoms**: High latency in aggregation processing

**Diagnosis**:
```bash
# Check reducer metrics
curl http://localhost:9092/metrics | grep aggregation_duration

# Check queue depth
curl http://localhost:9092/metrics | grep aggregation_queue_size
```

**Solutions**:
- Increase `REDUCER_WORKERS` and `REDUCER_CONCURRENCY`
- Optimize `AGGREGATION_BATCH_SIZE` based on workload
- Reduce `AGGREGATION_WINDOW_SIZE_MS` for faster processing
- Scale reducer horizontally: `--scale reducer=3`

#### 3. Kafka Connection Errors

**Symptoms**: "Failed to publish Kafka event" in logs

**Diagnosis**:
```bash
# Check RedPanda health
docker exec omninode-bridge-redpanda rpk cluster info

# Check topics exist
docker exec omninode-bridge-redpanda rpk topic list
```

**Solutions**:
- Verify RedPanda is healthy: `docker-compose -f docker-compose.bridge.yml ps redpanda`
- Run topic initialization: Topics are auto-created by `redpanda-topics` service
- Check `KAFKA_BOOTSTRAP_SERVERS` in `.env`

#### 4. High Memory Usage

**Symptoms**: Container OOM kills or high memory consumption

**Diagnosis**:
```bash
# Check memory usage
docker stats omninode-bridge-orchestrator
docker stats omninode-bridge-reducer

# Check resource limits
docker inspect omninode-bridge-orchestrator | grep -A 5 Memory
```

**Solutions**:
- Reduce `MAX_WORKFLOW_QUEUE_SIZE` and `MAX_AGGREGATION_QUEUE_SIZE`
- Decrease worker count (`ORCHESTRATOR_WORKERS`, `REDUCER_WORKERS`)
- Increase Docker memory limits in `docker-compose.bridge.yml`
- Implement queue backpressure mechanisms

#### 5. Database Connection Pool Exhausted

**Symptoms**: "Connection pool exhausted" errors

**Diagnosis**:
```bash
# Check active connections
docker exec omninode-bridge-postgres \
  psql -U postgres -d omninode_bridge \
  -c "SELECT count(*) FROM pg_stat_activity;"
```

**Solutions**:
- Increase `POSTGRES_POOL_MAX_SIZE`
- Reduce `POSTGRES_CONNECTION_MAX_AGE`
- Enable `POSTGRES_LEAK_DETECTION=true` to identify leaks
- Reduce concurrent workers if connection count is high

### Debug Mode

Enable debug logging for detailed diagnostics:

```bash
# Set debug level
export LOG_LEVEL=debug

# Rebuild and restart
docker-compose -f docker-compose.bridge.yml up -d --build

# View debug logs
docker-compose -f docker-compose.bridge.yml logs -f | grep DEBUG
```

## Production Best Practices

### Security

1. **Change Default Passwords**
   ```bash
   # Generate strong passwords
   openssl rand -base64 32

   # Update .env
   POSTGRES_PASSWORD=<generated-password>
   JWT_SECRET=<generated-secret>
   API_KEY=<generated-key>
   ```

2. **Enable SSL/TLS**
   ```bash
   # Database SSL
   POSTGRES_SSL_ENABLED=true
   POSTGRES_SSL_MODE=require

   # HTTPS
   REQUIRE_HTTPS=true
   ```

3. **Network Isolation**
   - Use Docker internal networks
   - Expose only necessary ports
   - Configure firewall rules
   - Use reverse proxy (nginx/traefik)

4. **API Authentication**
   ```bash
   # Enable JWT authentication
   JWT_EXPIRATION_HOURS=24

   # Enable API key validation
   API_KEY_ROTATION_ENABLED=true
   API_KEY_ROTATION_DAYS=30
   ```

### High Availability

1. **Database Replication**
   - Set up PostgreSQL replication
   - Configure read replicas for analytics
   - Implement automatic failover

2. **Load Balancing**
   - Use nginx or HAProxy
   - Configure health check endpoints
   - Implement sticky sessions if needed

3. **Backup Strategy**
   ```bash
   # Automated PostgreSQL backups
   docker exec omninode-bridge-postgres \
     pg_dump -U postgres omninode_bridge > backup.sql

   # Volume backups
   docker run --rm -v omninode-bridge-postgres-data:/data \
     -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
   ```

### Resource Management

1. **CPU Limits**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2.0'
       reservations:
         cpus: '0.5'
   ```

2. **Memory Limits**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 1G
       reservations:
         memory: 512M
   ```

3. **Disk I/O**
   - Use SSD storage for PostgreSQL
   - Configure volume drivers appropriately
   - Monitor disk usage

### Monitoring and Alerting

1. **Prometheus Integration**
   ```yaml
   # prometheus.yml
   scrape_configs:
     - job_name: 'orchestrator'
       static_configs:
         - targets: ['orchestrator:9091']

     - job_name: 'reducer'
       static_configs:
         - targets: ['reducer:9092']
   ```

2. **Grafana Dashboards**
   - Import pre-built dashboards for FastAPI/uvicorn
   - Create custom dashboards for bridge metrics
   - Set up alerting rules

3. **Health Check Automation**
   ```bash
   # Kubernetes liveness probe
   livenessProbe:
     httpGet:
       path: /health
       port: 8060
     initialDelaySeconds: 40
     periodSeconds: 30
   ```

### Updates and Rollbacks

1. **Rolling Updates**
   ```bash
   # Update orchestrator
   docker-compose -f docker-compose.bridge.yml up -d --no-deps --build orchestrator

   # Update reducer
   docker-compose -f docker-compose.bridge.yml up -d --no-deps --build reducer
   ```

2. **Rollback Strategy**
   ```bash
   # Tag images before updates
   docker tag omninode-bridge-orchestrator:latest omninode-bridge-orchestrator:backup

   # Rollback if needed
   docker-compose -f docker-compose.bridge.yml down
   # Update docker-compose.yml to use backup tag
   docker-compose -f docker-compose.bridge.yml up -d
   ```

3. **Database Migrations**
   ```bash
   # Test migrations first
   docker-compose -f docker-compose.bridge.yml --profile migrations up --dry-run

   # Run migrations
   docker-compose -f docker-compose.bridge.yml --profile migrations up

   # Rollback if needed
   docker-compose -f docker-compose.bridge.yml exec migrations \
     poetry run alembic downgrade -1
   ```

## Support and Resources

- **Documentation**: See project README.md
- **Issues**: GitHub Issues
- **Architecture**: See ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md
- **API Documentation**:
  - Orchestrator: http://localhost:8060/docs
  - Reducer: http://localhost:8061/docs

## License

MIT License - see LICENSE file for details
