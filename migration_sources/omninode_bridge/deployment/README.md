# Docker Deployment Guide - Generic ONEX Nodes

This directory contains Docker deployment infrastructure for generic ONEX-compliant nodes in the omninode_bridge ecosystem.

## 📁 Directory Contents

### Dockerfiles (Generic Node Types)

- **`Dockerfile.generic-effect`** - Generic Effect nodes (database_adapter_effect, store_effect)
  - Base image: python:3.12-slim
  - Multi-stage build (development + production)
  - Contract volume mounting at `/contracts`
  - Non-root user: `effectnode`
  - Ports: 9093 (metrics)

- **`Dockerfile.generic-orchestrator`** - Generic Orchestrator nodes (codegen_orchestrator, workflow_orchestrator)
  - Base image: python:3.12-slim
  - Multi-stage build (development + production)
  - Contract volume mounting at `/contracts`
  - Non-root user: `orchestrator`
  - Ports: 8062 (API), 9094 (metrics)

- **`Dockerfile.generic-reducer`** - Generic Reducer nodes (codegen_metrics_reducer, aggregation_reducer)
  - Base image: python:3.12-slim
  - Multi-stage build (development + production)
  - Contract volume mounting at `/contracts`
  - Non-root user: `reducer`
  - Ports: 8063 (API), 9095 (metrics)

### Docker Compose Files

- **`docker-compose.stamping.yml`** - Metadata stamping workflow
  - Services: database_adapter_effect
  - Purpose: Database persistence for metadata stamps
  - External dependencies: postgres, redpanda (from main docker-compose.yml)

- **`docker-compose.metrics.yml`** - Code generation metrics workflow
  - Services: codegen_orchestrator, codegen_metrics_reducer, store_effect
  - Purpose: Metrics collection, aggregation, and storage
  - External dependencies: postgres, redpanda (from main docker-compose.yml)

- **`docker-compose.codegen.yml`** - Code generation infrastructure (existing)
  - Services: redpanda, topic-creator
  - Purpose: Kafka infrastructure for code generation events

### Environment Templates

- **`.env.stamping.example`** - Environment configuration for stamping workflow
- **`.env.codegen.example`** - Environment configuration for metrics workflow

## 🚀 Quick Start

### Prerequisites

1. Ensure Docker and Docker Compose are installed
2. Ensure the main network exists:
   ```bash
   docker network create omninode-bridge-network
   ```

3. Start base infrastructure (PostgreSQL + RedPanda):
   ```bash
   docker-compose -f deployment/docker-compose.yml up -d postgres redpanda
   ```

### Metadata Stamping Workflow

1. **Copy environment template:**
   ```bash
   cp .env.stamping.example .env.stamping
   ```

2. **Configure secrets:**
   Edit `.env.stamping` and set:
   - `POSTGRES_PASSWORD` - Secure database password
   - `API_KEY` - Secure API authentication key

3. **Start stamping workflow:**
   ```bash
   docker-compose -f deployment/docker-compose.stamping.yml --env-file .env.stamping up -d
   ```

4. **Check health:**
   ```bash
   curl http://localhost:8070/health  # Database adapter
   curl http://localhost:9093/metrics  # Prometheus metrics
   ```

5. **View logs:**
   ```bash
   docker-compose -f deployment/docker-compose.stamping.yml logs -f database-adapter-effect
   ```

### Code Generation Metrics Workflow

1. **Copy environment template:**
   ```bash
   cp .env.codegen.example .env.codegen
   ```

2. **Configure secrets:**
   Edit `.env.codegen` and set:
   - `POSTGRES_PASSWORD` - Secure database password
   - `API_KEY` - Secure API authentication key

3. **Create code generation topics:**
   ```bash
   docker-compose -f deployment/docker-compose.codegen.yml up topic-creator
   ```

4. **Start metrics workflow:**
   ```bash
   docker-compose -f deployment/docker-compose.metrics.yml --env-file .env.codegen up -d
   ```

5. **Check health:**
   ```bash
   curl http://localhost:8062/health  # Orchestrator
   curl http://localhost:8063/health  # Reducer
   curl http://localhost:8071/health  # Store Effect
   ```

6. **View metrics:**
   ```bash
   curl http://localhost:9094/metrics  # Orchestrator metrics
   curl http://localhost:9095/metrics  # Reducer metrics
   curl http://localhost:9096/metrics  # Store metrics
   ```

## 🔧 Configuration

### Environment Variables

All services support these common variables:

- **`NODE_TYPE`** - Specifies which node to run (e.g., `database_adapter_effect`)
- **`LOG_LEVEL`** - Logging level (debug, info, warning, error)
- **`ENVIRONMENT`** - Deployment environment (development, staging, production)
- **`BUILD_TARGET`** - Docker build stage (development, production)
- **`MOUNT_MODE`** - Volume mount mode (ro=read-only, rw=read-write)

### Contract Volume Mounting

Each generic node supports dynamic contract loading via volume mounts:

```yaml
volumes:
  - stamping_contracts:/contracts:ro  # Read-only contract volume
```

Contract files should be placed in the named volume and referenced via:
```
CONTRACT_PATH: /contracts/database_adapter_effect.yaml
```

### Port Mappings

| Service | API Port | Metrics Port | Purpose |
|---------|----------|--------------|---------|
| database-adapter-effect | 8070 | 9093 | Database persistence |
| codegen-orchestrator | 8062 | 9094 | Workflow coordination |
| codegen-metrics-reducer | 8063 | 9095 | Metrics aggregation |
| store-effect | 8071 | 9096 | Persistent storage |

## 🏗️ Building Custom Images

### Development Build (with hot reload)

```bash
docker build \
  --target development \
  --build-arg PYTHON_VERSION=3.12 \
  -f deployment/Dockerfile.generic-effect \
  -t omninode-bridge/effect-node:dev .
```

### Production Build (optimized)

```bash
docker build \
  --target production \
  --build-arg PYTHON_VERSION=3.12 \
  -f deployment/Dockerfile.generic-orchestrator \
  -t omninode-bridge/orchestrator-node:latest .
```

### Multi-Architecture Builds

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target production \
  -f deployment/Dockerfile.generic-reducer \
  -t omninode-bridge/reducer-node:latest \
  --push .
```

## 🔍 Health Checks

All nodes include ONEX-compliant health checks:

- **Interval:** 30 seconds
- **Timeout:** 10 seconds
- **Retries:** 3
- **Start Period:** 40 seconds

Health check command:
```bash
python -m omninode_bridge.nodes.health_check_cli <NODE_TYPE>
```

## 📊 Monitoring

### Prometheus Metrics

All nodes expose Prometheus-compatible metrics on their metrics ports:

```yaml
scrape_configs:
  - job_name: 'codegen-orchestrator'
    static_configs:
      - targets: ['localhost:9094']

  - job_name: 'codegen-reducer'
    static_configs:
      - targets: ['localhost:9095']

  - job_name: 'store-effect'
    static_configs:
      - targets: ['localhost:9096']
```

### Logging

All services log to:
- **stdout/stderr** - Captured by Docker
- **`/app/logs`** - Volume-mounted logs directory

View logs:
```bash
docker-compose -f deployment/docker-compose.metrics.yml logs -f <service-name>
```

## 🔒 Security

### Production Deployment Checklist

- ✅ Use strong, randomly generated passwords
- ✅ Enable SSL/TLS for database connections
- ✅ Configure proper network isolation
- ✅ Use Docker secrets or HashiCorp Vault
- ✅ Set `MOUNT_MODE=ro` for read-only volumes
- ✅ Configure resource limits (CPU/memory)
- ✅ Enable webhook signature verification
- ✅ Configure proper CORS origins
- ✅ Use non-root users (automatically configured)

### Secrets Management

**Development:**
Use environment variables in `.env` files (never commit to git)

**Production:**
Use Docker secrets:
```yaml
secrets:
  postgres_password:
    external: true
```

Or HashiCorp Vault:
```yaml
environment:
  VAULT_ENABLED: true
  VAULT_ADDR: https://vault.example.com
  VAULT_ROLE_ID: ${VAULT_ROLE_ID}
  VAULT_SECRET_ID: ${VAULT_SECRET_ID}
```

## 🐛 Troubleshooting

### Service Won't Start

1. **Check dependencies:**
   ```bash
   docker ps | grep -E "(postgres|redpanda)"
   ```

2. **View service logs:**
   ```bash
   docker-compose -f deployment/docker-compose.metrics.yml logs <service-name>
   ```

3. **Check health:**
   ```bash
   docker inspect <container-name> | grep -A 10 Health
   ```

### Port Conflicts

If ports are already in use, update port mappings in `.env`:
```bash
DATABASE_ADAPTER_PORT=8080  # Instead of 8070
```

### Permission Errors

Ensure log/data directories have correct permissions:
```bash
chmod -R 755 logs data
```

### Contract Loading Issues

1. **Verify contract volume:**
   ```bash
   docker volume inspect stamping_contracts
   ```

2. **Check contract path:**
   ```bash
   docker exec <container-name> ls -la /contracts
   ```

## 📈 Performance Tuning

### Orchestrator Performance

```yaml
environment:
  ORCHESTRATOR_CONCURRENCY: 20  # Increase concurrent workflows
  MAX_WORKFLOW_QUEUE_SIZE: 2000
  WORKFLOW_TIMEOUT_MS: 600000
```

### Reducer Performance

```yaml
environment:
  REDUCER_CONCURRENCY: 20
  MAX_AGGREGATION_QUEUE_SIZE: 10000
  AGGREGATION_WINDOW_SIZE_MS: 3000  # Reduce window for faster aggregation
  AGGREGATION_BATCH_SIZE: 200
```

### Resource Limits

Add to docker-compose.yml:
```yaml
services:
  codegen-orchestrator:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

## 🔗 Integration

### With Existing Services

The generic nodes are designed to work with existing omninode_bridge services:

- **metadata-stamping** - Main stamping service
- **onextree** - Project structure intelligence
- **hook-receiver** - Service lifecycle management
- **model-metrics** - AI Lab integration

### With External Services

- **PostgreSQL** - Database persistence
- **RedPanda/Kafka** - Event streaming
- **Consul** - Service discovery
- **Vault** - Secrets management
- **Prometheus** - Metrics collection
- **Grafana** - Visualization

## 📝 Development Workflow

### Local Development

1. **Use development build target:**
   ```yaml
   BUILD_TARGET=development
   MOUNT_MODE=rw
   ```

2. **Enable hot reload:**
   Docker automatically reloads on source changes with development target

3. **Debug logging:**
   ```yaml
   LOG_LEVEL=debug
   ```

### Testing

Run integration tests:
```bash
# Start all services
docker-compose -f deployment/docker-compose.metrics.yml up -d

# Run tests
pytest tests/integration/ -v

# Stop services
docker-compose -f deployment/docker-compose.metrics.yml down
```

## 📚 Additional Resources

- **[Architecture Guide](../docs/architecture/ARCHITECTURE.md)** - System architecture
- **[Bridge Nodes Guide](../docs/guides/BRIDGE_NODES_GUIDE.md)** - Bridge node implementation
- **[API Reference](../docs/api/API_REFERENCE.md)** - API documentation
- **[ONEX Architecture Patterns](../docs/ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md)** - ONEX patterns

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review service logs
3. Check [project documentation](../docs/INDEX.md)
4. Create an issue in the repository

---

**Version:** 1.0.0
**Last Updated:** October 2025
**Maintainer:** OmniNode Bridge Team
