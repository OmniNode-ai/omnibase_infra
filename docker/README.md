# ONEX Infrastructure Runtime - Docker Deployment

This directory contains Docker configuration for deploying the ONEX Infrastructure Runtime. The runtime provides a contract-driven kernel for event-based infrastructure orchestration.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Profiles](#profiles)
- [Security](#security)
- [Health Checks](#health-checks)
- [Performance Optimization](#performance-optimization)
- [Environment Variables](#environment-variables)
- [Volume Management](#volume-management)
- [Troubleshooting](#troubleshooting)
- [CI/CD Integration](#cicd-integration)
- [Development Workflow](#development-workflow)

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env

# 2. CRITICAL: Edit .env and replace ALL placeholder values
#    Look for: __REPLACE_WITH_SECURE_PASSWORD__, __REPLACE_WITH_VAULT_TOKEN__
#    Generate secure passwords: openssl rand -base64 32

# 3. Build and start (main profile)
docker compose -f docker-compose.runtime.yml --profile main up -d --build

# 4. View logs
docker compose -f docker-compose.runtime.yml logs -f

# 5. Verify health
curl http://localhost:8085/health
```

## Architecture

The runtime deployment follows a multi-profile architecture designed for flexible scaling and separation of concerns:

```
                                    Docker Compose Profiles
+--------------------------------------------------------------------+
|                                                                    |
|   main profile          effects profile       workers profile      |
|   +---------------+     +---------------+     +---------------+    |
|   |               |     |               |     |               |    |
|   | runtime-main  |     | runtime-      |     | runtime-      |    |
|   | (core kernel) |     | effects       |     | worker x N    |    |
|   |               |     | (ext. I/O)    |     | (parallel)    |    |
|   +-------+-------+     +-------+-------+     +-------+-------+    |
|           |                     |                     |            |
|           +---------------------+---------------------+            |
|                                 |                                  |
|                    +------------+------------+                     |
|                    |                         |                     |
|                    | omnibase-infra-network  |                     |
|                    |    (bridge network)     |                     |
|                    |                         |                     |
|                    +------------+------------+                     |
|                                 |                                  |
|                    +------------+------------+                     |
|                    |    External Services    |                     |
|                    |  (via extra_hosts or    |                     |
|                    |   external network)     |                     |
|                    +-------------------------+                     |
|                                                                    |
+--------------------------------------------------------------------+
```

### Service Communication Flow

```
External Request
       |
       v
+------+-------+
| runtime-main |  <-- Listens on requests topic, publishes to responses
+------+-------+
       |
       | (Event Bus - Kafka/Redpanda or InMemory)
       v
+------+-----------+     +-----------------+
| runtime-effects  |     | runtime-worker  |
| (external I/O)   |     | (compute tasks) |
+------------------+     +-----------------+
       |                         |
       v                         v
+------+-----------+     +-------+--------+
| External Services|     | Parallel Jobs  |
| - PostgreSQL     |     | - Compute      |
| - Consul         |     | - Transform    |
| - Vault          |     | - Aggregate    |
| - Kafka          |     +----------------+
+------------------+
```

## Profiles

| Profile   | Services                  | Use Case                                   | Command                                                        |
|-----------|---------------------------|--------------------------------------------|----------------------------------------------------------------|
| `main`    | runtime-main              | Core kernel only                           | `docker compose -f docker-compose.runtime.yml --profile main up -d` |
| `effects` | runtime-main + effects    | Main + external service interactions       | `docker compose -f docker-compose.runtime.yml --profile effects up -d` |
| `workers` | runtime-main + workers x2 | Main + parallel compute processing         | `docker compose -f docker-compose.runtime.yml --profile workers up -d` |
| `all`     | All services              | Full deployment with all profiles          | `docker compose -f docker-compose.runtime.yml --profile all up -d` |

### Profile Details

#### Main Profile (Core Kernel)

The primary runtime service that handles kernel bootstrap:

- Listens on configurable input topic (default: `requests`)
- Publishes to configurable output topic (default: `responses`)
- Exposes health endpoint on port 8085
- Must be healthy before effects/workers can start

```bash
docker compose -f docker-compose.runtime.yml --profile main up -d --build
```

#### Effects Profile (External Interactions)

Handles effect nodes that interact with external services:

- Consul for service discovery
- Kafka for event streaming
- Vault for secret management
- PostgreSQL for persistence

```bash
docker compose -f docker-compose.runtime.yml --profile effects up -d --build
```

#### Workers Profile (Parallel Processing)

Scalable worker containers for parallel compute:

```bash
# Start with default replicas (2)
docker compose -f docker-compose.runtime.yml --profile workers up -d --build

# Scale workers manually
docker compose -f docker-compose.runtime.yml --profile workers up -d --scale runtime-worker=4

# Or set via environment variable
WORKER_REPLICAS=8 docker compose -f docker-compose.runtime.yml --profile workers up -d
```

## Security

### Credential Management

**CRITICAL**: All placeholder values (`__REPLACE_WITH_*__`) in `.env.example` MUST be changed before ANY deployment.

```bash
# Generate secure random passwords
openssl rand -base64 32

# Generate multiple passwords at once
for i in {1..3}; do echo "Password $i: $(openssl rand -base64 32)"; done
```

### Required Credentials

| Credential         | Environment Variable  | Notes                                      |
|--------------------|-----------------------|--------------------------------------------|
| PostgreSQL         | `POSTGRES_PASSWORD`   | Database access password                   |
| Vault              | `VAULT_TOKEN`         | HashiCorp Vault access token               |
| Redis/Valkey       | `REDIS_PASSWORD`      | Cache access password                      |
| GitHub (build)     | `GITHUB_TOKEN`        | For private package installation           |

### Security Features

| Feature                  | Description                                          | Verification                              |
|--------------------------|------------------------------------------------------|-------------------------------------------|
| Non-root execution       | Container runs as `omniinfra` user (UID 1000)        | `docker exec <container> whoami`          |
| BuildKit secrets         | GitHub tokens passed via secret mounts               | `docker history --no-trunc`               |
| No hardcoded credentials | All secrets via environment variables                | Review Dockerfile                         |
| Resource limits          | CPU and memory limits prevent exhaustion             | `docker stats`                            |
| Read-only contracts      | Contract volume mounted read-only                    | `:ro` in docker-compose                   |

### Security Verification

```bash
# Verify non-root user
docker exec omnibase-infra-runtime-main whoami
# Expected output: omniinfra

# Verify UID/GID
docker exec omnibase-infra-runtime-main id
# Expected: uid=1000(omniinfra) gid=1000(omniinfra) groups=1000(omniinfra)

# Verify secrets not in image history
docker history omnibase-infra-runtime --no-trunc | grep -iE "token|password|secret"
# Expected: no output (empty result)

# Verify no secrets in environment inspection
docker inspect omnibase-infra-runtime-main --format='{{json .Config.Env}}' | grep -i password
# Should only show placeholder references, not actual values
```

### Security Best Practices

1. **Never commit `.env` files** - Already in `.gitignore`
2. **Rotate credentials regularly** - Use vault or secret manager
3. **Use minimal permissions** - Vault tokens should have scoped policies
4. **Enable audit logging** - Track credential access
5. **Network isolation** - Use internal Docker networks

## Health Checks

The runtime exposes HTTP health endpoints for container orchestration platforms.

### Endpoints

| Endpoint  | Method | Description          | Success | Failure |
|-----------|--------|----------------------|---------|---------|
| `/health` | GET    | Liveness probe       | 200     | 503     |
| `/ready`  | GET    | Readiness probe      | 200     | 503     |

### Health Check Commands

```bash
# Test health endpoint
curl -s http://localhost:8085/health | jq .

# Test from inside container network
docker exec omnibase-infra-runtime-main curl -s http://localhost:8085/health

# Check container health status
docker inspect --format='{{.State.Health.Status}}' omnibase-infra-runtime-main

# View health check history
docker inspect --format='{{json .State.Health}}' omnibase-infra-runtime-main | jq .
```

### Response Format

#### Healthy Response (HTTP 200)

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "details": {
    "healthy": true,
    "degraded": false,
    "runtime_active": true,
    "handlers": {
      "registered": 5,
      "active": 5
    }
  }
}
```

#### Degraded Response (HTTP 200)

```json
{
  "status": "degraded",
  "version": "0.1.0",
  "details": {
    "healthy": false,
    "degraded": true,
    "runtime_active": true,
    "handlers": {
      "registered": 5,
      "active": 3,
      "failed": ["handler_x", "handler_y"]
    }
  }
}
```

#### Unhealthy Response (HTTP 503)

```json
{
  "status": "unhealthy",
  "version": "0.1.0",
  "error": "Runtime not initialized",
  "correlation_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

### Health Check Configuration

Default health check timing in docker-compose:

| Parameter      | Value  | Description                                    |
|----------------|--------|------------------------------------------------|
| `interval`     | 30s    | Time between health checks                     |
| `timeout`      | 10s    | Maximum time for health check to complete      |
| `start_period` | 40s    | Grace period before checks start               |
| `retries`      | 3      | Consecutive failures before unhealthy          |

## Performance Optimization

### Build Optimization

The Dockerfile uses several techniques for efficient builds:

| Technique              | Benefit                                      | Implementation                        |
|------------------------|----------------------------------------------|---------------------------------------|
| Multi-stage build      | ~60% smaller final image                     | Separate builder and runtime stages   |
| BuildKit cache mounts  | Persistent caches across builds              | `--mount=type=cache` for apt, pip     |
| Layer ordering         | Dependencies cached across code changes      | pyproject.toml copied before source   |
| Minimal runtime        | Only essential packages in final image       | No build tools in runtime stage       |
| .dockerignore          | Reduced context transfer time                | Tests, docs, IDE files excluded       |

### Expected Build Times

| Scenario                      | Expected Time    | Notes                           |
|-------------------------------|------------------|---------------------------------|
| First build (cold cache)      | 3-5 minutes      | Downloads all dependencies      |
| Code change (cached deps)     | 30-60 seconds    | Only rebuilds source layer      |
| No changes (full cache)       | 5-10 seconds     | Cache hit on all layers         |
| With GitHub token             | +10-20 seconds   | Private repo authentication     |

### Build Commands

```bash
# Standard build (with BuildKit - enabled by default in Docker 23.0+)
docker compose -f docker-compose.runtime.yml build

# Force rebuild without cache
docker compose -f docker-compose.runtime.yml build --no-cache

# Build with version labels
docker build \
  --build-arg RUNTIME_VERSION=$(git describe --tags --always) \
  --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -f docker/Dockerfile.runtime \
  -t omnibase-infra-runtime:$(git describe --tags --always) .

# Build with GitHub token for private repos
GITHUB_TOKEN=$(cat ~/.github_token) docker compose -f docker-compose.runtime.yml build
```

### Resource Configuration

Default resource limits in docker-compose:

| Service         | CPU Limit | Memory Limit | CPU Reserve | Memory Reserve |
|-----------------|-----------|--------------|-------------|----------------|
| runtime-main    | 1.0 cores | 512 MB       | 0.25 cores  | 128 MB         |
| runtime-effects | 1.0 cores | 512 MB       | 0.25 cores  | 128 MB         |
| runtime-worker  | 0.5 cores | 256 MB       | 0.1 cores   | 64 MB          |

#### Adjusting Resources

```bash
# Monitor current resource usage
docker stats

# Override resources via environment (if supported)
# Or modify docker-compose.runtime.yml deploy section
```

## Environment Variables

### Required Variables (No Defaults - Fail Fast)

These variables must be set explicitly. The runtime will fail to start if they are missing or contain placeholder values.

| Variable           | Description                       | Security Level |
|--------------------|-----------------------------------|----------------|
| `POSTGRES_PASSWORD`| PostgreSQL database password      | Secret         |
| `VAULT_TOKEN`      | HashiCorp Vault access token      | Secret         |
| `REDIS_PASSWORD`   | Redis/Valkey cache password       | Secret         |

### Optional Variables (With Defaults)

| Variable                     | Default                            | Description                            |
|------------------------------|------------------------------------|----------------------------------------|
| **Network Configuration**    |                                    |                                        |
| `REMOTE_HOST`                | `host-gateway`                     | Override for extra_hosts mapping       |
| **Kafka/Redpanda**           |                                    |                                        |
| `KAFKA_BOOTSTRAP_SERVERS`    | `omninode-bridge-redpanda:9092`    | Kafka/Redpanda broker addresses        |
| **PostgreSQL**               |                                    |                                        |
| `POSTGRES_HOST`              | `omninode-bridge-postgres`         | PostgreSQL hostname                    |
| `POSTGRES_PORT`              | `5432`                             | PostgreSQL port                        |
| `POSTGRES_DATABASE`          | `omninode_bridge`                  | Database name                          |
| `POSTGRES_USER`              | `postgres`                         | Database user                          |
| **Consul**                   |                                    |                                        |
| `CONSUL_HOST`                | `omninode-bridge-consul`           | Consul agent hostname                  |
| `CONSUL_PORT`                | `8500`                             | Consul HTTP port                       |
| `CONSUL_SCHEME`              | `http`                             | Consul connection scheme               |
| **Vault**                    |                                    |                                        |
| `VAULT_ADDR`                 | `http://omninode-bridge-vault:8200`| Vault server address                   |
| **Redis**                    |                                    |                                        |
| `REDIS_HOST`                 | `omninode-bridge-redis`            | Redis hostname                         |
| `REDIS_PORT`                 | `6379`                             | Redis port                             |
| **ONEX Runtime**             |                                    |                                        |
| `ONEX_LOG_LEVEL`             | `INFO`                             | Logging level (DEBUG, INFO, etc.)      |
| `ONEX_ENVIRONMENT`           | `development`                      | Environment name                       |
| `CONTRACTS_DIR`              | `/app/contracts`                   | Path to contract files                 |
| **Service Ports**            |                                    |                                        |
| `RUNTIME_MAIN_PORT`          | `8085`                             | Main runtime exposed port              |
| `RUNTIME_EFFECTS_PORT`       | `8086`                             | Effects runtime exposed port           |
| **Topics**                   |                                    |                                        |
| `ONEX_INPUT_TOPIC`           | `requests`                         | Input topic for main runtime           |
| `ONEX_OUTPUT_TOPIC`          | `responses`                        | Output topic for main runtime          |
| `ONEX_GROUP_ID`              | `onex-runtime-main`                | Consumer group for main runtime        |
| **OpenTelemetry**            |                                    |                                        |
| `OTEL_EXPORTER_OTLP_ENDPOINT`| `http://localhost:4317`            | OTLP exporter endpoint                 |
| `OTEL_SERVICE_NAME`          | `omnibase-infra-runtime`           | Service name for tracing               |

## Volume Management

### Named Volumes

| Volume                          | Purpose                              | Service           |
|---------------------------------|--------------------------------------|-------------------|
| `omnibase-infra-runtime-logs`   | Main runtime logs                    | runtime-main      |
| `omnibase-infra-runtime-data`   | Main runtime persistent state        | runtime-main      |
| `omnibase-infra-effects-logs`   | Effects runtime logs                 | runtime-effects   |
| `omnibase-infra-effects-data`   | Effects runtime persistent state     | runtime-effects   |
| `omnibase-infra-worker-logs`    | Worker logs (shared across replicas) | runtime-worker    |

### Volume Commands

```bash
# List volumes
docker volume ls | grep omnibase-infra

# Inspect volume
docker volume inspect omnibase-infra-runtime-logs

# View volume contents
docker run --rm -v omnibase-infra-runtime-logs:/logs alpine ls -la /logs

# Remove all volumes (WARNING: destroys data)
docker compose -f docker-compose.runtime.yml down -v
```

### Bind Mounts

| Host Path        | Container Path     | Mode      | Purpose                    |
|------------------|--------------------|-----------|----------------------------|
| `../contracts`   | `/app/contracts`   | Read-only | Contract configuration     |

## Troubleshooting

### Container Won't Start

```bash
# Check container status
docker compose -f docker-compose.runtime.yml ps -a

# View startup logs
docker compose -f docker-compose.runtime.yml logs runtime-main

# View last N lines
docker compose -f docker-compose.runtime.yml logs --tail 50 runtime-main

# Follow logs in real-time
docker compose -f docker-compose.runtime.yml logs -f runtime-main
```

#### Common Startup Issues

| Issue                               | Cause                                         | Solution                                      |
|-------------------------------------|-----------------------------------------------|-----------------------------------------------|
| "Password authentication failed"   | Missing/wrong `POSTGRES_PASSWORD`             | Check .env file, verify password              |
| "Address already in use"           | Port 8085 already bound                       | Change `RUNTIME_MAIN_PORT` or free port       |
| "Connection refused"               | Infrastructure services not running           | Start omninode-bridge services first          |
| "No such file or directory"        | Missing contracts directory                   | Create `contracts/` or adjust `CONTRACTS_DIR` |
| "Permission denied"                | Volume permission issues                      | Check UID 1000 has write access               |

### Health Check Failing

```bash
# Check health endpoint manually
curl -v http://localhost:8085/health

# Check from inside container
docker exec omnibase-infra-runtime-main curl -s http://localhost:8085/health

# Check container health history
docker inspect --format='{{json .State.Health}}' omnibase-infra-runtime-main | jq .

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' omnibase-infra-runtime-main
```

### Build Failures

```bash
# Rebuild without cache
docker compose -f docker-compose.runtime.yml build --no-cache

# Build with verbose output
docker compose -f docker-compose.runtime.yml build --progress=plain

# Check Dockerfile syntax
docker build --check -f docker/Dockerfile.runtime .

# Build with GitHub token for private repos
GITHUB_TOKEN=$(cat ~/.github_token) docker compose -f docker-compose.runtime.yml build
```

#### Common Build Issues

| Issue                              | Cause                                  | Solution                                      |
|------------------------------------|----------------------------------------|-----------------------------------------------|
| "Could not find omnibase-core"    | Private repo access denied             | Set GITHUB_TOKEN with repo access             |
| "No space left on device"          | Docker disk full                       | `docker system prune -a`                      |
| "Network timeout"                  | Slow network during pip install        | Retry or use mirror                           |
| "Failed to fetch"                  | apt repository issues                  | Update base image or retry                    |

### Connectivity Issues

```bash
# Test connectivity to PostgreSQL
docker exec omnibase-infra-runtime-main \
  curl -v telnet://omninode-bridge-postgres:5432

# Test connectivity to Kafka
docker exec omnibase-infra-runtime-main \
  curl -v telnet://omninode-bridge-redpanda:9092

# Check DNS resolution
docker exec omnibase-infra-runtime-main \
  getent hosts omninode-bridge-postgres

# Check network configuration
docker network inspect omnibase-infra-network
```

### Debug Mode

```bash
# Start with debug logging
ONEX_LOG_LEVEL=DEBUG docker compose -f docker-compose.runtime.yml up

# Shell into running container
docker exec -it omnibase-infra-runtime-main /bin/bash

# Shell into new container for debugging
docker run -it --rm \
  --network omnibase-infra-network \
  omnibase-infra-runtime /bin/bash
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Push Runtime

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}/runtime
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha,prefix=

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.runtime
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          build-args: |
            RUNTIME_VERSION=${{ github.ref_name }}
            BUILD_DATE=${{ github.event.head_commit.timestamp }}
            VCS_REF=${{ github.sha }}
          secrets: |
            github_token=${{ secrets.GITHUB_TOKEN }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### GitLab CI Example

```yaml
build-runtime:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  variables:
    DOCKER_BUILDKIT: "1"
  script:
    - docker build
        --build-arg RUNTIME_VERSION=${CI_COMMIT_TAG:-$CI_COMMIT_SHORT_SHA}
        --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        --build-arg VCS_REF=$CI_COMMIT_SHA
        --secret id=github_token,env=GITHUB_TOKEN
        -f docker/Dockerfile.runtime
        -t $CI_REGISTRY_IMAGE/runtime:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE/runtime:$CI_COMMIT_SHA
  only:
    - main
    - tags
```

### Health Check Integration for CI

```yaml
# Add to your CI pipeline after deployment
- name: Wait for healthy
  run: |
    for i in {1..30}; do
      if curl -sf http://localhost:8085/health; then
        echo "Runtime is healthy"
        exit 0
      fi
      echo "Waiting for runtime to be healthy... (attempt $i/30)"
      sleep 5
    done
    echo "Runtime failed to become healthy"
    docker compose -f docker-compose.runtime.yml logs
    exit 1
```

## Development Workflow

### Local Development Cycle

```bash
# 1. Make code changes in src/

# 2. Rebuild and restart (picks up code changes)
docker compose -f docker-compose.runtime.yml up -d --build

# 3. View logs
docker compose -f docker-compose.runtime.yml logs -f

# 4. Test changes
curl http://localhost:8085/health

# 5. Stop when done
docker compose -f docker-compose.runtime.yml down
```

### Running Tests in Container

```bash
# Run tests with the same environment as production
docker run --rm \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  omnibase-infra-runtime \
  python -m pytest tests/ -v
```

### Inspecting Container State

```bash
# View running processes
docker exec omnibase-infra-runtime-main ps aux

# Check Python environment
docker exec omnibase-infra-runtime-main python --version
docker exec omnibase-infra-runtime-main pip list

# View environment variables (careful: may show secrets)
docker exec omnibase-infra-runtime-main env | grep -v PASSWORD | grep -v TOKEN

# Check disk usage
docker exec omnibase-infra-runtime-main df -h

# View memory usage
docker exec omnibase-infra-runtime-main free -h
```

## File Reference

| File                        | Purpose                                            |
|-----------------------------|----------------------------------------------------|
| `Dockerfile.runtime`        | Multi-stage Dockerfile for runtime image           |
| `docker-compose.runtime.yml`| Service orchestration with profiles                |
| `.env.example`              | Environment variable template (copy to `.env`)     |
| `README.md`                 | This documentation                                 |

## Related Documentation

- [ONEX Runtime Kernel](../src/omnibase_infra/runtime/kernel.py) - Core kernel implementation
- [Health Server](../src/omnibase_infra/runtime/health_server.py) - HTTP health endpoint
- [Main README](../README.md) - Project overview and setup

## License

MIT License - See [LICENSE](../LICENSE) file for details.
