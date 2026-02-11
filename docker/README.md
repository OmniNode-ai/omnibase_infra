# ONEX Infrastructure Runtime - Docker Deployment

This directory contains Docker configuration for deploying the ONEX Infrastructure Runtime. The runtime provides a contract-driven kernel for event-based infrastructure orchestration.

## Table of Contents

- [Prerequisites](#prerequisites)
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

## Prerequisites

Before deploying, ensure you have:

| Requirement        | Minimum Version | Verification Command             |
|--------------------|-----------------|----------------------------------|
| Docker             | 20.10+          | `docker --version`               |
| Docker Compose     | 2.0+ (V2)       | `docker compose version`         |
| BuildKit           | Enabled         | `docker buildx version`          |

### Docker BuildKit

BuildKit is required for:
- Multi-stage build optimization
- Cache mount optimizations

BuildKit is enabled by default in Docker 23.0+. For older versions:

```bash
# Option 1: Enable for current session
export DOCKER_BUILDKIT=1

# Option 2: Enable permanently in Docker daemon config
# Add to /etc/docker/daemon.json: {"features": {"buildkit": true}}
```

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env

# 2. CRITICAL: Edit .env and replace ALL placeholder values
#    Look for patterns: __REPLACE_WITH_*__
#    - POSTGRES_PASSWORD: openssl rand -hex 32
#    - INFISICAL_ENCRYPTION_KEY (secrets profile): openssl rand -hex 32
#    - INFISICAL_AUTH_SECRET (secrets profile): openssl rand -hex 32

# 3. Start core infrastructure (default profile)
docker compose -f docker-compose.infra.yml up -d

# 4. Or start with full runtime services
docker compose -f docker-compose.infra.yml --profile runtime up -d --build

# 5. View logs
docker compose -f docker-compose.infra.yml logs -f

# 6. Verify health (runtime profile only)
curl http://localhost:8085/health
```

## Architecture

The deployment follows a multi-profile architecture designed for flexible scaling and separation of concerns:

```text
                              Docker Compose Profiles
+------------------------------------------------------------------------+
|                                                                        |
|   Default Profile (Infrastructure)                                     |
|   +---------------+  +---------------+  +---------------+              |
|   |   postgres    |  |   redpanda    |  |    valkey     |              |
|   |   (storage)   |  |   (events)    |  |   (cache)     |              |
|   +-------+-------+  +-------+-------+  +-------+-------+              |
|           |                  |                  |                      |
|           +------------------+------------------+                      |
|                              |                                         |
|   +--------------------------------------------------------------+     |
|   |              omnibase-infra-network (bridge)                 |     |
|   +--------------------------------------------------------------+     |
|                              |                                         |
|   Runtime Profile (--profile runtime)                                  |
|   +---------------+  +---------------+  +---------------+              |
|   | omninode-     |  |  runtime-     |  |  runtime-     |              |
|   | runtime       |  |  effects      |  |  worker x N   |              |
|   | (kernel)      |  |  (ext. I/O)   |  |  (parallel)   |              |
|   +---------------+  +---------------+  +---------------+              |
|                                                                        |
|   Optional Profiles:                                                   |
|   +---------------+  +---------------+                                 |
|   |    consul     |  |  infisical    |                                 |
|   |  (discovery)  |  |  (secrets)    |                                 |
|   | --profile     |  | --profile     |                                 |
|   |   consul      |  |   secrets     |                                 |
|   +---------------+  +---------------+                                 |
|                                                                        |
+------------------------------------------------------------------------+
```

### Service Communication Flow

```text
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
| - Infisical      |     | - Aggregate    |
| - Kafka          |     +----------------+
+------------------+
```

## Profiles

| Profile    | Services                                      | Use Case                                   | Command                                                          |
|------------|-----------------------------------------------|--------------------------------------------|------------------------------------------------------------------|
| (default)  | postgres, redpanda, valkey, topic-manager     | Core infrastructure only                   | `docker compose -f docker-compose.infra.yml up -d`               |
| `runtime`  | (default) + runtime, effects, workers, consumer | Full ONEX runtime with observability     | `docker compose -f docker-compose.infra.yml --profile runtime up -d` |
| `consul`   | (default) + consul                            | Infrastructure + service discovery         | `docker compose -f docker-compose.infra.yml --profile consul up -d` |
| `secrets`  | (default) + infisical                         | Infrastructure + secrets management        | `docker compose -f docker-compose.infra.yml --profile secrets up -d` |
| `full`     | All services                                  | Complete stack with all features           | `docker compose -f docker-compose.infra.yml --profile full up -d` |

### Profile Details

#### Default Profile (Core Infrastructure)

Core infrastructure services required by all deployments:

- **PostgreSQL** (port 5436): Primary data store for projections and state
- **Redpanda** (port 29092): Kafka-compatible event streaming
- **Valkey** (port 16379): Redis-compatible cache and pub/sub
- **Topic Manager**: Automatic Kafka topic creation on startup

```bash
docker compose -f docker-compose.infra.yml up -d
```

#### Runtime Profile (ONEX Services)

Full ONEX runtime services with observability:

- **omninode-runtime** (port 8085): Core kernel handling requests
- **runtime-effects** (port 8086): External service interactions
- **runtime-worker** (x2): Scalable parallel compute workers
- **agent-actions-consumer** (port 8087): Observability event persistence

```bash
docker compose -f docker-compose.infra.yml --profile runtime up -d --build

# Scale workers manually
docker compose -f docker-compose.infra.yml --profile runtime up -d --scale runtime-worker=4

# Or set via environment variable
WORKER_REPLICAS=8 docker compose -f docker-compose.infra.yml --profile runtime up -d
```

#### Consul Profile (Service Discovery)

Adds HashiCorp Consul for service registration and discovery:

- **Consul** (port 28500): Service discovery UI and API

```bash
docker compose -f docker-compose.infra.yml --profile consul up -d
```

#### Secrets Profile (Secrets Management)

Adds Infisical for centralized secrets management:

- **Infisical** (port 8880): Secrets management UI and API
- Requires `INFISICAL_ENCRYPTION_KEY` and `INFISICAL_AUTH_SECRET` in `.env`

```bash
docker compose -f docker-compose.infra.yml --profile secrets up -d
```

#### Full Profile (All Services)

Complete stack with infrastructure, runtime, consul, and secrets:

```bash
docker compose -f docker-compose.infra.yml --profile full up -d --build
```

## Security

### Credential Management

**CRITICAL**: All placeholder values (`__REPLACE_WITH_*__`) in `.env.example` MUST be changed before ANY deployment.

```bash
# Generate secure random passwords
openssl rand -hex 32

# Generate multiple passwords at once
for i in {1..3}; do echo "Password $i: $(openssl rand -hex 32)"; done
```

### Required Credentials

| Credential              | Environment Variable       | Profile  | Notes                                                    |
|-------------------------|----------------------------|----------|----------------------------------------------------------|
| PostgreSQL              | `POSTGRES_PASSWORD`        | (default)| Database access password                                 |
| Infisical Encryption    | `INFISICAL_ENCRYPTION_KEY` | secrets  | Hex-encoded key: 16-byte (`openssl rand -hex 16`) or 32-byte (`openssl rand -hex 32`) |
| Infisical Auth          | `INFISICAL_AUTH_SECRET`    | secrets  | JWT signing secret (`openssl rand -hex 32`)              |
| Valkey (optional)       | `VALKEY_PASSWORD`          | (default)| Cache password (defaults to `valkey-dev-password` in dev) |

### Security Features

| Feature                  | Description                                          | Verification                              |
|--------------------------|------------------------------------------------------|-------------------------------------------|
| Non-root execution       | Container runs as `omniinfra` user (UID 1000)        | `docker exec <container> whoami`          |
| BuildKit secrets         | Sensitive values passed via secret mounts             | `docker history --no-trunc`               |
| No hardcoded credentials | All secrets via environment variables                | Review Dockerfile                         |
| Resource limits          | CPU and memory limits prevent exhaustion             | `docker stats`                            |
| Read-only contracts      | Contract volume mounted read-only                    | `:ro` in docker-compose                   |

### Security Verification

```bash
# Verify non-root user
docker exec omninode-runtime whoami
# Expected output: omniinfra

# Verify UID/GID
docker exec omninode-runtime id
# Expected: uid=1000(omniinfra) gid=1000(omniinfra) groups=1000(omniinfra)

# Verify secrets not in image history
docker history omninode-runtime --no-trunc | grep -iE "token|password|secret"
# Expected: no output (empty result)

# Verify no secrets in environment inspection
docker inspect omninode-runtime --format='{{json .Config.Env}}' | grep -i password
# Should only show placeholder references, not actual values
```

### Security Best Practices

1. **Never commit `.env` files** - Already in `.gitignore`
2. **Rotate credentials regularly** - Use Infisical or secret manager
3. **Use minimal permissions** - Infisical tokens should have scoped policies
4. **Enable audit logging** - Track credential access
5. **Network isolation** - Use internal Docker networks

### Network Security Considerations

#### Port Exposure

By default, ports are bound to all interfaces (`0.0.0.0`). For production deployments:

```yaml
# Default (accessible from any network interface):
ports:
  - "8085:8085"

# Production (localhost only, use with reverse proxy):
ports:
  - "127.0.0.1:8085:8085"
```

#### Network Isolation Guidelines

1. **Use internal networks** - The `omnibase-infra-network` is bridge-mode by default
2. **Reverse proxy** - In production, place containers behind nginx/traefik with TLS termination
3. **Firewall rules** - Restrict access to Docker ports at the host firewall level
4. **Docker socket** - Never expose Docker socket to containers

#### Environment-Specific Binding

For non-Docker deployments (local development), configure bind host:

```bash
# Local development - bind to localhost only
export RUNTIME_BIND_HOST=127.0.0.1

# Container deployment - bind to all interfaces (for Docker networking)
export RUNTIME_BIND_HOST=0.0.0.0
```

## Health Checks

The runtime exposes HTTP health endpoints for container orchestration platforms.

### Understanding Liveness vs Readiness

Container orchestration platforms distinguish between two types of health checks:

| Probe Type  | Question Answered                      | On Failure Action           |
|-------------|----------------------------------------|-----------------------------|
| **Liveness**  | "Is the container alive?"              | Restart the container       |
| **Readiness** | "Should traffic be routed here?"       | Stop routing traffic        |

**Liveness Probe** (`/health`):
- Checks if the runtime process is responsive
- Returns 200 if the server can respond to requests
- Failure triggers container restart (Docker: via restart policy)

**Readiness Probe** (`/ready`):
- Checks if all dependencies are connected (database, message bus, etc.)
- Returns 200 only when ready to handle production traffic
- Failure removes container from load balancer rotation (no restart)

**In Docker Compose**: The health check serves dual purpose - it determines both container
health status AND the `service_healthy` condition used in `depends_on`.

**For Kubernetes**: Consider using separate probes:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8085
readinessProbe:
  httpGet:
    path: /ready
    port: 8085
```

### Endpoints

| Endpoint  | Method | Description                                         | Success | Failure |
|-----------|--------|-----------------------------------------------------|---------|---------|
| `/health` | GET    | Liveness probe - is the process alive?              | 200     | 503     |
| `/ready`  | GET    | Readiness probe - are dependencies ready?           | 200     | 503     |

### Health Check Commands

```bash
# Test health endpoint
curl -s http://localhost:8085/health | jq .

# Test from inside container network
docker exec omninode-runtime curl -s http://localhost:8085/health

# Check container health status
docker inspect --format='{{.State.Health.Status}}' omninode-runtime

# View health check history
docker inspect --format='{{json .State.Health}}' omninode-runtime | jq .
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

### Build Commands

```bash
# Standard build (with BuildKit - enabled by default in Docker 23.0+)
docker compose -f docker-compose.infra.yml build

# Force rebuild without cache
docker compose -f docker-compose.infra.yml build --no-cache

# Build with version labels
docker build \
  --build-arg RUNTIME_VERSION=$(git describe --tags --always) \
  --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -f docker/Dockerfile.runtime \
  -t omninode-runtime:$(git describe --tags --always) .
```

### Resource Configuration

Default resource limits in docker-compose:

| Service         | CPU Limit | Memory Limit | CPU Reserve | Memory Reserve |
|-----------------|-----------|--------------|-------------|----------------|
| omninode-runtime| 1.0 cores | 512 MB       | 0.25 cores  | 128 MB         |
| runtime-effects | 1.0 cores | 512 MB       | 0.25 cores  | 128 MB         |
| runtime-worker  | 0.5 cores | 256 MB       | 0.1 cores   | 64 MB          |

#### Adjusting Resources

```bash
# Monitor current resource usage
docker stats

# Override resources via environment (if supported)
# Or modify docker-compose.infra.yml deploy section
```

## Environment Variables

### Required Variables (No Defaults - Fail Fast)

These variables must be set explicitly. The runtime will fail to start if they are missing or contain placeholder values.

| Variable                  | Description                                         | Security Level | Profile    |
|---------------------------|-----------------------------------------------------|----------------|------------|
| `POSTGRES_PASSWORD`       | PostgreSQL database password                        | Secret         | (default)  |
| `OMNIBASE_INFRA_DB_URL`  | Full PostgreSQL DSN for omnibase_infra. Required for CLI/scripts. For Docker-only usage, `POSTGRES_PASSWORD` alone suffices (the fallback constructs the DSN automatically). Set this explicitly if your password contains special characters (`@`, `:`, `/`, `%`, `#`) since the Docker fallback cannot URL-encode. | Secret | (default) |
| `INFISICAL_ENCRYPTION_KEY`| Hex-encoded AES key (32 or 64 hex chars)            | Secret         | secrets    |
| `INFISICAL_AUTH_SECRET`   | JWT signing secret for authentication               | Secret         | secrets    |

### Optional Variables (With Defaults)

| Variable                     | Default                            | Description                            |
|------------------------------|------------------------------------|----------------------------------------|
| **Kafka/Redpanda**           |                                    |                                        |
| `KAFKA_BOOTSTRAP_SERVERS`    | `localhost:9092`                   | Kafka/Redpanda broker addresses        |
| **PostgreSQL**               |                                    |                                        |
| `POSTGRES_HOST`              | `localhost`                        | PostgreSQL hostname (legacy fallback)  |
| `POSTGRES_PORT`              | `5432`                             | PostgreSQL port (legacy fallback)      |
| `POSTGRES_USER`              | `postgres`                         | Database user (legacy fallback)        |
| **Consul**                   |                                    |                                        |
| `CONSUL_HOST`                | `localhost`                        | Consul agent hostname                  |
| `CONSUL_PORT`                | `8500`                             | Consul HTTP port                       |
| `CONSUL_SCHEME`              | `http`                             | Consul connection scheme               |
| **Infisical**                |                                    |                                        |
| `INFISICAL_ADDR`             | `http://infisical:8080`            | Infisical service address              |
| `INFISICAL_SITE_URL`         | `http://localhost:8880`            | Infisical web UI URL                   |
| `INFISICAL_REDIS_URL`        | `redis://:valkey-dev-password@valkey:6379` | Redis URL for Infisical cache (must match `VALKEY_PASSWORD`) |
| **Valkey**                   |                                    |                                        |
| `VALKEY_HOST`                | `valkey`                           | Valkey hostname                        |
| `VALKEY_PORT`                | `6379`                             | Valkey port                            |
| `VALKEY_PASSWORD`            | `valkey-dev-password`              | Cache auth password (optional for local dev) |
| **ONEX Runtime**             |                                    |                                        |
| `ONEX_LOG_LEVEL`             | `INFO`                             | Logging level (DEBUG, INFO, etc.)      |
| `ONEX_ENVIRONMENT`           | `development`                      | Environment name                       |
| `ONEX_CONTRACTS_DIR`         | `/app/contracts`                   | Path to contract files                 |
| **Service Ports**            |                                    |                                        |
| `RUNTIME_MAIN_PORT`          | `8085`                             | Main runtime exposed port              |
| `RUNTIME_EFFECTS_PORT`       | `8086`                             | Effects runtime exposed port           |
| **Topics**                   |                                    |                                        |
| `ONEX_INPUT_TOPIC`           | `requests`                         | Input topic for main runtime           |
| `ONEX_OUTPUT_TOPIC`          | `responses`                        | Output topic for main runtime          |
| `ONEX_GROUP_ID`              | `onex-runtime-main`                | Consumer group for main runtime        |
| **OpenTelemetry**            |                                    |                                        |
| `OTEL_EXPORTER_OTLP_ENDPOINT`| `http://localhost:4317`            | OTLP exporter endpoint                 |
| `OTEL_SERVICE_NAME`          | `omninode-runtime`                 | Service name for tracing               |

## Volume Management

### Named Volumes

| Volume                          | Purpose                              | Service                |
|---------------------------------|--------------------------------------|------------------------|
| `omnibase-infra-postgres-data`  | PostgreSQL data                      | postgres               |
| `omnibase-infra-redpanda-data`  | Redpanda event streaming data        | redpanda               |
| `omnibase-infra-valkey-data`    | Valkey cache data                    | valkey                 |
| `omnibase-infra-consul-data`    | Consul service discovery data        | consul                 |
| `omninode-runtime-logs`         | Main runtime logs                    | omninode-runtime       |
| `omninode-runtime-data`         | Main runtime persistent state        | omninode-runtime       |
| `omninode-effects-logs`         | Effects runtime logs                 | runtime-effects        |
| `omninode-effects-data`         | Effects runtime persistent state     | runtime-effects        |
| `omninode-worker-logs`          | Worker logs (shared across replicas) | runtime-worker         |
| `omninode-agent-actions-logs`   | Agent actions consumer logs          | agent-actions-consumer |

### Volume Commands

```bash
# List volumes
docker volume ls | grep omnibase-infra

# Inspect volume
docker volume inspect omninode-runtime-logs

# View volume contents
docker run --rm -v omninode-runtime-logs:/logs alpine ls -la /logs

# Remove all volumes (WARNING: destroys data)
docker compose -f docker-compose.infra.yml down -v
```

### Bind Mounts

| Host Path        | Container Path     | Mode      | Purpose                    |
|------------------|--------------------|-----------|----------------------------|
| `../contracts`   | `/app/contracts`   | Read-only | Contract configuration     |

## Troubleshooting

### Container Won't Start

```bash
# Check container status
docker compose -f docker-compose.infra.yml ps -a

# View startup logs
docker compose -f docker-compose.infra.yml logs omninode-runtime

# View last N lines
docker compose -f docker-compose.infra.yml logs --tail 50 omninode-runtime

# Follow logs in real-time
docker compose -f docker-compose.infra.yml logs -f omninode-runtime
```

#### Common Startup Issues

| Issue                               | Cause                                         | Solution                                      |
|-------------------------------------|-----------------------------------------------|-----------------------------------------------|
| "Password authentication failed"   | Missing/wrong `POSTGRES_PASSWORD`             | Check .env file, verify password              |
| "Address already in use"           | Port 8085 already bound                       | Change `RUNTIME_MAIN_PORT` or free port       |
| "Connection refused"               | Infrastructure services not running           | Start required infrastructure services first  |
| "No such file or directory"        | Missing contracts directory                   | Create `contracts/` or adjust `ONEX_CONTRACTS_DIR` |
| "Permission denied"                | Volume permission issues                      | Check UID 1000 has write access               |
| "No handlers registered"           | Handler contracts not found or invalid        | See "No Handlers Registered" section below    |

### No Handlers Registered

If you see "No handlers registered" error, follow these troubleshooting steps:

```bash
# 1. Check ONEX_CONTRACTS_DIR is set correctly
docker exec omninode-runtime printenv ONEX_CONTRACTS_DIR

# 2. Verify contracts directory exists and contains handler contracts
docker exec omninode-runtime ls -la /app/contracts
docker exec omninode-runtime find /app/contracts -name "handler_contract.yaml" -o -name "contract.yaml"

# 3. Validate handler contract syntax
docker exec omninode-runtime cat /app/contracts/handlers/*/handler_contract.yaml

# 4. Check for import errors in logs
docker logs omninode-runtime 2>&1 | grep -E "(MODULE_NOT_FOUND|CLASS_NOT_FOUND|IMPORT_ERROR)"
```

**Required handler contract fields:**
- `handler_name`: Unique identifier for the handler
- `handler_class`: Fully-qualified Python class path (e.g., `myapp.handlers.MyHandler`)
- `handler_type`: One of: `effect`, `compute`, `reducer`, `orchestrator`

### Health Check Failing

```bash
# Check health endpoint manually
curl -v http://localhost:8085/health

# Check from inside container
docker exec omninode-runtime curl -s http://localhost:8085/health

# Check container health history
docker inspect --format='{{json .State.Health}}' omninode-runtime | jq .

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' omninode-runtime
```

### Build Failures

```bash
# Rebuild without cache
docker compose -f docker-compose.infra.yml build --no-cache

# Build with verbose output
docker compose -f docker-compose.infra.yml build --progress=plain

# Check Dockerfile syntax
docker build --check -f docker/Dockerfile.runtime .
```

#### Common Build Issues

| Issue                              | Cause                                  | Solution                                      |
|------------------------------------|----------------------------------------|-----------------------------------------------|
| "No space left on device"          | Docker disk full                       | `docker system prune -a`                      |
| "Network timeout"                  | Slow network during pip install        | Retry or use mirror                           |
| "Failed to fetch"                  | apt repository issues                  | Update base image or retry                    |

### Connectivity Issues

Test connectivity from within the runtime container. Replace hostnames with your configured values (defaults shown use localhost, but in Docker networks you may use service names like `postgres`, `redpanda`, etc.):

```bash
# Test connectivity to PostgreSQL (replace hostname with your POSTGRES_HOST value)
docker exec omninode-runtime \
  curl -v telnet://localhost:5432

# Test connectivity to Kafka (replace hostname with your KAFKA_BOOTSTRAP_SERVERS value)
docker exec omninode-runtime \
  curl -v telnet://localhost:9092

# Check DNS resolution (replace hostname with your configured service name)
docker exec omninode-runtime \
  getent hosts localhost

# Check network configuration
docker network inspect omnibase-infra-network
```

### Debug Mode

```bash
# Start with debug logging
ONEX_LOG_LEVEL=DEBUG docker compose -f docker-compose.infra.yml up

# Shell into running container
docker exec -it omninode-runtime /bin/bash

# Shell into new container for debugging
docker run -it --rm \
  --network omnibase-infra-network \
  omninode-runtime /bin/bash
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
    docker compose -f docker-compose.infra.yml logs
    exit 1
```

## Development Workflow

### Local Development Cycle

```bash
# 1. Make code changes in src/

# 2. Rebuild and restart (picks up code changes)
docker compose -f docker-compose.infra.yml up -d --build

# 3. View logs
docker compose -f docker-compose.infra.yml logs -f

# 4. Test changes
curl http://localhost:8085/health

# 5. Stop when done
docker compose -f docker-compose.infra.yml down
```

### Running Tests in Container

```bash
# Run tests with the same environment as production
docker run --rm \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  omninode-runtime \
  python -m pytest tests/ -v
```

### Inspecting Container State

```bash
# View running processes
docker exec omninode-runtime ps aux

# Check Python environment
docker exec omninode-runtime python --version
docker exec omninode-runtime pip list

# View environment variables (careful: may show secrets)
docker exec omninode-runtime env | grep -v PASSWORD | grep -v TOKEN

# Check disk usage
docker exec omninode-runtime df -h

# View memory usage
docker exec omninode-runtime free -h
```

## E2E Testing with Infrastructure

For true end-to-end testing, use `docker-compose.e2e.yml` which includes full infrastructure services (PostgreSQL, Redpanda, Consul) alongside the runtime.

### Quick Start for E2E Testing

```bash
# 1. Navigate to docker directory
cd docker

# 2. Copy E2E environment file
cp .env.e2e .env

# 3. Start infrastructure services only (for running tests from host)
docker compose -f docker-compose.e2e.yml up -d

# 4. Wait for all services to be healthy
docker compose -f docker-compose.e2e.yml ps

# 5. Run E2E tests from host
cd ..
poetry run pytest tests/integration/registration/e2e/ -v

# 6. Tear down when done
docker compose -f docker/docker-compose.e2e.yml down -v
```

### Full Stack Testing (with Runtime in Container)

```bash
# Start infrastructure AND runtime service
docker compose -f docker-compose.e2e.yml --profile runtime up -d

# Verify runtime is healthy
curl http://localhost:8085/health

# Run ALL E2E tests (component tests + runtime tests)
poetry run pytest tests/integration/registration/e2e/ -v

# Run ONLY true runtime E2E tests (requires runtime container)
poetry run pytest tests/integration/registration/e2e/test_runtime_e2e.py -v

# View runtime logs
docker compose -f docker-compose.e2e.yml logs -f runtime
```

### Two Types of E2E Tests

| Test File | What It Tests | Requires Runtime? |
|-----------|---------------|-------------------|
| `test_two_way_registration_e2e.py` | Component integration (handlers, reducers, effects called directly) | No - just infrastructure |
| `test_full_orchestrator_flow.py` | Pipeline components working together | No - just infrastructure |
| `test_runtime_e2e.py` | **True E2E**: Publish to Kafka → Runtime processes → Verify in DB | **Yes** |

The `test_runtime_e2e.py` tests will skip automatically if the runtime container isn't running.

### E2E Service Architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│                    omnibase-infra-network                          │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   postgres   │  │   redpanda   │  │    consul    │             │
│  │    :5432     │  │    :9092     │  │    :8500     │             │
│  │  (test DB)   │  │   (Kafka)    │  │  (discovery) │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│         │                 │                 │                      │
│         └─────────────────┴─────────────────┘                      │
│                           │                                        │
│                  ┌────────┴────────┐                               │
│                  │     runtime     │  (profile: runtime)           │
│                  │     :8085       │                               │
│                  │  onex-runtime   │                               │
│                  └─────────────────┘                               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
        │                   │                   │
   Host:5433           Host:29092          Host:8500
   (postgres)          (kafka)             (consul)
```

### E2E Environment Variables

| Variable              | Default           | Description                    |
|-----------------------|-------------------|--------------------------------|
| `POSTGRES_PASSWORD`   | `test-password`   | PostgreSQL password            |
| `POSTGRES_PORT`       | `5433`            | Host port for PostgreSQL       |
| `KAFKA_PORT`          | `29092`           | Host port for Kafka external   |
| `CONSUL_PORT`         | `8500`            | Host port for Consul           |
| `RUNTIME_PORT`        | `8085`            | Host port for runtime health   |
| `ONEX_LOG_LEVEL`      | `DEBUG`           | Runtime log level              |

### Connecting from Host (for Tests)

When running tests from the host machine (outside Docker):

```python
# PostgreSQL
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5433  # Maps to container 5432

# Kafka (external listener)
KAFKA_BOOTSTRAP_SERVERS = "localhost:29092"

# Consul
CONSUL_HOST = "localhost"
CONSUL_PORT = 8500
```

### Connecting from Runtime Container

When the runtime runs inside Docker:

```python
# PostgreSQL (internal DNS)
POSTGRES_HOST = "postgres"
POSTGRES_PORT = 5432

# Kafka (internal listener)
KAFKA_BOOTSTRAP_SERVERS = "redpanda:9092"

# Consul (internal DNS)
CONSUL_HOST = "consul"
CONSUL_PORT = 8500
```

## File Reference

| File                        | Purpose                                            |
|-----------------------------|----------------------------------------------------|
| `Dockerfile.runtime`        | Multi-stage Dockerfile for runtime image           |
| `docker-compose.infra.yml`  | Infrastructure + runtime orchestration with profiles |
| `docker-compose.e2e.yml`    | E2E testing with infrastructure services           |
| `.env.example`              | Environment variable template (copy to `.env`)     |
| `.env.e2e`                  | E2E testing environment template                   |
| `migrations/forward/`       | Forward migrations (auto-applied on first start)   |
| `migrations/rollback/`      | Rollback scripts (manual execution only)           |
| `README.md`                 | This documentation                                 |

## Related Documentation

- [ONEX Runtime Kernel](../src/omnibase_infra/runtime/kernel.py) - Core kernel implementation
- [Main README](../README.md) - Project overview and setup

## License

MIT License - See [LICENSE](../LICENSE) file for details.
