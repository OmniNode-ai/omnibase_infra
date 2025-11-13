# Bridge Nodes Docker Configuration

Docker setup for Orchestrator and Reducer nodes with multi-stage builds, security hardening, and testing support.

## Quick Start

### Build Images Locally

```bash
# Build orchestrator node
docker build -f docker/bridge-nodes/Dockerfile.orchestrator -t omninode-bridge-orchestrator:local .

# Build reducer node
docker build -f docker/bridge-nodes/Dockerfile.reducer -t omninode-bridge-reducer:local .
```

### Run Test Environment

```bash
# Start all test services (PostgreSQL, Kafka, Orchestrator, Reducer)
cd docker/bridge-nodes
docker-compose -f docker-compose.test.yml up -d

# View logs
docker-compose -f docker-compose.test.yml logs -f

# Stop test environment
docker-compose -f docker-compose.test.yml down -v
```

## Architecture

### Multi-Stage Builds

Both Dockerfiles use multi-stage builds for optimal image size and security:

1. **Builder Stage**: Installs Poetry and all dependencies
2. **Runtime Stage**: Minimal production image with only runtime dependencies

### Security Features

- Non-root user execution (UIDs 1000 and 1001)
- Minimal base image (python:3.12-slim)
- No development dependencies in runtime
- Health checks for container orchestration
- OCI-compliant image labels

### Image Details

| Component | User | UID | Base Image | Ports |
|-----------|------|-----|------------|-------|
| Orchestrator | orchestrator | 1000 | python:3.12-slim | 8080 (configurable) |
| Reducer | reducer | 1001 | python:3.12-slim | 8081 (configurable) |

## CI/CD Integration

### GitHub Actions Workflow

The `docker-build.yml` workflow automatically:

1. **Builds** images for both nodes on every push
2. **Tests** image imports and basic functionality
3. **Scans** for vulnerabilities using Trivy
4. **Analyzes** image sizes for optimization
5. **Pushes** to GitHub Container Registry on main branch

### Build Triggers

- Push to `main`, `develop`, or `feature/node-implementation`
- Changes to node source code or Dockerfiles
- Manual workflow dispatch

### Registry

Images are published to GitHub Container Registry:

```text
ghcr.io/omninode-ai/omninode-bridge-orchestrator:latest
ghcr.io/omninode-ai/omninode-bridge-reducer:latest
```

## Test Environment

### Services

The test environment (`docker-compose.test.yml`) includes:

- **PostgreSQL 15**: Database for state persistence
- **RedPanda**: Kafka-compatible event streaming
- **Orchestrator Node**: Workflow coordination
- **Reducer Node**: State aggregation

### Configuration

All services use test credentials and configurations:

```yaml
Database:
  Host: postgres-test
  Port: 5432
  Database: bridge_test
  User: test
  Password: test-password

Kafka:
  Bootstrap: kafka-test:9092
  Topics:
    - test.workflows.v1
    - test.task-events.v1
```

### Health Checks

All services have health checks configured:

- PostgreSQL: `pg_isready` every 10s
- Kafka: `rpk cluster info` every 30s
- Nodes: Python import check every 30s

## Development

### Local Development with Volume Mounts

The test compose file mounts source code for live development:

```yaml
volumes:
  - ../../src:/app/src
  - ../../logs:/app/logs
```

Changes to source code are immediately reflected without rebuilding.

### Building with Build Args

```bash
docker build \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  --build-arg VERSION=1.0.0 \
  -f docker/bridge-nodes/Dockerfile.orchestrator \
  -t orchestrator:1.0.0 \
  .
```

### Multi-Platform Builds

Both Dockerfiles support multi-platform builds. Currently, CI/CD builds for linux/amd64:

```bash
# Current CI/CD platform
docker buildx build \
  --platform linux/amd64 \
  -f docker/bridge-nodes/Dockerfile.orchestrator \
  -t orchestrator:amd64 \
  --push \
  .

# To build for multiple platforms (optional, not used in CI/CD)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/bridge-nodes/Dockerfile.orchestrator \
  -t orchestrator:multi-arch \
  --push \
  .
```

## Performance

### Image Sizes

Target image sizes (approximate):

- Orchestrator: < 500MB
- Reducer: < 500MB

### Optimization Tips

1. **Layer Caching**: Dependency installation is separate from source code
2. **Production Dependencies**: Only `--only main` dependencies installed
3. **Minimal Base**: Using `slim` variant reduces size by ~300MB vs full Python
4. **No Cache**: `PIP_NO_CACHE_DIR=1` prevents cache bloat

## Troubleshooting

### Import Errors

If you see import errors:

```bash
# Verify PYTHONPATH is set correctly
docker run --rm orchestrator:local python -c "import sys; print(sys.path)"

# Test imports
docker run --rm orchestrator:local python -c "from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeOrchestratorV1"
```

### Permission Issues

If you encounter permission errors:

```bash
# Check user/group
docker run --rm orchestrator:local id

# Should output:
# uid=1000(orchestrator) gid=1000(orchestrator) groups=1000(orchestrator)
```

### Database Connection

If nodes can't connect to PostgreSQL:

```bash
# Check PostgreSQL health
docker-compose -f docker-compose.test.yml exec postgres-test pg_isready -U test

# Test connection from node
docker-compose -f docker-compose.test.yml exec orchestrator python -c "import asyncpg; print('OK')"
```

### Kafka Connection

If nodes can't connect to Kafka:

```bash
# Check Kafka health
docker-compose -f docker-compose.test.yml exec kafka-test rpk cluster info

# List topics
docker-compose -f docker-compose.test.yml exec kafka-test rpk topic list
```

## Security

### Vulnerability Scanning

Images are automatically scanned for vulnerabilities:

```bash
# Manual scan
trivy image orchestrator:local
trivy image reducer:local
```

### Non-Root Execution

Both images run as non-root users:

```bash
# Verify non-root
docker run --rm orchestrator:local whoami
# Output: orchestrator
```

### Secrets Management

Never hardcode secrets in Dockerfiles:

```bash
# Use environment variables
docker run --rm \
  -e POSTGRES_PASSWORD=<secret> \
  orchestrator:local
```

## Next Steps

1. **Production Deployment**: Use Kubernetes manifests or Helm charts
2. **Monitoring**: Add Prometheus exporters for metrics
3. **Logging**: Configure structured logging to stdout/stderr
4. **Tracing**: Integrate OpenTelemetry for distributed tracing

## References

- [Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)
- [Security best practices](https://docs.docker.com/develop/security-best-practices/)
- [OCI image labels](https://github.com/opencontainers/image-spec/blob/main/annotations.md)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
