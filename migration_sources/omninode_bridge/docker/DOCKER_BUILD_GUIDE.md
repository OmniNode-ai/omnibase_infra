# Docker Build Guide - OmniNode Bridge

## Overview

This guide explains the Docker build system for the OmniNode Bridge services, including handling of Poetry dependencies and multi-stage builds.

## Poetry Lock File Handling

All Dockerfiles in this project are designed to handle scenarios where `poetry.lock` might not be present:

### How It Works

1. **Optional Copy**: Uses wildcard pattern to copy poetry.lock without failing if missing
   ```dockerfile
   COPY poetry.loc[k] ./poetry.lock* ./
   ```

2. **Automatic Generation**: If poetry.lock doesn't exist, it's generated from pyproject.toml
   ```dockerfile
   RUN if [ ! -f poetry.lock ]; then \
           echo "poetry.lock not found, generating from pyproject.toml..."; \
           poetry lock --no-update; \
       fi && \
       poetry install --only=main
   ```

### Build Scenarios

#### Scenario 1: With Existing poetry.lock (Recommended)
```bash
# Fastest build - uses pinned dependencies
docker build -f deployment/Dockerfile.orchestrator -t orchestrator:latest .
```

**Advantages:**
- Faster build (no lock generation)
- Reproducible builds with pinned versions
- Recommended for production

#### Scenario 2: Without poetry.lock
```bash
# Works automatically - generates lock file during build
docker build -f deployment/Dockerfile.orchestrator -t orchestrator:latest .
```

**Advantages:**
- Works in fresh clones
- Works in CI/CD without committed lock file
- Automatically resolves latest compatible versions

## Available Dockerfiles

### Root Level (Multi-Stage Development + Production)

#### Dockerfile.orchestrator
- **Location**: `/deployment/Dockerfile.orchestrator`
- **Stages**: `base` → `development` → `production`
- **Development**: Full dev dependencies, hot reload, debug logging
- **Production**: Minimal dependencies, multi-worker, optimized

```bash
# Development build
docker build -f deployment/Dockerfile.orchestrator --target development -t orchestrator:dev .

# Production build
docker build -f deployment/Dockerfile.orchestrator --target production -t orchestrator:prod .
```

#### Dockerfile.reducer
- **Location**: `/deployment/Dockerfile.reducer`
- **Stages**: `base` → `development` → `production`
- **Development**: Full dev dependencies, hot reload, debug logging
- **Production**: Minimal dependencies, multi-worker, optimized

```bash
# Development build
docker build -f deployment/Dockerfile.reducer --target development -t reducer:dev .

# Production build
docker build -f deployment/Dockerfile.reducer --target production -t reducer:prod .
```

### docker/bridge-nodes (Production-Only Minimal Images)

#### Dockerfile.orchestrator
- **Location**: `/docker/bridge-nodes/Dockerfile.orchestrator`
- **Type**: Multi-stage builder → runtime
- **Optimizations**: Smaller image, security hardening, non-root user

```bash
# Production build (from project root)
docker build -f docker/bridge-nodes/Dockerfile.orchestrator -t orchestrator:slim .
```

#### Dockerfile.reducer
- **Location**: `/docker/bridge-nodes/Dockerfile.reducer`
- **Type**: Multi-stage builder → runtime
- **Optimizations**: Smaller image, security hardening, non-root user

```bash
# Production build (from project root)
docker build -f docker/bridge-nodes/Dockerfile.reducer -t reducer:slim .
```

## Docker Compose Usage

### Bridge Nodes Stack

```bash
# Start all services
docker compose -f deployment/docker-compose.bridge.yml up

# Start with rebuild
docker compose -f deployment/docker-compose.bridge.yml up --build

# Start in background
docker compose -f deployment/docker-compose.bridge.yml up -d

# Scale nodes
docker compose -f deployment/docker-compose.bridge.yml up --scale orchestrator=3 --scale reducer=2
```

### Build Arguments

The deployment/docker-compose.bridge.yml file supports build target selection:

```bash
# Development build (default)
docker compose -f deployment/docker-compose.bridge.yml up

# Production build
BUILD_TARGET=production docker compose -f deployment/docker-compose.bridge.yml up --build
```

## Pre-Build Steps (Optional)

### Generate poetry.lock Before Build

If you want to ensure poetry.lock exists before building:

```bash
# Generate lock file
poetry lock --no-update

# Then build
docker build -f deployment/Dockerfile.orchestrator -t orchestrator:latest .
```

### CI/CD Integration

For CI/CD pipelines, you have two options:

#### Option 1: Commit poetry.lock (Recommended)
```bash
# Ensure poetry.lock is in version control
git add poetry.lock
git commit -m "Update dependencies"
```

#### Option 2: Generate During CI
```yaml
# .github/workflows/docker-build.yml
- name: Generate poetry.lock
  run: poetry lock --no-update

- name: Build Docker image
  run: docker build -f deployment/Dockerfile.orchestrator -t orchestrator:latest .
```

## Troubleshooting

### Build Fails at COPY poetry.lock

**Problem**: Old Dockerfile version without automatic generation

**Solution**: Update Dockerfile to use the pattern from this guide:
```dockerfile
COPY pyproject.toml ./
COPY poetry.loc[k] ./poetry.lock* ./
RUN if [ ! -f poetry.lock ]; then poetry lock --no-update; fi
```

### Different Dependencies in Local vs Docker

**Problem**: Local environment has different versions than Docker build

**Solution**: Ensure poetry.lock is up-to-date:
```bash
poetry lock --no-update
docker build --no-cache -f deployment/Dockerfile.orchestrator -t orchestrator:latest .
```

### Build is Slow

**Problem**: Lock file generation during build takes time

**Solution**: Generate and commit poetry.lock:
```bash
poetry lock --no-update
git add poetry.lock
git commit -m "Add poetry.lock for faster Docker builds"
```

## Best Practices

1. **Commit poetry.lock**: Always commit poetry.lock to version control for reproducible builds
2. **Use Layer Caching**: Copy pyproject.toml and poetry.lock before copying source code
3. **Multi-Stage Builds**: Use appropriate build target (development vs production)
4. **Security**: All production images run as non-root user
5. **Health Checks**: Production images include health checks for orchestration

## Image Size Comparison

| Dockerfile | Target | Approximate Size | Use Case |
|------------|--------|-----------------|----------|
| deployment/Dockerfile.orchestrator | development | ~1.5GB | Local development |
| deployment/Dockerfile.orchestrator | production | ~800MB | Production deployment |
| docker/bridge-nodes/Dockerfile.orchestrator | runtime | ~400MB | Minimal production |
| docker/bridge-nodes/Dockerfile.reducer | runtime | ~400MB | Minimal production |

## Security Features

All production images include:
- Non-root user execution
- Minimal system dependencies
- Security-hardened base images
- Health check endpoints
- Resource limits in docker-compose

## Related Documentation

- [Bridge Nodes Guide](../docs/BRIDGE_NODES_GUIDE.md) - Implementation details
- [API Reference](../docs/API_REFERENCE.md) - API documentation
- [docker-compose.bridge.yml](../deployment/docker-compose.bridge.yml) - Service orchestration
