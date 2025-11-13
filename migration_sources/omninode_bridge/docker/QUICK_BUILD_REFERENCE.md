# Quick Docker Build Reference

## TL;DR - Just Build It

```bash
# Deployment Dockerfiles (recommended for most use cases)
docker build -f deployment/Dockerfile.orchestrator -t orchestrator:latest .
docker build -f deployment/Dockerfile.reducer -t reducer:latest .

# Or use docker-compose (builds both + infrastructure)
docker compose -f deployment/docker-compose.bridge.yml up --build
```

**Works with or without poetry.lock** - builds handle it automatically! ✅

---

## Common Build Commands

### Development Builds
```bash
# With hot-reload and debug logging
docker build -f deployment/Dockerfile.orchestrator --target development -t orchestrator:dev .
docker build -f deployment/Dockerfile.reducer --target development -t reducer:dev .
```

### Production Builds
```bash
# Optimized, multi-worker, minimal dependencies
docker build -f deployment/Dockerfile.orchestrator --target production -t orchestrator:prod .
docker build -f deployment/Dockerfile.reducer --target production -t reducer:prod .
```

### Minimal Production Images
```bash
# Smallest images with security hardening
docker build -f docker/bridge-nodes/Dockerfile.orchestrator -t orchestrator:slim .
docker build -f docker/bridge-nodes/Dockerfile.reducer -t reducer:slim .
```

---

## Docker Compose Commands

### Start Everything
```bash
# Start with build
docker compose -f deployment/docker-compose.bridge.yml up --build

# Background mode
docker compose -f deployment/docker-compose.bridge.yml up -d

# Scale services
docker compose -f deployment/docker-compose.bridge.yml up --scale orchestrator=3 --scale reducer=2
```

### Stop & Clean
```bash
# Stop services
docker compose -f deployment/docker-compose.bridge.yml down

# Remove volumes too
docker compose -f deployment/docker-compose.bridge.yml down -v
```

---

## Troubleshooting

### Build Fails?

**Quick Fix:**
```bash
# Generate poetry.lock first
poetry lock --no-update

# Then build with no cache
docker build --no-cache -f deployment/Dockerfile.orchestrator -t orchestrator:latest .
```

### Slow Build?

**Solution:** Commit poetry.lock to speed up future builds
```bash
git add poetry.lock
git commit -m "Add poetry.lock for faster builds"
```

### Different Versions?

**Fix:**
```bash
# Update lock file
poetry lock --no-update

# Rebuild without cache
docker build --no-cache -f deployment/Dockerfile.orchestrator -t orchestrator:latest .
```

---

## Build Targets Explained

| Target | Use Case | Size | Speed |
|--------|----------|------|-------|
| `development` | Local dev with hot-reload | ~1.5GB | Slow first build |
| `production` | Production deployment | ~800MB | Medium build |
| Slim (bridge-nodes) | Minimal production | ~400MB | Fast multi-stage |

---

## Environment Variables

```bash
# Change build target in docker-compose
BUILD_TARGET=production docker compose -f deployment/docker-compose.bridge.yml up --build

# Custom Poetry version
docker build --build-arg POETRY_VERSION=1.8.0 -f deployment/Dockerfile.orchestrator .

# Custom Python version
docker build --build-arg PYTHON_VERSION=3.11 -f deployment/Dockerfile.orchestrator .
```

---

## CI/CD Quick Start

### GitHub Actions
```yaml
- name: Build Images
  run: |
    docker build -f deployment/Dockerfile.orchestrator -t orchestrator:${{ github.sha }} .
    docker build -f deployment/Dockerfile.reducer -t reducer:${{ github.sha }} .
```

### GitLab CI
```yaml
build:
  script:
    - docker build -f deployment/Dockerfile.orchestrator -t orchestrator:$CI_COMMIT_SHA .
    - docker build -f deployment/Dockerfile.reducer -t reducer:$CI_COMMIT_SHA .
```

---

## Key Features

✅ **Auto-handles missing poetry.lock** - generates if needed
✅ **Multi-stage builds** - optimized layer caching
✅ **Security hardened** - non-root user, minimal dependencies
✅ **Health checks included** - ready for orchestration
✅ **Development & production** - single Dockerfile for both

---

## Need More Help?

- **Full Guide**: [docker/DOCKER_BUILD_GUIDE.md](DOCKER_BUILD_GUIDE.md)
- **Fix Details**: [DOCKER_BUILD_FIX_SUMMARY.md](../DOCKER_BUILD_FIX_SUMMARY.md)
- **Architecture**: [docs/BRIDGE_NODES_GUIDE.md](../docs/BRIDGE_NODES_GUIDE.md)

---

**Last Updated:** October 2025
**Tested With:** Docker 27.x, Poetry 1.7.1, Python 3.12
