# CI/CD Pipeline Configuration for Bridge Nodes

## Overview

Comprehensive CI/CD pipeline configuration for automated testing, validation, and deployment of Orchestrator and Reducer bridge nodes.

## Created Components

### GitHub Actions Workflows

#### 1. test-bridge-nodes.yml
**Purpose**: Automated testing suite for bridge nodes

**Triggers**:
- Push to main, develop, feature/node-implementation branches
- Pull requests to main, develop
- Changes to node source code or test files

**Jobs**:
- **unit-tests**: Fast, isolated tests for orchestrator and reducer
  - Matrix strategy: Python 3.12
  - Separate coverage reports for each node
  - 80% coverage target
  - Timeout: 30s per test

- **integration-tests**: End-to-end workflow tests
  - Services: PostgreSQL 15, RedPanda (Kafka)
  - Tests orchestrator-reducer communication flow
  - Environment: Full service stack
  - Timeout: 60s per test

- **performance-tests**: Performance benchmarks
  - pytest-benchmark integration
  - Threshold validation
  - JSON result output for trend analysis

- **coverage-report**: Aggregated coverage analysis
  - Combines unit + integration coverage
  - PR comment with coverage summary
  - Artifact uploads for detailed analysis

- **test-summary**: Overall test status
  - Workflow status summary
  - Links to artifacts
  - Next steps guidance

**Environment Variables**:
```yaml
ENVIRONMENT: test
POSTGRES_HOST: localhost
POSTGRES_PORT: 5432
POSTGRES_DATABASE: bridge_test
POSTGRES_USER: test
KAFKA_BOOTSTRAP_SERVERS: localhost:9092
```

**Success Criteria**:
- All unit tests pass
- All integration tests pass
- Coverage ≥ 80% for both nodes
- Performance benchmarks within thresholds

---

#### 2. lint-and-type-check.yml
**Purpose**: Code quality validation and ONEX pattern compliance

**Triggers**:
- Push to main, develop, feature branches
- Pull requests
- Changes to node source code or tests

**Jobs**:
- **black-formatting**: Code formatting validation
  - Checks orchestrator, reducer, and tests
  - Provides auto-fix command on failure

- **isort-check**: Import sorting validation
  - Profile: black (88 char limit)
  - Separate checks per node

- **ruff-linting**: Fast Python linting
  - GitHub annotations for errors
  - ONEX-specific rule validation
  - Output format: github

- **mypy-type-check**: Type checking (informational)
  - Continue-on-error enabled
  - Informational only, doesn't block build

- **bandit-security**: Security vulnerability scanning
  - JSON output format
  - Separate reports per node
  - Artifact uploads for analysis

- **onex-pattern-validation**: Custom ONEX compliance
  - Naming convention validation
  - Contract pattern checking
  - Model import validation

- **quality-summary**: Aggregate status report
  - Required checks: black, isort, ruff
  - Optional checks: mypy, bandit
  - Overall pass/fail status

**Validated Patterns**:
- Node naming: `Node<Name><Type>` (e.g., NodeOrchestratorV1)
- File naming: `node_*_<type>.py`
- Model imports: `from .models import Model*`
- Enum naming: `Enum<Name>`

---

#### 3. docker-build.yml
**Purpose**: Docker image builds and registry publishing

**Triggers**:
- Push to main, develop, feature/node-implementation
- Pull requests
- Changes to Dockerfiles or node source
- Manual workflow dispatch

**Jobs**:
- **build-orchestrator**: Orchestrator build
  - Platforms: linux/amd64
  - Multi-stage build for optimization
  - Metadata labels (OCI-compliant)
  - Import validation test
  - Push to GHCR on main branch

- **build-reducer**: Reducer build
  - Platforms: linux/amd64
  - Same configuration as orchestrator
  - Separate user/UID (1001 vs 1000)
  - Independent build cache

- **security-scan**: Trivy vulnerability scanning
  - Matrix: [orchestrator, reducer]
  - Severity: CRITICAL, HIGH
  - SARIF upload to GitHub Security
  - Artifact retention: 30 days

- **image-analysis**: Image size validation
  - Target: < 500MB per image
  - Size comparison and reporting
  - Warning thresholds

- **build-summary**: Aggregate build status
  - Build results for all jobs
  - Registry URLs for published images
  - Overall pass/fail status

**Registry Configuration**:
```yaml
Registry: ghcr.io
Namespace: omninode-ai
Images:
  - omninode-bridge-orchestrator:latest
  - omninode-bridge-reducer:latest
```

**Image Tags**:
- `latest`: Main branch builds
- `<branch>-<sha>`: All branches
- `v<version>`: Semantic version tags
- `<pr-number>`: Pull request builds

---

### Docker Configuration

#### Dockerfiles

**Dockerfile.orchestrator** (`docker/bridge-nodes/`)
- Base: python:3.12-slim
- Multi-stage build (builder + runtime)
- User: orchestrator (UID 1000)
- Security: Non-root execution, minimal dependencies
- Health check: Python import validation
- Size target: < 500MB

**Dockerfile.reducer** (`docker/bridge-nodes/`)
- Base: python:3.12-slim
- Multi-stage build (builder + runtime)
- User: reducer (UID 1001)
- Security: Non-root execution, minimal dependencies
- Health check: Python import validation
- Size target: < 500MB

**Common Features**:
- Poetry 1.8.3 for dependency management
- Production-only dependencies in runtime
- OCI-compliant labels (version, source, authors)
- Configurable via environment variables
- Volume mounts for logs and data

#### docker-compose.test.yml

**Purpose**: Local testing environment

**Services**:
1. **postgres-test**
   - Image: postgres:15
   - Port: 5433 (mapped from 5432)
   - Database: bridge_test
   - Health check: pg_isready

2. **kafka-test**
   - Image: redpandadata/redpanda:v24.2.7
   - Port: 9093 (mapped from 9092)
   - Mode: dev-container
   - Auto-create topics: enabled

3. **orchestrator**
   - Build: Dockerfile.orchestrator
   - Depends: postgres-test, kafka-test
   - Volumes: Source code, logs
   - Environment: Test configuration

4. **reducer**
   - Build: Dockerfile.reducer
   - Depends: postgres-test, kafka-test
   - Volumes: Source code, logs
   - Environment: Test configuration

**Usage**:
```bash
# Start test environment
cd docker/bridge-nodes
docker-compose -f docker-compose.test.yml up -d

# View logs
docker-compose -f docker-compose.test.yml logs -f

# Stop and clean up
docker-compose -f docker-compose.test.yml down -v
```

---

## Pipeline Flow

### Pull Request Flow

```text
PR Created
    ↓
┌─────────────────────────────────────┐
│  lint-and-type-check.yml            │
│  - Black formatting                 │
│  - isort import sorting             │
│  - Ruff linting                     │
│  - mypy type checking               │
│  - Bandit security                  │
│  - ONEX pattern validation          │
└─────────────────────────────────────┘
    ↓ (parallel)
┌─────────────────────────────────────┐
│  test-bridge-nodes.yml              │
│  - Unit tests (orchestrator)        │
│  - Unit tests (reducer)             │
│  - Integration tests                │
│  - Performance benchmarks           │
│  - Coverage report                  │
└─────────────────────────────────────┘
    ↓ (parallel)
┌─────────────────────────────────────┐
│  docker-build.yml                   │
│  - Build orchestrator               │
│  - Build reducer                    │
│  - Security scan (Trivy)            │
│  - Image analysis                   │
└─────────────────────────────────────┘
    ↓
All Checks Pass → Merge Enabled
```

### Main Branch Flow

```
Merge to Main
    ↓
┌─────────────────────────────────────┐
│  All PR checks run again            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  docker-build.yml                   │
│  - Build images                     │
│  - Security scan                    │
│  - Push to GHCR                     │
│  - Tag: latest, <sha>, <version>    │
└─────────────────────────────────────┘
    ↓
Images Published to GHCR
```

---

## Performance Metrics

### Test Execution Times

| Job | Target | Actual (Est.) |
|-----|--------|---------------|
| Unit Tests (Orchestrator) | < 30s | ~15-20s |
| Unit Tests (Reducer) | < 30s | ~15-20s |
| Integration Tests | < 60s | ~40-50s |
| Performance Benchmarks | < 120s | ~60-90s |
| Linting (all checks) | < 60s | ~30-40s |
| Docker Build (single) | < 300s | ~180-240s |

### Coverage Targets

| Component | Target | Enforcement |
|-----------|--------|-------------|
| Orchestrator | ≥ 80% | Warning only |
| Reducer | ≥ 80% | Warning only |
| Integration | N/A | Informational |
| Overall | ≥ 80% | Recommended |

### Image Size Targets

| Image | Target | Warning Threshold |
|-------|--------|-------------------|
| Orchestrator | < 400MB | 500MB |
| Reducer | < 400MB | 500MB |

---

## Status Checks for PRs

### Required Checks (Block Merge)
- ✅ Black formatting
- ✅ isort import sorting
- ✅ Ruff linting
- ✅ Unit tests (orchestrator)
- ✅ Unit tests (reducer)
- ✅ Integration tests
- ✅ Docker build (orchestrator)
- ✅ Docker build (reducer)

### Optional Checks (Informational)
- ℹ️ mypy type checking
- ℹ️ Bandit security scan
- ℹ️ Performance benchmarks
- ℹ️ Image size analysis
- ℹ️ ONEX pattern validation

---

## Environment Configuration

### Test Environment Variables

```bash
# Core configuration
ENVIRONMENT=test
LOG_LEVEL=info
SERVICE_VERSION=${GITHUB_SHA}

# Security (test values only)
SECURITY_MODE=permissive
API_KEY=test-api-key-for-ci
JWT_SECRET=test-jwt-secret-minimum-32-characters-long

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=bridge_test
POSTGRES_USER=test
POSTGRES_PASSWORD=test-password

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_WORKFLOW_TOPIC=test.omninode_bridge.onex.workflows.v1
KAFKA_TASK_EVENTS_TOPIC=test.omninode_bridge.onex.task-events.v1
```

### Production Environment Variables

```bash
# Set via GitHub Secrets
POSTGRES_PASSWORD=<secret>
API_KEY=<secret>
JWT_SECRET=<secret>
GITHUB_TOKEN=<auto-provided>
```

---

## Artifact Management

### Test Artifacts (30 day retention)

**Unit Tests**:
- `coverage-orchestrator.xml`: Orchestrator coverage report
- `coverage-reducer.xml`: Reducer coverage report
- `htmlcov-orchestrator/`: HTML coverage report (orchestrator)
- `htmlcov-reducer/`: HTML coverage report (reducer)
- `pytest-orchestrator.xml`: JUnit test results (orchestrator)
- `pytest-reducer.xml`: JUnit test results (reducer)

**Integration Tests**:
- `coverage-integration.xml`: Integration test coverage
- `pytest-integration.xml`: JUnit test results

**Performance Tests**:
- `benchmark-results.json`: Performance benchmark data
- `.benchmarks/`: Historical benchmark data

**Security Scans**:
- `bandit-orchestrator.json`: Security scan (orchestrator)
- `bandit-reducer.json`: Security scan (reducer)
- `trivy-orchestrator.sarif`: Vulnerability scan (orchestrator)
- `trivy-reducer.sarif`: Vulnerability scan (reducer)

---

## Troubleshooting

### Common Issues

**1. Test Timeout**
```
Error: Test exceeded timeout
Solution: Increase timeout in workflow or optimize slow tests
```

**2. Coverage Below Threshold**
```
Error: Coverage 75% < 80%
Solution: Add tests or adjust threshold in pyproject.toml
```

**3. Docker Build Failed**
```
Error: Layer cache miss
Solution: Check GITHUB_TOKEN permissions for cache access
```

**4. Security Vulnerabilities**
```
Error: Trivy found CRITICAL vulnerabilities
Solution: Update dependencies or add exceptions
```

### Debug Commands

```bash
# Run tests locally (same as CI)
poetry run pytest tests/unit/nodes/ -v

# Check formatting
poetry run black --check src/omninode_bridge/nodes/
poetry run isort --check src/omninode_bridge/nodes/
poetry run ruff check src/omninode_bridge/nodes/

# Build Docker images locally
docker build -f docker/bridge-nodes/Dockerfile.orchestrator -t test:orchestrator .
docker build -f docker/bridge-nodes/Dockerfile.reducer -t test:reducer .

# Run security scans locally
poetry run bandit -r src/omninode_bridge/nodes/
trivy image test:orchestrator
```

---

## Next Steps

### Phase 2 Enhancements

1. **Performance Optimization**
   - Parallel test execution with pytest-xdist
   - Docker layer caching improvements
   - Test result caching between runs

2. **Advanced Testing**
   - Load testing with Locust
   - Chaos testing with chaos-mesh
   - Contract testing with Pact

3. **Deployment Automation**
   - Kubernetes deployment manifests
   - Helm chart generation
   - ArgoCD GitOps integration

4. **Monitoring & Observability**
   - Prometheus metrics export
   - Grafana dashboards
   - OpenTelemetry tracing

5. **Security Enhancements**
   - SBOM generation
   - Image signing with Cosign
   - Runtime security with Falco

---

## References

- [GitHub Actions](https://docs.github.com/en/actions)
- [Docker Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [pytest Documentation](https://docs.pytest.org/)
- [GHCR Documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [ONEX Architecture](/docs/ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md)
