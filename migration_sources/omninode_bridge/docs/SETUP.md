# Development Environment Setup Guide

This guide covers the complete development environment setup for omninode_bridge, including dependencies, configuration, and quick start instructions.

## Dependencies

### Core Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.1"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
asyncpg = "^0.29.0"
blake3 = "^0.4.1"
pydantic = "^2.5.0"
structlog = "^23.2.0"
prometheus-client = "^0.19.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
pytest-asyncio = "^0.21.1"
pytest-benchmark = "^4.0.0"
black = "^23.11.0"
mypy = "^1.7.1"
```

## Environment Configuration

### Development Environment

```bash
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=metadata_stamping_dev
POSTGRES_USER=dev_user
POSTGRES_PASSWORD=dev_password

# Service Configuration
SERVICE_PORT=8053
LOG_LEVEL=DEBUG

# Performance Tuning (Development)
HASH_GENERATOR_POOL_SIZE=50
DATABASE_POOL_MAX_SIZE=20
DATABASE_POOL_MIN_SIZE=5
```

### Production Environment

```bash
# PostgreSQL Configuration
POSTGRES_HOST=postgres-prod.example.com
POSTGRES_PORT=5432
POSTGRES_DB=metadata_stamping_prod
POSTGRES_USER=prod_user
POSTGRES_PASSWORD=${SECURE_PASSWORD}
ENABLE_SSL_DATABASE=true

# Service Configuration
SERVICE_PORT=8053
SERVICE_WORKERS=4
LOG_LEVEL=INFO

# Performance Tuning (Production)
HASH_GENERATOR_POOL_SIZE=200
DATABASE_POOL_MAX_SIZE=100
DATABASE_POOL_MIN_SIZE=20
```

### Circuit Breaker Configuration (Production Tuning)

Configure database circuit breaker thresholds for resilience:

```bash
# Circuit Breaker Settings
DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5    # Failures before circuit opens (default: 5)
DB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60     # Seconds before retry attempt (default: 60)
DB_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS=3  # Max concurrent calls during recovery (default: 3)
```

**Production Tuning Guidelines:**

- **Low-latency environments**: Lower `failure_threshold` (3-5) for faster failover
- **High-latency environments**: Increase `recovery_timeout` (90-120s) to avoid premature retries
- **High-traffic systems**: Increase `half_open_max_calls` (5-10) for faster recovery validation
- **Database maintenance windows**: Temporarily increase `recovery_timeout` to reduce retry load

**Environment Variable Override:**

Set environment variables to tune circuit breaker behavior without code changes:

```bash
# More aggressive failover for low-latency systems
DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
DB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30

# Conservative settings for high-latency systems
DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD=10
DB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=120
DB_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS=5
```

**Configuration via ModelONEXContainer:**

```python
# Circuit breaker reads config from ModelONEXContainer
container = ModelONEXContainer()
circuit_breaker_config = container.config.database.circuit_breaker()
circuit_breaker = DatabaseCircuitBreaker.from_config(circuit_breaker_config)
```

### Pool Exhaustion Monitoring

The PostgresConnectionManager implements automatic pool exhaustion detection to prevent connection starvation and identify resource contention issues.

**Configuration:**

```bash
# Pool Exhaustion Settings
POSTGRES_POOL_EXHAUSTION_THRESHOLD=0.90    # Alert at 90% utilization (default)
POSTGRES_POOL_EXHAUSTION_LOG_INTERVAL=60   # Seconds between warnings to prevent log spam (default)

# Example: More aggressive monitoring
POSTGRES_POOL_EXHAUSTION_THRESHOLD=0.80  # Alert at 80% utilization
POSTGRES_POOL_EXHAUSTION_LOG_INTERVAL=30  # Log every 30 seconds
```

**How It Works:**

- Monitors connection pool utilization at every connection acquisition and stats retrieval
- Logs warnings when utilization exceeds configured threshold (default: 90%)
- Rate-limited logging prevents log spam (default: 60-second interval between warnings)
- Tracks exhaustion metrics for trending and analysis

**Monitoring Metrics:**

```python
# Get pool statistics with exhaustion metrics
stats = manager.get_pool_stats()
print(f"Utilization: {stats['utilization_percent']:.1f}%")
print(f"Exhaustion warnings: {stats['exhaustion_warning_count']}")
print(f"Threshold: {stats['exhaustion_threshold_percent']:.1f}%")

# Example output when pool is exhausted:
# {
#   'pool_size': 50,
#   'pool_free': 3,
#   'pool_max': 50,
#   'used_connections': 47,
#   'utilization_percent': 94.0,
#   'exhaustion_threshold_percent': 90.0,
#   'exhaustion_warning_count': 5
# }
```

**Warning Log Format:**

```python
# Structured log output when threshold exceeded
{
    "event": "pool_exhaustion",
    "utilization_percent": 94.0,
    "used_connections": 47,
    "free_connections": 3,
    "pool_size": 50,
    "pool_max": 50,
    "threshold_percent": 90.0,
    "exhaustion_count": 5,
    "time_since_last_warning_seconds": 62.3,
    "recommendation": "Consider increasing max_connections or investigating connection leaks"
}
```

**Production Tuning Guidelines:**

- **High-traffic systems**: Lower threshold (0.85 or 0.80) for earlier warnings
- **Burst workloads**: Increase log interval (120s) to reduce log volume during expected spikes
- **Development**: More aggressive settings (0.70 threshold, 30s interval) for early detection
- **Investigation**: Monitor `exhaustion_warning_count` trends to identify capacity issues

**Performance Impact:**

- Monitoring overhead: <1ms per connection operation
- No blocking operations or database queries
- Rate-limited logging prevents performance degradation

### Query Metrics Configuration

Configure maximum number of query metrics to store in circular buffer:

```bash
# Query Metrics Settings
POSTGRES_MAX_METRICS_STORED=10000         # Max metrics stored (default: 10000, range: 1000-1000000)
```

## Docker Setup

### Dockerfile

```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 libpq-dev build-essential

# Install Poetry
RUN pip install poetry==1.7.1

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --only=main

COPY src/ ./src/
EXPOSE 8053 9090

CMD ["poetry", "run", "uvicorn", "src.metadata_stamping.main:app", \
     "--host", "0.0.0.0", "--port", "8053", "--workers", "4"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: metadata_stamping_dev
      POSTGRES_USER: dev_user
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  metadata_stamping:
    build: .
    ports:
      - "8053:8053"
      - "9090:9090"
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_PORT: 5432
      POSTGRES_DB: metadata_stamping_dev
      POSTGRES_USER: dev_user
      POSTGRES_PASSWORD: dev_password
      SERVICE_PORT: 8053
      LOG_LEVEL: INFO
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
```

## Quick Start

### Local Development

```bash
# 1. Clone repository
git clone <repository-url>
cd omninode_bridge

# 2. Install dependencies
poetry install

# 3. Configure Kafka/Redpanda hostname (ONE-TIME SETUP)
# Kafka/Redpanda uses a two-step broker discovery protocol that requires
# resolving the Docker container hostname. Other services (PostgreSQL, HTTP)
# do not have this requirement.
echo "127.0.0.1 omninode-bridge-redpanda" | sudo tee -a /etc/hosts

# Verify the hostname was added:
grep omninode-bridge-redpanda /etc/hosts

# 4. Start PostgreSQL (Docker)
docker compose up -d postgres

# 5. Run database migrations
poetry run alembic upgrade head

# 6. Start development server
poetry run uvicorn src.metadata_stamping.main:app --reload --port 8053

# 7. Access services
# API: http://localhost:8053
# API Docs: http://localhost:8053/docs
# Metrics: http://localhost:9090
```

### Running Tests

```bash
# Run all tests
poetry run pytest tests/

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html

# Run performance tests
poetry run pytest tests/performance/ -m performance

# Run integration tests
poetry run pytest tests/integration/ -v

# Run specific test file
poetry run pytest tests/unit/test_hash_generator.py -v
```

### Development Workflow

```bash
# Format code
poetry run black src/ tests/

# Type checking
poetry run mypy src/

# Linting
poetry run ruff check src/ tests/

# Run pre-commit hooks
poetry run pre-commit run --all-files
```

## Bridge Nodes Development

### Docker Build Requirements

**Prerequisites for Building Docker Images:**

OmniNode Bridge uses Docker Compose v2 and Docker BuildKit for multi-stage builds and private package access.

#### 1. Docker Compose v2

**Required**: Docker Compose v2+ (uses `docker compose` command, not `docker-compose`)

```bash
# ✅ Correct (Docker Compose v2)
docker compose build
docker compose up -d

# ❌ Incorrect (Docker Compose v1 - deprecated)
docker-compose build
docker-compose up -d
```

**Check your version:**
```bash
docker compose version
# Should output: Docker Compose version v2.x.x or later
```

**Upgrade if needed:**
- macOS: Update Docker Desktop to latest version
- Linux: Follow [Docker Compose v2 installation guide](https://docs.docker.com/compose/install/)

#### 2. Docker BuildKit

Docker BuildKit is **enabled by default** in Docker 23.0+ and Docker Desktop.

**Verify BuildKit is enabled:**
```bash
docker buildx version
# Should output: github.com/docker/buildx vX.X.X or later
```

**Manual BuildKit Configuration** (only if needed for older Docker versions):
```bash
# Enable BuildKit for current session
export DOCKER_BUILDKIT=1

# Enable BuildKit permanently (Linux)
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
source ~/.bashrc

# Enable BuildKit permanently (macOS)
echo 'export DOCKER_BUILDKIT=1' >> ~/.zshrc
source ~/.zshrc
```

#### 3. GitHub Personal Access Token (GH_PAT)

**Required for**: Accessing private `omnibase_core` package during Docker builds

**Setup GitHub PAT:**

1. **Generate a Personal Access Token** at: https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Name: `omninode-bridge-docker-build`
   - Expiration: 90 days (or as needed)
   - **Required Permissions**:
     - ✅ `read:packages` - Download packages from GitHub Container Registry
     - ✅ `repo` (or `read:org` for private repos) - Access private repositories

2. **Save the token** (you won't be able to see it again!)

3. **Export the token** in your shell:
   ```bash
   # Set GH_PAT for current session
   export GH_PAT=ghp_your_token_here_1234567890abcdefghijklmnop  # pragma: allowlist secret

   # Verify it's set
   echo $GH_PAT
   # Should output: ghp_your_token_here...
   ```

4. **Optional: Persist GH_PAT** (for convenience, but be cautious with security):
   ```bash
   # Linux
   echo 'export GH_PAT=ghp_your_token_here...' >> ~/.bashrc
   source ~/.bashrc

   # macOS
   echo 'export GH_PAT=ghp_your_token_here...' >> ~/.zshrc
   source ~/.zshrc

   # Alternative: Use .env file (recommended for projects)
   echo "GH_PAT=ghp_your_token_here..." >> .env.local
   source .env.local
   ```

**Security Best Practices:**
- ✅ Use token expiration (90 days recommended)
- ✅ Use minimal permissions (`read:packages`, `read:org` only)
- ✅ Rotate tokens regularly
- ❌ Never commit tokens to git
- ❌ Never share tokens in chat/email

#### 4. Building Docker Images

**Build with GH_PAT configured:**

```bash
# Ensure GH_PAT is exported
export GH_PAT=ghp_your_token_here...

# Build all services
docker compose -f deployment/docker-compose.bridge.yml build

# Build specific service
docker compose -f deployment/docker-compose.bridge.yml build metadata_stamping

# Build with no cache (force rebuild)
docker compose -f deployment/docker-compose.bridge.yml build --no-cache

# Build and start services
docker compose -f deployment/docker-compose.bridge.yml up --build
```

**How it works:**
```yaml
# docker-compose.bridge.yml
services:
  metadata_stamping:
    build:
      context: .
      dockerfile: Dockerfile
      secrets:
        - gh_pat  # GH_PAT environment variable passed as build secret
```

```dockerfile
# Dockerfile
# --mount=type=secret accesses the GH_PAT during build
RUN --mount=type=secret,id=gh_pat sh -c '\
    set -euo pipefail; \
    GH_PAT="$(cat /run/secrets/gh_pat)"; \
    pip install "git+https://${GH_PAT}:x-oauth-basic@github.com/yourorg/omnibase_core.git@main" \
'
```

#### Troubleshooting

**Issue: `secret "gh_pat" not found`**

```bash
# Symptom
ERROR: failed to solve: failed to read dockerfile secret "gh_pat": file does not exist

# Cause
# GH_PAT environment variable is not set in your shell

# Solution 1: Export GH_PAT
export GH_PAT=ghp_your_token_here...
docker compose -f deployment/docker-compose.bridge.yml build

# Solution 2: Inline GH_PAT (one-time use)
GH_PAT=ghp_your_token_here... docker compose -f deployment/docker-compose.bridge.yml build

# Verify GH_PAT is set
echo $GH_PAT
# Should output: ghp_your_token_here... (if set correctly)
```

**Issue: `fatal: could not read Username` or `Authentication failed`**

```bash
# Symptom
fatal: could not read Username for 'https://github.com': No such device or address
# OR
remote: Invalid username or password.

# Cause
# Token lacks required permissions or is invalid

# Solution 1: Verify token permissions
# Go to https://github.com/settings/tokens and check:
# - ✅ read:packages
# - ✅ repo (or read:org for private repos)

# Solution 2: Generate new token
# Follow "Setup GitHub PAT" steps above

# Solution 3: Test token manually
curl -H "Authorization: token $GH_PAT" https://api.github.com/user
# Should return your GitHub user info (not an error)
```

**Issue: `docker-compose: command not found`**

```bash
# Symptom
docker-compose: command not found

# Cause
# Using Docker Compose v1 syntax on system with only v2 installed

# Solution: Use Docker Compose v2 syntax
docker compose -f deployment/docker-compose.bridge.yml build  # ✅ Correct
# NOT: docker-compose -f deployment/docker-compose.bridge.yml build  # ❌ Incorrect
```

**Issue: BuildKit not enabled**

```bash
# Symptom
ERROR: BuildKit is not enabled

# Cause
# Docker version < 23.0 with BuildKit not enabled

# Solution 1: Enable BuildKit for session
export DOCKER_BUILDKIT=1
docker compose build

# Solution 2: Enable BuildKit permanently
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc

# Solution 3: Upgrade Docker to 23.0+ (recommended)
# BuildKit is enabled by default in Docker 23.0+
```

**Issue: Build succeeds but service fails to start**

```bash
# Symptom
# Build completes successfully but container fails with import errors

# Cause
# Code changes require rebuild, not just restart

# Solution: Use --build flag
docker compose -f deployment/docker-compose.bridge.yml up --build

# Or rebuild then start
docker compose -f deployment/docker-compose.bridge.yml build
docker compose -f deployment/docker-compose.bridge.yml up -d
```

### Starting Bridge Nodes

```bash
# Run bridge orchestrator node
python -m omninode_bridge.nodes.orchestrator.v1_0_0.node

# Run bridge reducer node
python -m omninode_bridge.nodes.reducer.v1_0_0.node

# Run bridge registry node
python -m omninode_bridge.nodes.registry.v1_0_0.node
```

### Bridge Development Environment

**Prerequisites:** Complete "Docker Build Requirements" section above (GH_PAT setup, Docker Compose v2)

```bash
# 1. Ensure GH_PAT is exported
export GH_PAT=ghp_your_token_here...

# 2. Start bridge development environment
docker compose -f deployment/docker-compose.bridge.yml up -d

# 3. Build with latest code changes
docker compose -f deployment/docker-compose.bridge.yml up --build

# 4. Run bridge integration tests
pytest tests/integration/bridge/ -v

# 5. Run bridge performance tests
pytest tests/performance/bridge/ -m performance
```

## Troubleshooting

### Common Issues

**Issue: Kafka/Redpanda connection failures**

```bash
# Symptom: "Failed to resolve 'omninode-bridge-redpanda'" errors
# Cause: Kafka's two-step broker discovery requires hostname resolution

# Solution 1: Add hostname to /etc/hosts (RECOMMENDED)
echo "127.0.0.1 omninode-bridge-redpanda" | sudo tee -a /etc/hosts

# Solution 2: Verify hostname is configured
grep omninode-bridge-redpanda /etc/hosts
# Should output: 127.0.0.1 omninode-bridge-redpanda

# Solution 3: Test Kafka connectivity
poetry run python -c "from aiokafka import AIOKafkaProducer; import asyncio; asyncio.run(AIOKafkaProducer(bootstrap_servers='localhost:29092').start())"

# Note: This is Kafka/Redpanda-specific. Other services (PostgreSQL, Consul, HTTP)
# accept direct connections and do not require hostname configuration.
```

**Why Kafka/Redpanda is Different:**

1. **Bootstrap connection**: Client connects to `localhost:29092` ✅
2. **Metadata request**: Broker returns "I'm at `omninode-bridge-redpanda:9092`"
3. **Broker connection**: Client must resolve this hostname to connect

This is standard Kafka behavior and affects all Kafka clients (aiokafka, confluent-kafka, kcat).

**Issue: Connection refused to PostgreSQL**

```bash
# Check PostgreSQL is running
docker compose ps

# Check PostgreSQL logs
docker compose logs postgres

# Verify connection settings
poetry run python -c "import asyncpg; import asyncio; asyncio.run(asyncpg.connect('postgresql://dev_user:dev_password@localhost:5432/metadata_stamping_dev'))"  # pragma: allowlist secret
```

**Issue: Port already in use**

```bash
# Find process using port 8053
lsof -i :8053

# Kill process
kill -9 <PID>

# Or use different port
poetry run uvicorn src.metadata_stamping.main:app --reload --port 8054
```

**Issue: Poetry dependency conflicts**

```bash
# Clear Poetry cache
poetry cache clear --all pypi

# Remove lock file and reinstall
rm poetry.lock
poetry install
```

**Issue: Database migration failures**

```bash
# Check current migration version
poetry run alembic current

# Downgrade to previous version
poetry run alembic downgrade -1

# Re-run migration
poetry run alembic upgrade head

# Generate new migration
poetry run alembic revision --autogenerate -m "description"
```

## Performance Monitoring

### Prometheus Metrics

Access Prometheus metrics at `http://localhost:9090/metrics`:

- `hash_generation_duration_seconds` - Hash generation latency histogram
- `api_request_duration_seconds` - API request latency histogram
- `database_query_duration_seconds` - Database query latency histogram
- `active_connections` - Current active database connections
- `pool_exhaustion_warnings_total` - Total pool exhaustion warnings

### Structured Logging

Logs are structured in JSON format for easy parsing:

```json
{
  "timestamp": "2025-10-14T12:34:56.789Z",
  "level": "INFO",
  "logger": "metadata_stamping.hash_generator",
  "event": "hash_generated",
  "file_size_bytes": 1024,
  "execution_time_ms": 1.23,
  "performance_grade": "A"
}
```

### Grafana Dashboard (Pure Reducer Monitoring)

The Pure Reducer Refactor (v2.0) includes a pre-built Grafana dashboard for comprehensive monitoring of event sourcing, CQRS, and reducer operations.

**Dashboard Location**: `dashboards/pure_reducer_monitoring.json`

**Import Instructions**:

1. **Access Grafana UI**:
   ```bash
   # Local development
   http://localhost:3000

   # Production
   https://grafana.your-domain.com
   ```

2. **Import Dashboard**:
   - Navigate to: **Dashboards** → **Import** (+ icon in sidebar)
   - Click **Upload JSON file**
   - Select: `dashboards/pure_reducer_monitoring.json`
   - Configure data source:
     - **Prometheus**: Select your Prometheus data source
     - **PostgreSQL** (optional): For advanced queries
   - Click **Import**

3. **Verify Dashboard**:
   - Dashboard name: "Pure Reducer Monitoring"
   - Panels: 15+ monitoring panels including:
     - Reducer throughput and latency
     - State commit success/conflict rates
     - Projection materialization lag
     - Action deduplication metrics
     - Hot key contention indicators
     - Event bus performance

**Required Metrics** (ensure these are exposed):
```bash
# Reducer metrics
reducer_successful_actions_total
reducer_conflict_attempts_total
reducer_backoff_ms

# Canonical store metrics
canonical_store_state_commits_total
canonical_store_state_conflicts_total
canonical_store_commit_latency_ms

# Action dedup metrics
action_dedup_checks_total
action_dedup_hits_total
```

**Dashboard Features**:
- Real-time reducer performance tracking
- Conflict resolution monitoring
- Projection lag detection with watermark tracking
- Deduplication effectiveness metrics
- System health overview with alerting thresholds

**Troubleshooting**:
- **No data showing**: Ensure Prometheus is scraping metrics endpoint (`/metrics`)
- **Missing panels**: Verify all Pure Reducer services are running
- **Incorrect values**: Check that `KAFKA_BOOTSTRAP_SERVERS` and `POSTGRES_*` env vars are set

### Health Check

```bash
# Check service health
curl http://localhost:8053/health

# Expected response
{
  "status": "healthy",
  "components": {
    "database": "healthy",
    "cache": "healthy",
    "kafka": "healthy"
  },
  "uptime_seconds": 3600
}
```

## Development Best Practices

1. **Always use virtual environments**: `poetry shell` or activate venv
2. **Run tests before committing**: `pytest tests/`
3. **Format code**: `black src/ tests/` before committing
4. **Type checking**: Run `mypy src/` to catch type errors
5. **Database migrations**: Always generate migrations for schema changes
6. **Environment variables**: Never commit `.env` files with secrets
7. **Docker testing**: Test in Docker containers before deployment
8. **Performance testing**: Run benchmarks after optimization changes

## Next Steps

- **[Bridge Nodes Guide](./BRIDGE_NODES_GUIDE.md)** - Implementation guide for bridge nodes
- **[API Reference](./API_REFERENCE.md)** - Complete API documentation
- **[LlamaIndex Workflows](./LLAMAINDEX_WORKFLOWS_GUIDE.md)** - Workflow integration guide
- **[Database Schema](../migrations/schema.sql)** - Database schema reference
