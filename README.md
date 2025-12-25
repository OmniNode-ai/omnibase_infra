# omnibase_infra

Fresh ONEX-compliant infrastructure repository for OmniNode AI.

## Status

**Created**: December 2, 2025
**Version**: 0.1.0 (MVP)

## Overview

This repository contains ONEX infrastructure services built with:
- `omnibase-core` ^0.3.5 (PyPI)
- `omnibase-spi` ^0.2.0 (PyPI)

## Structure

```
src/omnibase_infra/
├── adapters/          # Thin external service wrappers
├── clients/           # Service clients
├── enums/             # Centralized enums
├── models/            # Centralized Pydantic models
├── nodes/             # ONEX nodes (Effect, Compute, Reducer, Orchestrator)
├── infrastructure/    # Infrastructure utilities
├── shared/            # Shared utilities
└── utils/             # General utilities
```

## Getting Started

```bash
poetry install
poetry run python -c "import omnibase_infra; print('Ready')"
```

## Deployment Options

### Docker Deployment

For containerized deployment, see the [Docker documentation](docker/README.md).

**Quick Start with Docker:**

```bash
cd docker
cp .env.example .env
# Edit .env and replace ALL __REPLACE_WITH_*__ placeholders
docker compose -f docker-compose.runtime.yml --profile main up -d --build
curl http://localhost:8085/health
```

**Available Docker Profiles:**

| Profile   | Services                    | Use Case                          |
|-----------|-----------------------------|-----------------------------------|
| `main`    | runtime-main                | Core kernel only                  |
| `effects` | runtime-main + effects      | Main + external service I/O       |
| `workers` | runtime-main + workers ×2   | Main + parallel compute           |
| `all`     | All services                | Full deployment                   |

See [docker/README.md](docker/README.md) for detailed configuration, security, and troubleshooting.

### Non-Docker Development

For local development without Docker:

1. **Install Python dependencies:**
   ```bash
   poetry install
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your infrastructure settings
   # See "Integration Testing" section for required variables
   ```

3. **Run the application:**
   ```bash
   poetry run python -m omnibase_infra.runtime.kernel
   ```

**Infrastructure Options for Non-Docker Users:**

| Option | Description |
|--------|-------------|
| **Remote Server** | Connect to `192.168.86.200` (ONEX development infrastructure) |
| **Local Services** | Run PostgreSQL, Consul, Vault, Kafka locally |
| **CI/CD Mode** | Leave infrastructure variables unset (tests skip gracefully) |

**Local Service Setup (if needed):**

```bash
# PostgreSQL (via Homebrew on macOS)
brew install postgresql@15 && brew services start postgresql@15

# Consul (via Homebrew on macOS)
brew install consul && consul agent -dev

# Vault (via Homebrew on macOS)
brew install vault && vault server -dev

# Or use Docker for just the services (without running the app in Docker)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=dev postgres:15
docker run -d -p 8500:8500 consul:1.15
docker run -d -p 8200:8200 vault:1.15 server -dev
```

**Note:** Most integration tests will skip gracefully if infrastructure is unavailable. The HTTP handler tests always run using local mock servers (pytest-httpserver).

## Documentation

See `docs/` for planning documents:
- `MVP_EXECUTION_PLAN.md` - Detailed execution plan
- `HANDOFF_OMNIBASE_INFRA_MVP.md` - Project handoff document
- `DECLARATIVE_EFFECT_NODES_PLAN.md` - Contract-driven effect nodes plan

## Development

### Pre-commit Hooks Setup

This repository uses pre-commit hooks for automatic code formatting and validation.

**Initial Setup** (run once):
```bash
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
```

**What happens automatically**:
- On `git commit`: Ruff formatting, file checks, ONEX validations
- On `git push`: Type checking with mypy

**Manual execution** (optional):
```bash
# Run all pre-commit hooks
poetry run pre-commit run --all-files

# Run specific hook
poetry run pre-commit run ruff-format --all-files

# Update hook versions
poetry run pre-commit autoupdate
```

**Note**: If hooks modify files, you need to re-stage and commit again. This ensures formatting is always applied before code reaches CI.

### Ruff Formatting

Ruff handles both formatting and linting (replaces black + isort):
- Format code: `poetry run ruff format .`
- Check and fix linting: `poetry run ruff check --fix .`
- Both run automatically via pre-commit hooks

### Development Workflow

Follow the patterns in `omniintelligence` repository for ONEX compliance.

## Integration Testing

Integration tests validate handlers against real infrastructure services. They are designed to skip gracefully when infrastructure is unavailable, enabling CI/CD pipelines to run without hard failures.

### Required Infrastructure

| Service | Default Host | Default Port | Environment Variables |
|---------|-------------|--------------|----------------------|
| PostgreSQL | `192.168.86.200` | `5436` | `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_PASSWORD` |
| Consul | `192.168.86.200` | `28500` | `CONSUL_HOST`, `CONSUL_PORT` |
| Vault | `192.168.86.200` | `8200` | `VAULT_ADDR`, `VAULT_TOKEN` |
| Kafka/Redpanda | `192.168.86.200` | `29092` | `KAFKA_BOOTSTRAP_SERVERS` |

### Environment Setup

```bash
# Copy example environment file
cp .env.example .env

# Set required credentials (example values - use your own)
export POSTGRES_HOST=192.168.86.200
export POSTGRES_PORT=5436
export POSTGRES_PASSWORD=your_password
export CONSUL_HOST=192.168.86.200
export CONSUL_PORT=28500
export VAULT_ADDR=http://192.168.86.200:8200
export VAULT_TOKEN=your_vault_token
```

### Running Integration Tests

```bash
# Run all integration tests
poetry run pytest tests/integration/ -v

# Run specific handler integration tests
poetry run pytest tests/integration/handlers/ -v

# Run with markers
poetry run pytest -m integration -v
```

### CI/CD Graceful Skip Behavior

Integration tests automatically skip when infrastructure is unavailable:

| Handler | Skip Conditions |
|---------|----------------|
| **DbHandler** | `POSTGRES_HOST` not set, or `POSTGRES_PASSWORD` not set |
| **ConsulHandler** | `CONSUL_HOST` not set, or TCP connection fails |
| **VaultHandler** | `VAULT_ADDR` not set, `VAULT_TOKEN` not set, or health endpoint unreachable |
| **HttpRestHandler** | Never skips (uses pytest-httpserver for local mock testing) |

Example CI/CD output when infrastructure is unavailable:

```
tests/.../test_db_handler_integration.py::test_db_health_check SKIPPED (PostgreSQL not available)
tests/.../test_consul_handler_integration.py::test_consul_health_check SKIPPED (Consul not available)
tests/.../test_vault_handler_integration.py::test_vault_health_check SKIPPED (Vault not available)
tests/.../test_http_handler_integration.py::test_http_get_success PASSED
```

### Local Development Without Infrastructure

For local development without access to the remote infrastructure server:

1. **Use Docker Compose** (recommended): See [docker/README.md](docker/README.md) for running local services
2. **Skip infrastructure tests**: Tests automatically skip when services are unavailable
3. **Mock testing**: HTTP handler tests use `pytest-httpserver` and always run locally

### Test Isolation

Integration tests use isolation patterns to prevent test pollution:

- **Unique identifiers**: Each test generates unique table names, KV keys, and secret paths
- **Cleanup fixtures**: Fixtures ensure test data is cleaned up after each test
- **Idempotent cleanup**: Cleanup operations ignore errors (e.g., deleting non-existent resources)

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Commit message format and conventions
- Agent attribution guidelines
- Code review processes
