# ONEX Infrastructure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy.readthedocs.io/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Framework: Infrastructure](https://img.shields.io/badge/framework-infrastructure-green.svg)](https://github.com/OmniNode-ai/omnibase_infra)

**Production infrastructure services for the ONEX execution layer.** Handlers, adapters, and runtime services for PostgreSQL, Kafka, Consul, Vault, and Redis.

## What is This?

This repository provides the **infrastructure layer** for ONEX-based systems. While [omnibase_core](https://github.com/OmniNode-ai/omnibase_core) defines the execution protocol and node archetypes, this package provides:

- **Handlers** for external services (database, HTTP, messaging)
- **Adapters** wrapping infrastructure clients
- **Event bus** abstractions for Kafka/Redpanda
- **Runtime services** deployable via Docker

Built on `omnibase-core` ^0.8.0 and `omnibase-spi` ^0.5.0.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/OmniNode-ai/omnibase_infra.git
cd omnibase_infra

# Start the runtime with Docker
cd docker
cp .env.example .env
# Edit .env - set POSTGRES_PASSWORD, VAULT_TOKEN, REDIS_PASSWORD

docker compose -f docker-compose.runtime.yml --profile main up -d --build

# Verify it's running
curl http://localhost:8085/health
```

## Docker Services

The runtime deploys as containerized services connecting to your infrastructure:

| Service | Profile | Port | Description |
|---------|---------|------|-------------|
| **runtime-main** | `main` | 8085 | Core kernel - request/response handling |
| **runtime-effects** | `effects` | 8086 | External service I/O (DB, HTTP, messaging) |
| **runtime-worker** | `workers` | — | Scalable compute workers (default: 2 replicas) |

**Profiles:**
```bash
# Core only
docker compose -f docker-compose.runtime.yml --profile main up -d

# Core + effects
docker compose -f docker-compose.runtime.yml --profile effects up -d

# Core + workers (parallel compute)
docker compose -f docker-compose.runtime.yml --profile workers up -d

# Everything
docker compose -f docker-compose.runtime.yml --profile all up -d
```

## Infrastructure Dependencies

The runtime connects to external services (not included in compose):

| Service | Purpose | Default Host | Environment Variable |
|---------|---------|--------------|---------------------|
| **PostgreSQL** | Persistence | `localhost:5432` | `POSTGRES_HOST`, `POSTGRES_PORT` |
| **Kafka/Redpanda** | Event bus | `localhost:9092` | `KAFKA_BOOTSTRAP_SERVERS` |
| **Consul** | Service discovery | `localhost:8500` | `CONSUL_HOST`, `CONSUL_PORT` |
| **Vault** | Secrets management | `localhost:8200` | `VAULT_ADDR` |
| **Redis/Valkey** | Caching | `localhost:6379` | `REDIS_HOST`, `REDIS_PORT` |

Configure via `.env` file - see [docker/README.md](docker/README.md) for details.

## Documentation

| I want to... | Go to... |
|--------------|----------|
| Get started quickly | [Quick Start Guide](docs/getting-started/quickstart.md) |
| Understand the architecture | [Architecture Overview](docs/architecture/overview.md) |
| Deploy with Docker | [Docker Guide](docker/README.md) |
| See a complete example | [Registration Walkthrough](docs/guides/registration-example.md) |
| Write a contract | [Contract Reference](docs/reference/contracts.md) |
| Find implementation patterns | [Pattern Documentation](docs/patterns/README.md) |
| Read coding standards | [CLAUDE.md](CLAUDE.md) |

**Full documentation**: [docs/index.md](docs/index.md)

## Repository Structure

```
src/omnibase_infra/
├── handlers/          # Request/message handlers
├── event_bus/         # Kafka/Redpanda abstractions
├── clients/           # Service clients
├── models/            # Pydantic models
├── nodes/             # ONEX nodes (Effect, Compute, Reducer, Orchestrator)
├── errors/            # Error hierarchy
├── mixins/            # Reusable behaviors
└── enums/             # Centralized enums
```

## Development

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Type checking
poetry run mypy src/omnibase_infra/

# Format code
poetry run ruff format .
poetry run ruff check --fix .
```

### Pre-commit Hooks Setup

Run once after cloning:
```bash
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit conventions and PR guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
