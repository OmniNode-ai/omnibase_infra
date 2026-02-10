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

# Start infrastructure services with Docker
cd docker
cp .env.example .env
# Edit .env - set OMNIBASE_INFRA_DB_URL and POSTGRES_PASSWORD (required for Docker)

docker compose -f docker-compose.infra.yml up -d

# Verify services are running
docker compose -f docker-compose.infra.yml ps
```

## Docker Services

Self-contained infrastructure via `docker-compose.infra.yml`:

| Service | Profile | Port | Description |
|---------|---------|------|-------------|
| **PostgreSQL** | default | 5436 | Persistence (always starts) |
| **Redpanda** | default | 29092 | Event bus (always starts) |
| **Valkey** | default | 16379 | Caching (always starts) |
| **Consul** | `consul` | 28500 | Service discovery (optional) |
| **Infisical** | `secrets` | 8880 | Secrets management (optional) |
| **Runtime** | `runtime` | 8085 | ONEX runtime services (optional) |

**Profiles:**
```bash
# Infrastructure only (default)
docker compose -f docker-compose.infra.yml up -d

# With service discovery
docker compose -f docker-compose.infra.yml --profile consul up -d

# With secrets management
docker compose -f docker-compose.infra.yml --profile secrets up -d

# Everything
docker compose -f docker-compose.infra.yml --profile full up -d
```

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

## Schema Migration Freeze

> **Schema migrations are currently frozen.** See [`.migration_freeze`](.migration_freeze) for details.

A DB-per-repo refactor is in progress (OMN-2055). While the freeze is active:

- **No new migrations** may be added
- **Allowed**: migration file moves, table ownership fixes, rollback bug fixes
- **To lift**: remove `.migration_freeze` and reference OMN-2055 in the PR

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
