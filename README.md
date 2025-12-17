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

## Docker Deployment

For containerized deployment, see the [Docker documentation](docker/README.md).

### Quick Start

```bash
cd docker
cp .env.example .env
# Edit .env and replace ALL __REPLACE_WITH_*__ placeholders
docker compose -f docker-compose.runtime.yml --profile main up -d --build
curl http://localhost:8085/health
```

### Available Profiles

| Profile   | Services                    | Use Case                          |
|-----------|-----------------------------|-----------------------------------|
| `main`    | runtime-main                | Core kernel only                  |
| `effects` | runtime-main + effects      | Main + external service I/O       |
| `workers` | runtime-main + workers ×2   | Main + parallel compute           |
| `all`     | All services                | Full deployment                   |

See [docker/README.md](docker/README.md) for detailed configuration, security, and troubleshooting.

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

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Commit message format and conventions
- Agent attribution guidelines
- Code review processes
