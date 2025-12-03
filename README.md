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

## Documentation

See `docs/` for planning documents:
- `MVP_EXECUTION_PLAN.md` - Detailed execution plan
- `HANDOFF_OMNIBASE_INFRA_MVP.md` - Project handoff document
- `DECLARATIVE_EFFECT_NODES_PLAN.md` - Contract-driven effect nodes plan

## Development

Follow the patterns in `omniintelligence` repository for ONEX compliance.
