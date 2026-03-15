# omnibase_infra

Production infrastructure services for the ONEX execution layer.

[![CI](https://github.com/OmniNode-ai/omnibase_infra/actions/workflows/test.yml/badge.svg)](https://github.com/OmniNode-ai/omnibase_infra/actions/workflows/test.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Install

```bash
uv add omnibase-infra
```

## Minimal Example

```python
from omnibase_infra.handlers.handler_postgres import HandlerPostgres
from omnibase_infra.runtime.runtime_host import RuntimeHostProcess

# Contract-driven handler registration
process = RuntimeHostProcess(
    contract_paths=["src/omnibase_infra/nodes/my_node/contract.yaml"],
)
await process.start()
```

## Key Features

- **Handlers**: Database (PostgreSQL), HTTP, messaging (Kafka/Redpanda), caching (Valkey)
- **Adapters**: Infrastructure client wrappers with protocol-driven DI
- **Event bus**: Kafka producer/consumer abstractions with topic provisioning
- **Runtime services**: Deployable via Docker with contract-driven wiring
- **Config management**: Infisical integration with env var fallback
- **50+ ONEX nodes**: EFFECT, COMPUTE, REDUCER, ORCHESTRATOR implementations

## Documentation

- [Architecture](docs/architecture/)
- [Getting started](docs/getting-started/)
- [CLAUDE.md](CLAUDE.md) -- developer context and conventions

## License

[MIT](LICENSE)
