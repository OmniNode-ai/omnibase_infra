> **Navigation**: [Home](../index.md) > Getting Started > Quick Start

# Quick Start Guide

Get ONEX Infrastructure running in 5 minutes.

## TL;DR - Minimal Working Example

Copy-paste this to create your first ONEX node in under 2 minutes:

```bash
# 1. Setup (one-time)
git clone <repo-url> omnibase_infra3 && cd omnibase_infra3
poetry install

# 2. Create a minimal node
mkdir -p src/omnibase_infra/nodes/node_hello_effect/models
```

```python
# src/omnibase_infra/nodes/node_hello_effect/models/__init__.py
from pydantic import BaseModel

class ModelHelloRequest(BaseModel):
    name: str

class ModelHelloResponse(BaseModel):
    greeting: str
```

```yaml
# src/omnibase_infra/nodes/node_hello_effect/contract.yaml
contract_version: { major: 1, minor: 0, patch: 0 }
node_version: "1.0.0"
name: "node_hello_effect"
node_type: "EFFECT_GENERIC"
description: "Minimal effect node example."
input_model: { name: "ModelHelloRequest", module: "omnibase_infra.nodes.node_hello_effect.models" }
output_model: { name: "ModelHelloResponse", module: "omnibase_infra.nodes.node_hello_effect.models" }
```

```python
# src/omnibase_infra/nodes/node_hello_effect/node.py
from omnibase_core.nodes.node_effect import NodeEffect

class NodeHelloEffect(NodeEffect):
    """Declarative - all behavior from contract.yaml."""
    pass  # Yes, really. The node is empty!
```

```bash
# 3. Verify it works
poetry run python -c "from omnibase_infra.nodes.node_hello_effect.node import NodeHelloEffect; print('Node loads!')"
```

**That's it!** The node class is intentionally empty - all behavior is contract-driven.

---

## Prerequisites

- Python 3.12+
- Poetry
- Docker (optional, for infrastructure services)

## Installation

```bash
# Clone and install
git clone <repo-url> omnibase_infra3
cd omnibase_infra3
poetry install

# Verify installation
poetry run python -c "import omnibase_infra; print('Ready!')"
```

## Understanding ONEX in 60 Seconds

ONEX uses **contract-driven nodes** organized into four archetypes:

### ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     ONEX 4-Node Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    events     ┌──────────────┐               │
│   │ ORCHESTRATOR │ ────────────► │   REDUCER    │               │
│   │  (workflow)  │               │  (state/FSM) │               │
│   └──────────────┘               └──────────────┘               │
│          │                              │                        │
│          │ routes                       │ emits intents          │
│          ▼                              ▼                        │
│   ┌──────────────┐               ┌──────────────┐               │
│   │   COMPUTE    │               │    EFFECT    │               │
│   │    (pure)    │               │  (external)  │               │
│   └──────────────┘               └──────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Mermaid Diagram

```mermaid
flowchart LR
    accTitle: ONEX 4-Node Architecture
    accDescr: The four ONEX node archetypes and their relationships. ORCHESTRATOR coordinates workflows and can publish events. It routes to COMPUTE nodes for pure transformations and sends events to REDUCER nodes. REDUCER manages state via FSM and emits intents to EFFECT nodes which handle external I/O like databases and APIs.

    subgraph Archetypes["ONEX 4-Node Architecture"]
        ORCH[ORCHESTRATOR<br/>workflow] -->|events| RED[REDUCER<br/>state/FSM]
        ORCH -->|routes| COMP[COMPUTE<br/>pure]
        RED -->|emits intents| EFF[EFFECT<br/>external]
    end

    style ORCH fill:#e3f2fd
    style RED fill:#fff3e0
    style COMP fill:#e8f5e9
    style EFF fill:#fce4ec
```

| Archetype | Purpose | Side Effects |
|-----------|---------|--------------|
| **ORCHESTRATOR** | Coordinates workflows, routes events to handlers | Publishes events |
| **REDUCER** | Manages state via FSM, emits intents | None (pure) |
| **COMPUTE** | Pure transformations and validation | None (pure) |
| **EFFECT** | External I/O (databases, APIs, services) | Yes |

## Your First Node

Every ONEX node has two parts:

1. **`contract.yaml`** - Declares what the node does (the "what")
2. **`node.py`** - Extends base class, contains no logic (declarative)

### Example: A Minimal Effect Node

```bash
# Create node directory
mkdir -p src/omnibase_infra/nodes/node_hello_effect
```

**`contract.yaml`**:
```yaml
contract_version:
  major: 1
  minor: 0
  patch: 0
node_version: "1.0.0"
name: "node_hello_effect"
node_type: "EFFECT_GENERIC"
description: "Simple effect node example."

input_model:
  name: "ModelHelloRequest"
  module: "omnibase_infra.nodes.node_hello_effect.models"

output_model:
  name: "ModelHelloResponse"
  module: "omnibase_infra.nodes.node_hello_effect.models"
```

**`node.py`**:
```python
from __future__ import annotations
from typing import TYPE_CHECKING
from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer

class NodeHelloEffect(NodeEffect):
    """Declarative effect node - all behavior from contract.yaml."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

__all__ = ["NodeHelloEffect"]
```

**Key principle**: The node class is empty! All behavior is driven by the contract.

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run unit tests only
poetry run pytest tests/unit/

# Run with coverage
poetry run pytest --cov=omnibase_infra
```

## Running ONEX Validators

ONEX enforces coding standards via validators:

```bash
# Run all validators
poetry run python scripts/validate.py all

# Check for forbidden Any types
poetry run python scripts/validate.py any_types

# Validate architecture patterns
poetry run python scripts/validate.py architecture
```

## Docker Deployment (Optional)

For running with full infrastructure:

```bash
cd docker
cp .env.example .env
# Edit .env with your settings

# Start core runtime
docker compose -f docker-compose.runtime.yml --profile main up -d

# Check health
curl http://localhost:8085/health
```

## Project Structure

```
src/omnibase_infra/
├── nodes/              # ONEX nodes (Effect, Compute, Reducer, Orchestrator)
│   ├── node_*/         # Each node has contract.yaml + node.py
│   └── reducers/       # Reducer implementations
├── handlers/           # Infrastructure handlers (Consul, DB, Vault, HTTP)
├── models/             # Pydantic models
├── enums/              # Centralized enums
├── adapters/           # External service adapters
└── runtime/            # Runtime kernel and dispatchers
```

## Key Concepts

### Contract-Driven Development

Everything is declared in YAML contracts:
- Handler routing → `handler_routing:` section
- State machines → `state_machine:` section
- Workflow graphs → `execution_graph:` section

### Dependency Injection

All nodes use container-based DI:
```python
def __init__(self, container: ModelONEXContainer) -> None:
    super().__init__(container)
```

### No `Any` Types

ONEX forbids `Any` - use `object` for generic payloads. This is enforced by CI.

### Handlers Cannot Publish Events

Only ORCHESTRATOR nodes may publish events. Handlers return `ModelHandlerOutput` with events, and the orchestrator publishes them.

## Next Steps

| Goal | Documentation |
|------|---------------|
| Understand the architecture | [Architecture Overview](../architecture/overview.md) |
| Learn the 4 node types | [Node Archetypes Reference](../reference/node-archetypes.md) |
| See a real example | [2-Way Registration Walkthrough](../guides/registration-example.md) |
| Write contracts | [Contract.yaml Reference](../reference/contracts.md) |
| Implement patterns | [Pattern Documentation](../patterns/README.md) |

## Common Commands

```bash
# Development
poetry install                          # Install dependencies
poetry run pytest                       # Run tests
poetry run ruff format .               # Format code
poetry run ruff check --fix .          # Lint and fix

# Validation
poetry run python scripts/validate.py all      # All validators
poetry run pre-commit run --all-files          # Pre-commit hooks

# Docker
docker compose -f docker-compose.runtime.yml --profile main up -d
docker compose -f docker-compose.runtime.yml logs -f
docker compose -f docker-compose.runtime.yml down
```

## Common Mistakes

New users often encounter these pitfalls. Here's how to avoid them:

### 1. Adding Logic to Node Classes

**Wrong:**
```python
class NodeHelloEffect(NodeEffect):
    def process(self, request):  # Don't do this!
        return {"greeting": f"Hello, {request.name}"}
```

**Right:**
```python
class NodeHelloEffect(NodeEffect):
    """All behavior from contract.yaml."""
    pass  # Node is declarative - no custom logic
```

Nodes are declarative. Business logic belongs in handlers, which are declared in the contract.

### 2. Using `Any` Type

**Wrong:**
```python
def process_data(payload: Any) -> Any:  # CI will reject this
    return payload
```

**Right:**
```python
def process_data(payload: object) -> object:  # Use object for generic payloads
    return payload
```

ONEX forbids `Any` types. Use `object` for generic payloads, or better yet, use specific Pydantic models.

### 3. Publishing Events from Handlers

**Wrong:**
```python
class MyHandler:
    def __init__(self, event_bus):  # Handlers cannot have bus access
        self._bus = event_bus

    async def handle(self, event):
        await self._bus.publish(response)  # Not allowed!
```

**Right:**
```python
class MyHandler:
    async def handle(self, envelope) -> ModelHandlerOutput:
        return ModelHandlerOutput(events=[response_event])  # Return events
```

Only ORCHESTRATOR nodes may publish events. Handlers return events in their output, and the orchestrator publishes them.

### 4. Wrong Node Type in Contract

**Wrong:**
```yaml
node_type: "EFFECT"  # Missing _GENERIC suffix
```

**Right:**
```yaml
node_type: "EFFECT_GENERIC"  # Always use _GENERIC variants
```

Contract files must use the `_GENERIC` suffix: `EFFECT_GENERIC`, `COMPUTE_GENERIC`, `REDUCER_GENERIC`, `ORCHESTRATOR_GENERIC`.

### 5. Forgetting Container Injection

**Wrong:**
```python
class NodeHelloEffect(NodeEffect):
    def __init__(self):  # Missing container parameter
        super().__init__()
```

**Right:**
```python
class NodeHelloEffect(NodeEffect):
    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
```

All nodes receive dependencies via `ModelONEXContainer`. For simple declarative nodes, you can omit `__init__` entirely and just use `pass`.

### 6. Importing Types Outside TYPE_CHECKING

**Wrong:**
```python
from omnibase_core.models.container import ModelONEXContainer  # Always imported

class MyNode(NodeEffect):
    def __init__(self, container: ModelONEXContainer) -> None: ...
```

**Right:**
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer

class MyNode(NodeEffect):
    def __init__(self, container: ModelONEXContainer) -> None: ...
```

Use `TYPE_CHECKING` blocks to avoid circular imports. The `from __future__ import annotations` enables forward references.

---

## Getting Help

- **Code standards**: See [CLAUDE.md](../../CLAUDE.md) - the **authoritative source** for all coding rules and standards. Documentation in `docs/` provides explanations and examples, but CLAUDE.md defines the rules.
- **Patterns**: See [docs/patterns/](../patterns/README.md) for implementation guides
- **Issues**: Open an issue on GitHub
