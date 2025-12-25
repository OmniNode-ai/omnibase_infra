# Claude Code Rules for ONEX Infrastructure

**Quick Start**: Essential rules for ONEX development.

**Detailed Patterns**: See `docs/patterns/` for implementation guides:
- `container_dependency_injection.md` - Complete DI patterns
- `error_handling_patterns.md` - Error hierarchy and usage
- `error_recovery_patterns.md` - Backoff, circuit breakers, degradation
- `circuit_breaker_implementation.md` - Circuit breaker details

---

## 🚨 MANDATORY: Agent-Driven Development

**ALL CODING TASKS MUST USE SUB-AGENTS - NO EXCEPTIONS**

| Task Type | Agent |
|-----------|-------|
| Simple tasks | Direct specialist (`agent-commit`, `agent-testing`, `agent-contract-validator`) |
| Complex workflows | `agent-onex-coordinator` → `agent-workflow-coordinator` |
| Multi-domain | `agent-ticket-manager` for planning, orchestrators for execution |

**Prefer `subagent_type: "polymorphic-agent"`** for ONEX development workflows.

## 🚫 CRITICAL POLICIES

### No Background Agents
- **NEVER** use `run_in_background: true` for Task tool
- Parallel execution: call multiple Task tools in a **single message**

### No Backwards Compatibility
- Breaking changes are always acceptable
- Remove old patterns immediately

### No Versioned Directories
- **NEVER** create `v1_0_0/`, `v2/` directories
- Version through `contract.yaml` fields only

## 🎯 MANDATORY: Declarative Nodes

**ALL nodes MUST be declarative - no custom Python logic in node.py**

```python
# CORRECT - Declarative node (extends base, no custom logic)
from omnibase_core.nodes import NodeOrchestrator

class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Declarative orchestrator - all behavior defined in contract.yaml."""
    pass  # No custom code - driven entirely by contract
```

```python
# WRONG - Imperative node with custom logic
class NodeRegistrationOrchestrator:
    def __init__(self, projection_reader):  # Direct injection
        self._handlers = {...}  # Manual handler wiring

    async def handle(self, envelope):
        if isinstance(payload, EventA):  # Manual routing
            return self._handler_a.handle(...)
```

**Declarative Pattern Requirements:**
1. Extend base class from `omnibase_core.nodes` (`NodeEffect`, `NodeCompute`, `NodeReducer`, `NodeOrchestrator`)
2. Use `container: ModelONEXContainer` for dependency injection
3. Define all behavior in `contract.yaml` (handlers, routing, workflows)
4. `node.py` contains ONLY the class definition extending base - no custom logic
5. Base class handles routing, wiring, and execution from contract

**Contract-Driven Handler Routing:**
```yaml
# contract.yaml - handler_routing section
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model: "ModelNodeIntrospectionEvent"
      handler_class: "HandlerNodeIntrospected"
    - event_model: "ModelRuntimeTick"
      handler_class: "HandlerRuntimeTick"
```

## 🎯 Core ONEX Principles

### Strong Typing & Models
- **NEVER use `Any`** - Use `object` for generic payloads
- **Pydantic Models** - All data structures must be proper Pydantic models
- **One model per file** - Each file contains exactly one `Model*` class
- **PEP 604 unions** - Use `X | None` not `Optional[X]`

### File & Class Naming

| Type | File Pattern | Class Pattern |
|------|-------------|---------------|
| Model | `model_<name>.py` | `Model<Name>` |
| Enum | `enum_<name>.py` | `Enum<Name>` |
| Protocol | `protocol_<name>.py` | `Protocol<Name>` |
| Mixin | `mixin_<name>.py` | `Mixin<Name>` |
| Node | `node.py` | `Node<Name><Type>` |
| Error | In `errors/` | `<Domain><Type>Error` |

### Container-Based Dependency Injection

**All services MUST use ModelONEXContainer for dependency injection.**

```python
from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes import NodeOrchestrator

class MyOrchestrator(NodeOrchestrator):
    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
        # Dependencies resolved from container, not passed directly
```

### Node Archetypes (from `omnibase_core`)

| Layer | Responsibility |
|-------|---------------|
| `omnibase_core` | Node archetypes, I/O models, enums |
| `omnibase_spi` | Protocol definitions |
| `omnibase_infra` | Infrastructure implementations |

```python
from omnibase_core.nodes import (
    NodeEffect,        # External I/O operations
    NodeCompute,       # Pure transformations
    NodeReducer,       # State aggregation (FSM-driven)
    NodeOrchestrator,  # Workflow coordination
)
```

### Enum Usage

| Enum | Purpose |
|------|---------|
| `EnumMessageCategory` | Message routing (`EVENT`, `COMMAND`, `INTENT`) |
| `EnumNodeOutputType` | Node validation (adds `PROJECTION` for reducers) |

## 🚨 Infrastructure Error Patterns

### Error Class Selection

| Scenario | Error Class |
|----------|-------------|
| Config invalid | `ProtocolConfigurationError` |
| Connection failed | `InfraConnectionError` |
| Timeout | `InfraTimeoutError` |
| Auth failed | `InfraAuthenticationError` |
| Unavailable | `InfraUnavailableError` |

### Error Context
```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext

context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
    correlation_id=request.correlation_id,
)
raise InfraConnectionError("Failed to connect", context=context) from e
```

### Error Hierarchy
```
ModelOnexError (omnibase_core)
└── RuntimeHostError
    ├── ProtocolConfigurationError
    ├── InfraConnectionError (transport-aware codes)
    ├── InfraTimeoutError
    ├── InfraAuthenticationError
    └── InfraUnavailableError
```

### Error Sanitization
**NEVER include**: passwords, API keys, PII, connection strings with credentials
**SAFE to include**: service names, operation names, correlation IDs, ports

## 🏗️ Infrastructure Patterns

### Correlation ID Rules
1. Always propagate from incoming requests
2. Auto-generate with `uuid4()` if missing
3. Include in all error context

### Circuit Breaker (MixinAsyncCircuitBreaker)

Use for external service integrations:
```python
class MyAdapter(MixinAsyncCircuitBreaker):
    def __init__(self, config):
        self._init_circuit_breaker(
            threshold=5, reset_timeout=60.0,
            service_name="my-service",
            transport_type=EnumInfraTransportType.HTTP,
        )

    async def connect(self):
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("connect", correlation_id)
        # ... operation ...
```

**States**: CLOSED → OPEN (after failures) → HALF_OPEN → CLOSED (on success)

**See**: `docs/patterns/circuit_breaker_implementation.md` for full details

### Transport Types
| Type | Value |
|------|-------|
| `DATABASE` | `"db"` |
| `KAFKA` | `"kafka"` |
| `HTTP` | `"http"` |
| `CONSUL` | `"consul"` |
| `VAULT` | `"vault"` |
| `VALKEY` | `"valkey"` |

### Infrastructure 4-Node Pattern
- **EFFECT** - External service interactions (adapters)
- **COMPUTE** - Message processing and transformation
- **REDUCER** - State consolidation and decision making
- **ORCHESTRATOR** - Workflow coordination

## 🚀 Node Structure

**Canonical Structure:**
```
nodes/<adapter>/
├── contract.yaml     # ONEX contract (handlers, routing, version)
├── node.py          # Declarative node extending base class
├── models/          # Node-specific models
└── registry/        # registry_infra_<name>.py
```

**Contract Requirements:**
- Semantic versioning (`contract_version`, `node_version`)
- Node type (EFFECT/COMPUTE/REDUCER/ORCHESTRATOR)
- Strongly typed I/O (`input_model`, `output_model`)
- Handler routing (for orchestrators)
- Zero `Any` types

## 🤖 Agent Architecture

| Category | Agents |
|----------|--------|
| Orchestration | `agent-onex-coordinator`, `agent-workflow-coordinator` |
| Development | `agent-contract-validator`, `agent-commit`, `agent-testing` |
| DevOps | `agent-devops-infrastructure`, `agent-security-audit` |
| Quality | `agent-pr-review`, `agent-pr-create` |

## 🔒 Zero Tolerance

- `Any` types forbidden
- Direct coding without agent delegation
- Hand-written Pydantic models (must be contract-generated)
- Hardcoded service configurations
- Imperative nodes with custom routing logic

## 📦 Service Ports

| Service | Port |
|---------|------|
| Event Bus | 8083 |
| Consul | 8500 |
| Kafka | 9092 |
| Vault | 8200 |
| PostgreSQL | 5432 |

---

**Bottom Line**: Declarative nodes, container injection, agent-driven development. No backwards compatibility, no custom node logic.
