# Claude Code Rules for ONEX Infrastructure

**Quick Start**: Essential rules and references for ONEX development.

**Detailed Patterns**: See `docs/patterns/` for implementation guides:
- `container_dependency_injection.md` - Complete DI patterns
- `error_handling_patterns.md` - Error hierarchy and usage
- `error_recovery_patterns.md` - Backoff, circuit breakers, degradation
- `correlation_id_tracking.md` - Request tracing
- `circuit_breaker_implementation.md` - Circuit breaker details

---

## 🚨 MANDATORY: Agent-Driven Development

**ALL CODING TASKS MUST USE SUB-AGENTS - NO EXCEPTIONS**

| Task Type | Agent |
|-----------|-------|
| Simple tasks | Direct specialist (`agent-commit`, `agent-testing`, `agent-contract-validator`) |
| Complex workflows | `agent-onex-coordinator` → `agent-workflow-coordinator` |
| Multi-domain | `agent-ticket-manager` for planning, orchestrators for execution |

## 🚫 CRITICAL POLICY: NO BACKWARDS COMPATIBILITY

- Breaking changes are always acceptable
- No deprecated code maintenance
- Remove old patterns immediately

## 🎯 Core ONEX Principles

### Strong Typing & Models
- **NEVER use `Any`** - Always use specific types
- **Pydantic Models** - All data structures must be proper Pydantic models
- **One model per file** - Each file contains exactly one `Model*` class

### File & Class Naming Conventions

| Type | File Pattern | Class Pattern | Example |
|------|-------------|---------------|---------|
| Model | `model_<name>.py` | `Model<Name>` | `model_kafka_message.py` → `ModelKafkaMessage` |
| Enum | `enum_<name>.py` | `Enum<Name>` | `enum_handler_type.py` → `EnumHandlerType` |
| Protocol | `protocol_<name>.py` | `Protocol<Name>` | `protocol_event_bus.py` → `ProtocolEventBus` |
| Mixin | `mixin_<name>.py` | `Mixin<Name>` | `mixin_health_check.py` → `MixinHealthCheck` |
| Service | `service_<name>.py` | `Service<Name>` | `service_discovery.py` → `ServiceDiscovery` |
| Util | `util_<name>.py` | (functions) | `util_retry.py` → `retry_with_backoff()` |
| Error | In `errors/` | `<Domain><Type>Error` | `InfraConnectionError` |
| Node | `node.py` | `Node<Name><Type>` | `NodePostgresAdapterEffect` |

### Registry Naming Conventions

**Node-Specific Registries** (`nodes/<name>/v<version>/registry/`):
- File: `registry_infra_<node_name>.py`
- Class: `RegistryInfra<NodeName>`
- Examples: `registry_infra_postgres_adapter.py` → `RegistryInfraPostgresAdapter`

**Standalone Registries** (in domain directories):
- File: `registry_<purpose>.py`
- Class: `Registry<Purpose>`
- Examples: `registry_handler.py` → `RegistryHandler`, `registry_policy.py` → `RegistryPolicy`, `registry_compute.py` → `RegistryCompute`

### ONEX Architecture
- **Contract-Driven** - All tools/services follow contract patterns
- **Container Injection** - `def __init__(self, container: ModelONEXContainer)`
- **Protocol Resolution** - Duck typing through protocols, never isinstance
- **OnexError Only** - `raise OnexError(...) from e`

### Container-Based Dependency Injection

**All services MUST use ModelONEXContainer for dependency injection.**

```python
from omnibase_core.container import ModelONEXContainer
from omnibase_infra.runtime.container_wiring import wire_infrastructure_services

# Bootstrap and resolve
container = ModelONEXContainer()
wire_infrastructure_services(container)
service = container.service_registry.resolve_service(ServiceType)
```

**See**: `docs/patterns/container_dependency_injection.md` for complete patterns and examples

## 🚨 Infrastructure Error Patterns

### Error Class Selection (Quick Reference)

| Scenario | Error Class | Transport Code |
|----------|-------------|----------------|
| Config invalid | `ProtocolConfigurationError` | N/A |
| Secret not found | `SecretResolutionError` | N/A |
| Connection failed | `InfraConnectionError` | `DATABASE_CONNECTION_ERROR` / `NETWORK_ERROR` / `SERVICE_UNAVAILABLE` |
| Timeout | `InfraTimeoutError` | Same as connection |
| Auth failed | `InfraAuthenticationError` | Same as connection |
| Unavailable | `InfraUnavailableError` | `SERVICE_UNAVAILABLE` |

### Error Context Example

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext

context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
    target_name="postgresql-primary",
    correlation_id=request.correlation_id,
)
raise InfraConnectionError("Failed to connect", context=context) from original_error
```

**See Also**:
- `docs/patterns/error_handling_patterns.md` - Complete error hierarchy and usage
- `docs/patterns/error_recovery_patterns.md` - Recovery strategies (backoff, circuit breaker, degradation)
- `docs/patterns/correlation_id_tracking.md` - Request tracing patterns
- `docs/patterns/circuit_breaker_implementation.md` - Circuit breaker details

## 🏗️ Infrastructure Architecture

### 4-Node Pattern
- **EFFECT** - External service interactions (Consul, Kafka, Vault adapters)
- **COMPUTE** - Message processing and transformation
- **REDUCER** - State consolidation and decision making
- **ORCHESTRATOR** - Workflow coordination

### Service Adapters
| Adapter | Purpose |
|---------|---------|
| `consul_adapter` | Service discovery |
| `kafka_adapter` | Event streaming |
| `vault_adapter` | Secret management |
| `postgres_adapter` | Database operations |

## 🤖 Agent Architecture

### Orchestration Agents

| Agent | Purpose |
|-------|---------|
| `agent-onex-coordinator` | Primary routing and workflow orchestration |
| `agent-workflow-coordinator` | Multi-step execution, sub-agent fleet coordination |
| `agent-ticket-manager` | Ticket lifecycle, dependency analysis |

### Specialist Agents

| Category | Agents |
|----------|--------|
| Development | `agent-contract-validator`, `agent-contract-driven-generator`, `agent-ast-generator`, `agent-commit` |
| DevOps | `agent-devops-infrastructure`, `agent-security-audit`, `agent-performance`, `agent-production-monitor` |
| Quality | `agent-pr-review`, `agent-pr-create`, `agent-address-pr-comments`, `agent-testing` |
| Intelligence | `agent-research`, `agent-debug-intelligence`, `agent-rag-query`, `agent-rag-update` |

## 🔒 Zero Tolerance Policies

- `Any` types forbidden
- Direct coding without agent delegation prohibited
- Hand-written Pydantic models (must be contract-generated)
- Hardcoded service configurations

## 🔧 DevOps Quick Reference

**Container Troubleshooting**: `docker logs <container>` first, then `docker inspect` for exit codes
- **Exit 0 OK**: Init containers (topic creation, migrations, SSL cert gen)
- **Should run**: Services (web, brokers, databases, load balancers)


## 📦 Service Ports

| Service | Port |
|---------|------|
| Event Bus | 8083 |
| Infrastructure Hub | 8085 |
| Consul | 8500 (HTTP), 8600 (DNS) |
| Kafka | 9092 (plaintext), 9093 (SSL) |
| Vault | 8200 |
| PostgreSQL | 5432 |
| Debug Dashboard | 8096 |

## 🚀 Node Structure Pattern

```
nodes/<adapter>/v1_0_0/
├── contract.yaml          # ONEX contract definition
├── node.py               # Node<Name><Type> implementation
├── models/               # Node-specific models
└── registry/             # registry_infra_<name>.py
```

**Contract Requirements**:
- Semantic versioning (`contract_version`, `node_version`)
- Node type (EFFECT/COMPUTE/REDUCER/ORCHESTRATOR)
- Strongly typed I/O (`input_model`, `output_model`)
- Protocol-based dependencies
- Zero `Any` types, use `omnibase_core.*` imports

---

**Bottom Line**: Agent-driven development. Route through orchestrators, delegate to specialists. Strong typing, contract-driven configuration, no backwards compatibility.
