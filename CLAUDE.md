# Claude Code Rules for ONEX Infrastructure

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
from omnibase_infra.runtime.policy_registry import PolicyRegistry

# Bootstrap container
container = ModelONEXContainer()
wire_infrastructure_services(container)

# Resolve services using type interface
policy_registry = container.service_registry.resolve_service(PolicyRegistry)

# Use in class constructors
class MyNode:
    def __init__(self, container: ModelONEXContainer):
        self.policy_registry = container.service_registry.resolve_service(PolicyRegistry)
```

**Service Registration Keys:**

| Service | Interface Type | Scope |
|---------|---------------|-------|
| PolicyRegistry | `PolicyRegistry` | global |

**Deprecated Patterns:**
- ❌ `get_policy_registry()` - Module-level singleton (use container instead)
- ❌ `get_policy_class()` - Singleton-based convenience function
- ❌ `register_policy()` - Singleton-based registration

**Preferred Patterns:**
- ✅ `container.service_registry.resolve_service(PolicyRegistry)` - Type-safe resolution
- ✅ `get_policy_registry_from_container(container)` - Helper function
- ✅ `wire_infrastructure_services(container)` - Bootstrap all services

## 🚨 Infrastructure Error Patterns

### Error Class Selection

| Scenario | Error Class |
|----------|-------------|
| Config invalid | `ProtocolConfigurationError` |
| Secret not found | `SecretResolutionError` |
| Connection failed | `InfraConnectionError` |
| Timeout | `InfraTimeoutError` |
| Auth failed | `InfraAuthenticationError` |
| Unavailable | `InfraUnavailableError` |

### Error Context Usage

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
    target_name="postgresql-primary",
    correlation_id=request.correlation_id,
)
raise InfraConnectionError("Failed to connect", context=context) from original_error
```

### Error Hierarchy

```
ModelOnexError (omnibase_core)
└── RuntimeHostError
    ├── ProtocolConfigurationError
    ├── SecretResolutionError
    ├── InfraConnectionError (transport-aware error codes)
    ├── InfraTimeoutError
    ├── InfraAuthenticationError
    └── InfraUnavailableError
```

### Transport-Aware Error Codes

| Transport | EnumCoreErrorCode |
|-----------|-------------------|
| DATABASE | `DATABASE_CONNECTION_ERROR` |
| HTTP/GRPC | `NETWORK_ERROR` |
| KAFKA/CONSUL/VAULT/REDIS | `SERVICE_UNAVAILABLE` |

### Error Sanitization

**NEVER include**: Passwords, API keys, tokens, PII, connection strings with credentials

**SAFE**: Service names, operation names, correlation IDs, sanitized hostnames, ports

### Error Recovery Patterns

| Pattern | Use For | Implementation |
|---------|---------|----------------|
| Exponential Backoff | `InfraConnectionError` | Retry with 2^attempt delay |
| Circuit Breaker | `InfraUnavailableError` | Track failures, open circuit at threshold |
| Graceful Degradation | `InfraTimeoutError` | Fallback to cache/secondary |
| Credential Refresh | `InfraAuthenticationError` | Auto-refresh tokens on expiry |

### Transport Types

| Type | Value | Usage |
|------|-------|-------|
| HTTP | `"http"` | REST APIs |
| DATABASE | `"db"` | PostgreSQL |
| KAFKA | `"kafka"` | Message broker |
| CONSUL | `"consul"` | Service discovery |
| VAULT | `"vault"` | Secrets |
| REDIS | `"redis"` | Cache |
| GRPC | `"grpc"` | gRPC |

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

## 🔧 DevOps Container Troubleshooting

**MANDATORY WORKFLOW:**

1. **Log Inspection First**: `docker logs <container>` - NO EXCEPTIONS
2. **Analyze**: Exit 0 + success = expected; Non-zero + errors = failure
3. **Verify**: `docker inspect <container> --format='{{.State.ExitCode}}'`
4. **Document**: Evidence-based findings only

**Expected Exit 0**: Topic creation, migrations, SSL cert generation, config init

**Should Stay Running**: Web services, brokers, databases, load balancers

## 📊 MCP Integration

| Server | Use For |
|--------|---------|
| Context7 | Documentation, patterns |
| Sequential | Architecture analysis |
| Archon | Task orchestration |
| Playwright | E2E testing |

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

## 🚀 Infrastructure Migration Plan

### Phase 1: Adapter Node Structure

Each adapter follows the message bus bridge pattern:

```
nodes/<adapter>/v1_0_0/
├── contract.yaml
├── node.py (NodeEffectService)
├── models/
│   ├── model_<adapter>_input.py
│   └── model_<adapter>_output.py
└── registry/
    └── registry_infra_<adapter>.py
```

### Shared Models (DRY)

```
models/
├── postgres/
│   ├── model_postgres_query_request.py
│   ├── model_postgres_transaction_request.py
│   └── model_postgres_health_response.py
├── consul/
│   ├── model_consul_kv_request.py
│   └── model_consul_service_registration.py
├── kafka/
│   └── model_kafka_message.py
└── vault/
    └── model_vault_secret_request.py
```

### Migration Priority

1. `consul_adapter` - Service discovery foundation
2. `kafka_adapter` - Event streaming backbone
3. `vault_adapter` - Secret management
4. `postgres_adapter` - Database operations
5. `infrastructure_reducer` - State consolidation
6. `infrastructure_orchestrator` - Workflow coordination

### Contract Dependencies Pattern

```yaml
dependencies:
  - name: "model_postgres_query_request"
    type: "model"
    class_name: "ModelPostgresQueryRequest"
    module: "omnibase_infra.models.postgres.model_postgres_query_request"
```

### Required Contract Sections

- `contract_version`, `node_version` - Semantic versioning
- `node_type` - EFFECT/COMPUTE/REDUCER/ORCHESTRATOR
- `input_model`, `output_model` - Strongly typed I/O
- `dependencies` - Protocol-based injection
- `io_operations` - For EFFECT nodes
- `definitions` - All models defined here

### Validation Requirements

- All contracts ONEX compliant
- Zero `Any` types
- All `omnibase.` imports → `omnibase_core.`
- OnexError chaining with CoreErrorCode
- Protocol-based resolution (no isinstance)

---

**Bottom Line**: Agent-driven development. Route through orchestrators, delegate to specialists. Strong typing, contract-driven configuration, no backwards compatibility.
