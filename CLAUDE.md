# Claude Code Rules for ONEX Infrastructure

**Quick Start**: Essential rules for ONEX development.

**Detailed Patterns**: See `docs/patterns/` for implementation guides:
- `container_dependency_injection.md` - Complete DI patterns
- `error_handling_patterns.md` - Error hierarchy and usage
- `error_recovery_patterns.md` - Backoff, circuit breakers, degradation
- `circuit_breaker_implementation.md` - Circuit breaker details
- `dispatcher_resilience.md` - Dispatcher-owned resilience pattern
- `protocol_patterns.md` - Protocol design patterns, cross-mixin composition, TYPE_CHECKING
- `security_patterns.md` - Node introspection security, input validation, secrets
- `handler_plugin_loader.md` - Plugin-based handler loading, security trade-offs

---

## MANDATORY: Agent-Driven Development

**ALL CODING TASKS MUST USE SUB-AGENTS - NO EXCEPTIONS**

| Task Type | Agent |
|-----------|-------|
| Simple tasks | Direct specialist (`agent-commit`, `agent-testing`, `agent-contract-validator`) |
| Complex workflows | `agent-onex-coordinator` → `agent-workflow-coordinator` |
| Multi-domain | `agent-ticket-manager` for planning, orchestrators for execution |

**Prefer `subagent_type: "polymorphic-agent"`** for ONEX development workflows.

## CRITICAL POLICIES

### No Background Agents
- **NEVER** use `run_in_background: true` for Task tool
- Parallel execution: call multiple Task tools in a **single message**

### No Backwards Compatibility

**ALL changes are breaking changes. NO backwards compatibility is maintained.**

- Breaking changes are **always** acceptable and encouraged
- Remove old patterns **immediately** - do not leave deprecated code
- **NO** backwards compatibility documentation required
- **NO** migration guides needed
- **NO** deprecation periods - old APIs are simply removed
- Downstream consumers are expected to update immediately
- Version bumps may contain any breaking change without warning

This policy applies to:
- API changes (function signatures, class interfaces)
- Model changes (Pydantic field additions/removals/renames)
- Import path changes (module reorganization)
- Type changes (type aliases, generics)
- Configuration changes (environment variables, settings)

### No Versioned Directories
- **NEVER** create `v1_0_0/`, `v2/` directories
- Version through `contract.yaml` fields only

## MANDATORY: Declarative Nodes

**ALL nodes MUST be declarative - no custom Python logic in node.py**

```python
# CORRECT - Declarative node (extends base, no custom logic)
from omnibase_core.nodes import NodeOrchestrator

class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Declarative orchestrator - all behavior defined in contract.yaml."""
    pass  # No custom code - driven entirely by contract
```

**Declarative Pattern Requirements:**
1. Extend base class from `omnibase_core.nodes`
2. Use `container: ModelONEXContainer` for dependency injection
3. Define all behavior in `contract.yaml` (handlers, routing, workflows)
4. `node.py` contains ONLY the class definition extending base - no custom logic

**Contract-Driven Handler Routing:**
```yaml
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model: "ModelNodeIntrospectionEvent"
      handler_class: "HandlerNodeIntrospected"
```

## Core ONEX Principles

### Strong Typing & Models
- **NEVER use `Any`** - Use `object` for generic payloads
- **Pydantic Models** - All data structures must be proper Pydantic models
- **One model per file** - Each file contains exactly one `Model*` class
- **PEP 604 unions** - Use `X | None` not `Optional[X]`

### Any Type CI Enforcement

The `Any` type policy is enforced via pre-commit hook and CI check (`scripts/validate.py any_types`).

**Enforcement Levels:**
1. **Pre-commit Hook**: Runs `poetry run python scripts/validate.py any_types` before commit
2. **CI Pipeline**: Runs as part of `ONEX Validators` job - blocks merge on violations (non-zero exit)

| Context | Allowed | Enforcement |
|---------|---------|-------------|
| Function parameters | NO - use `object` | CI BLOCKED |
| Return types | NO - use `object` | CI BLOCKED |
| Pydantic `Field()` with `NOTE:` comment | YES | CI ALLOWED |
| Pydantic `Field()` without `NOTE:` comment | NO | CI BLOCKED |
| Variables, type aliases | NO | CI BLOCKED |

**Pydantic Workaround** (only when technically required):
```python
from typing import Any
from pydantic import Field

class MyModel(BaseModel):
    # NOTE: Any required for Pydantic discriminated union - see ADR
    payload: Any = Field(...)
```

**Exemption Mechanisms**:
- `@allow_any` decorator with documented reason
- `ONEX_EXCLUDE: any_type` inline comment (use sparingly)

**Related**: `docs/decisions/adr-any-type-pydantic-workaround.md`

### Any Type Validator Detection and Limitations

The AST-based Any type validator scans Python source files to detect `Any` usage.

**What IS Detected** (will trigger violations):
- Direct `Any` annotations: `def foo(x: Any) -> Any`
- `Any` in Pydantic fields: `field: Any = Field(...)`
- `Any` as generic argument: `list[Any]`, `dict[str, Any]`, `Callable[..., Any]`
- Type aliases with `Any`: `MyType = dict[str, Any]`
- String annotations: `from __future__ import annotations` - these ARE correctly resolved by the AST parser

**What is NOT Detected** (validator limitations):
1. **External type aliases**: When `Any` is hidden behind an imported alias:
   ```python
   from external_lib import DynamicType  # If DynamicType = Any, NOT detected
   def process(data: DynamicType): ...   # Violation NOT caught
   ```

2. **Runtime type construction**: Factory patterns creating types dynamically:
   ```python
   def make_type() -> type:
       return Any  # NOT detected - evaluated at runtime
   DynamicType = make_type()
   ```

3. **Indirect imports**: `Any` re-exported through intermediate modules may not be traced.

**Example @allow_any Usage**:
```python
from omnibase_infra.decorators import allow_any

@allow_any("Required for legacy API compatibility - see OMN-1234")
def legacy_handler(data: Any) -> Any:
    """Handle legacy API payloads."""
    return process_legacy(data)
```

The `@allow_any` decorator is recognized by the validator when applied to functions or classes. The decorator is a no-op at runtime - it only serves as an AST marker for the validator to skip the decorated definition.

### File & Class Naming

| Type | File Pattern | Class Pattern |
|------|-------------|---------------|
| Adapter | `adapter_<name>.py` | `Adapter<Name>` |
| Constants | `constants_<name>.py` | (constants) |
| Dispatcher | `dispatcher_<name>.py` | `Dispatcher<Name>` |
| Enum | `enum_<name>.py` | `Enum<Name>` |
| Error | In `errors/` | `<Domain><Type>Error` |
| Mixin | `mixin_<name>.py` | `Mixin<Name>` |
| Model | `model_<name>.py` | `Model<Name>` |
| Node | `node.py` | `Node<Name><Type>` |
| Plugin | `plugin_<name>.py` | `Plugin<Name>` |
| Protocol | `protocol_<name>.py` or `protocols.py` | `Protocol<Name>` |
| Reducer | `reducer_<name>.py` | `Reducer<Name>` |
| Service | `service_<name>.py` | `Service<Name>` |
| Store | `store_<name>.py` | `Store<Name>` |
| Transport | `transport_<name>.py` | `Transport<Name>` |
| Util | `util_<name>.py` | (functions) |
| Validator | `validator_<name>.py` | `Validator<Name>` |

**Protocol File Naming**:
- Single protocol: `protocol_<name>.py`
- Domain-grouped: `protocols.py` when protocols are tightly coupled

### Enum Usage: Message Routing vs Node Validation

| Enum | Values | Purpose |
|------|--------|---------|
| `EnumMessageCategory` | `EVENT`, `COMMAND`, `INTENT` | Message routing |
| `EnumNodeOutputType` | `EVENT`, `COMMAND`, `INTENT`, `PROJECTION` | Node validation |

**Key Rule**: `PROJECTION` exists only in `EnumNodeOutputType` and is only valid for REDUCER nodes.

```python
# MESSAGE ROUTING
from omnibase_infra.enums import EnumMessageCategory
category = EnumMessageCategory.EVENT  # For dispatcher selection

# NODE VALIDATION
from omnibase_infra.enums import EnumNodeOutputType
output_type.is_routable()  # False for PROJECTION
```

**Related**: `docs/decisions/adr-enum-message-category-vs-node-output-type.md`

### Registry Naming Conventions

**Node-Specific**: `nodes/<name>/registry/registry_infra_<node_name>.py` → `RegistryInfra<NodeName>`

**Standalone**: `registry_<purpose>.py` → `Registry<Purpose>`

### Custom `__bool__` for Result Models

Result models may override `__bool__` to enable idiomatic conditional checks. This differs from standard Pydantic behavior where `bool(model)` always returns `True`.

**Related**: `docs/decisions/adr-custom-bool-result-models.md`

**Categories of implementations**:

| Category | Models | Condition |
|----------|--------|-----------|
| Validity/Success | `ModelSecurityValidationResult`, `ModelValidationOutcome`, `ModelLifecycleResult` | Returns `True` when `valid`/`success`/`is_valid` is True |
| Collection-Based | `ModelReducerExecutionResult`, `ModelDispatchOutputs` | Returns `True` when intents/topics non-empty |
| Optional Wrappers | `ModelOptionalString`, `ModelOptionalUUID`, `ModelOptionalCorrelationId` | Returns `True` when value present |
| Matching Results | `ModelCategoryMatchResult`, `ModelExecutionShapeValidationResult` | Returns `True` when `matched`/`passed` is True |

**Usage pattern**:
```python
result = reducer.reduce(state, event)
if result:  # True only if there are intents to process
    execute_intents(result.intents)
```

**Documentation requirement**: Always include a `Warning` section in the `__bool__` docstring explaining the non-standard behavior.

### Type Annotation Conventions

**Nullable Types**: Use `X | None` (PEP 604), not `Optional[X]`
```python
def get_user(id: str) -> User | None: ...
```

**Envelope Typing**: Use `ModelEventEnvelope[object]` for generic dispatchers
```python
async def process_event(envelope: ModelEventEnvelope[object]) -> str | None:
    """Process any event type - uses object for generic payloads."""
```

**Type Alias Pattern**: Use underscore-prefixed unions for Pydantic, protocols for type hints
```python
_IntentUnion = ModelCommandIntent | ModelEventIntent  # Pydantic validation
def process(intent: ProtocolRegistrationIntent): ...  # Function signature
```

### JsonType Usage

**`JsonType` is the canonical type alias for JSON-compatible values** (from `omnibase_core.types`).

**Definition**: `JsonType = str | int | float | bool | None | list[JsonType] | dict[str, JsonType]`

**Import Pattern**:
```python
# Preferred: Import from omnibase_core
from omnibase_core.types import JsonType

# Also available: Import from omnibase_infra (re-exports)
from omnibase_infra.models.types import JsonType
```

**Related Type Aliases**:
| Type | Purpose | Use Case |
|------|---------|----------|
| `JsonType` | Recursive JSON union | Generic JSON values, configs, payloads |
| `JsonPrimitive` | Atomic JSON values | When only primitives needed (no containers) |
| `JsonDict` | `dict[str, object]` | When `.get()` or key access is needed |

**When to Use**:
```python
# For generic JSON data structures
def process_config(config: JsonType) -> None: ...

# For typed dict access with .get() method
def parse_payload(data: JsonDict) -> str:
    return str(data.get("key", "default"))

# For return types with JSON serialization
async def fetch_data() -> JsonType:
    return {"status": "ok", "items": [1, 2, 3]}
```

### Intent Model Architecture

**Overview**: Reducers emit intents that orchestrators route to Effect layer nodes. The implementation uses typed payload models that extend `ModelIntentPayloadBase`.

**Extension-Type Intent Pattern**: All infrastructure intents use `intent_type="extension"` at the outer `ModelIntent` level. The actual routing key is in `payload.intent_type` (e.g., `"consul.register"`, `"postgres.upsert_registration"`). This two-layer approach enables generic routing while preserving type-safe payloads.

**Two-Layer Intent Structure**:

| Layer | Model | Purpose |
|-------|-------|---------|
| 1. Typed Payload | `ModelPayloadConsulRegister` | Domain-specific Pydantic model with typed fields and `intent_type` |
| 2. Outer Container | `ModelIntent` | Standard intent envelope with `intent_type="extension"` |

**Defining Typed Payload Models** (in `nodes/reducers/models/`):
```python
from omnibase_core.models.reducer.payloads import ModelIntentPayloadBase
from pydantic import Field
from typing import Literal
from uuid import UUID

class ModelPayloadConsulRegister(ModelIntentPayloadBase):
    """Typed payload for Consul service registration."""
    intent_type: Literal["consul.register"] = Field(default="consul.register")
    correlation_id: UUID
    service_id: str
    service_name: str
    tags: list[str]
    health_check: dict[str, str] | None = None
```

**Building Intents in Reducers**:
```python
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.nodes.reducers.models import ModelPayloadConsulRegister

# Build typed payload with domain data
consul_payload = ModelPayloadConsulRegister(
    correlation_id=correlation_id,
    service_id=f"onex-{node_type}-{node_id}",
    service_name=f"onex-{node_type}",
    tags=["node_type:effect"],
)

# Return as ModelIntent from reducer
return ModelIntent(
    intent_type="extension",
    target=f"consul://service/{service_name}",
    payload=consul_payload,
)
```

**Intent Type Routing**:
- `ModelIntent.intent_type` is always `"extension"` for extension-based intents
- `payload.intent_type` contains the specific routing key (e.g., `"consul.register"`, `"postgres.upsert_registration"`)
- Effect layer routes based on `payload.intent_type`

**Target URI Convention**:
- Format: `{protocol}://{resource}/{identifier}`
- Examples: `postgres://node_registrations/{node_id}`, `consul://service/{service_name}`

**Serialization with Nested Models**:

When payload contains complex nested models (e.g., `ModelNodeRegistrationRecord`), use `SerializeAsAny`:
```python
from pydantic import BaseModel, Field, SerializeAsAny

class ModelPayloadPostgresUpsertRegistration(ModelIntentPayloadBase):
    """Typed payload for PostgreSQL upsert operations."""
    intent_type: Literal["postgres.upsert_registration"] = Field(
        default="postgres.upsert_registration"
    )
    correlation_id: UUID
    record: SerializeAsAny[BaseModel]  # Preserves subclass fields during serialization
```

**Why `SerializeAsAny`**: When a field is typed as `BaseModel` but contains a subclass, Pydantic only serializes base fields without this type wrapper.

**Accessing Payload Fields**:

Intent payloads use **direct typed field access** - no `.data` dict wrapper:
```python
# Typed intent payloads - direct field access
if isinstance(intent.payload, ModelPayloadConsulRegister):
    service_name = intent.payload.service_name
    tags = intent.payload.tags

# Handler response payloads - use .data for inner payload
result = await handler.handle(envelope)
payload_data = result.result.payload.data  # Access handler-specific data
```

**Key Distinction**:
| Context | Access Pattern | Example |
|---------|---------------|---------|
| Intent Payloads | Direct fields | `intent.payload.service_name` |
| Handler Responses | Via `.data` field | `result.result.payload.data.operation_type` |

**Effect Layer Routing**:

Effect nodes use `payload.intent_type` to route intents to appropriate handlers:
```python
from omnibase_infra.nodes.reducers.models import (
    ModelPayloadConsulRegister,
    ModelPayloadPostgresUpsertRegistration,
)

routing_table = {
    "consul.register": ConsulAdapter.handle_register,
    "postgres.upsert_registration": PostgresAdapter.handle_upsert,
}

async def route_intent(intent: ModelIntent) -> None:
    if intent.intent_type == "extension" and hasattr(intent.payload, "intent_type"):
        handler = routing_table.get(intent.payload.intent_type)
        if handler:
            await handler(intent.payload)
        else:
            raise ValueError(f"Unknown intent_type: {intent.payload.intent_type}")
```

**Reference Implementation**: `src/omnibase_infra/nodes/reducers/registration_reducer.py`
**Payload Models**: `src/omnibase_infra/nodes/reducers/models/model_payload_*.py`

**Envelope-Based Handler Routing**:

Infrastructure handlers (`HandlerConsul`, `HandlerDb`, etc.) work with envelope-based operation routing, NOT raw `ModelIntent` objects. Routing is based on `envelope["operation"]` field values, not `intent_type`.

When reducers emit intents with:
- `intent_type="extension"`
- `payload.intent_type="consul.register"` (or `"postgres.upsert_registration"`)

The Orchestrator/Runtime layer translates these to envelope format:
```python
# Consul example
{
    "operation": "consul.register",
    "payload": {...},  # Consul-specific data from intent.payload
    "correlation_id": "...",
}

# PostgreSQL example
{
    "operation": "db.execute",
    "payload": {
        "sql": "...",
        "parameters": [...],
    },
    "correlation_id": "...",
}
```

This design keeps infrastructure handlers decoupled from the intent format, allowing them to be reused for:
- Direct envelope-based invocation (CLI tools, direct API calls)
- Intent-driven workflows (via orchestrator translation)

**Handler Implementations**: `src/omnibase_infra/handlers/handler_consul.py`, `src/omnibase_infra/handlers/handler_db.py`

### ONEX Architecture
- **Contract-Driven** - All tools/services follow contract patterns
- **Container Injection** - `def __init__(self, container: ModelONEXContainer)`
- **Protocol Resolution** - Duck typing through protocols, never isinstance
- **OnexError Only** - `raise OnexError(...) from e`

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

### Container-Based Dependency Injection

**All services MUST use ModelONEXContainer for dependency injection.**

```python
from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes import NodeOrchestrator

class MyOrchestrator(NodeOrchestrator):
    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
```

## Infrastructure Error Patterns

### Error Class Selection

| Scenario | Error Class |
|----------|-------------|
| Config invalid | `ProtocolConfigurationError` |
| Connection failed | `InfraConnectionError` |
| Timeout | `InfraTimeoutError` |
| Auth failed | `InfraAuthenticationError` |
| Unavailable | `InfraUnavailableError` |

### Error Context

**MANDATORY**: Use `with_correlation()` factory method for error context creation.

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

# Auto-generate correlation_id (new error, no existing ID)
context = ModelInfraErrorContext.with_correlation(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
)

# Propagate existing correlation_id (preserve trace chain)
context = ModelInfraErrorContext.with_correlation(
    correlation_id=request.correlation_id,
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
)

raise InfraConnectionError("Failed to connect", context=context) from e
```

**See**: `docs/decisions/adr-error-context-factory-pattern.md`

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

## Infrastructure Patterns

### Correlation ID Rules
1. Always propagate from incoming requests
2. Auto-generate with `uuid4()` if missing
3. Include in all error context

### Circuit Breaker

Use `MixinAsyncCircuitBreaker` for external service integrations:
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

**See**: `docs/patterns/circuit_breaker_implementation.md` for full implementation details
**See**: `docs/patterns/mixin_dependencies.md` for mixin composition patterns and dependency requirements

### Transport Types
| Type | Value |
|------|-------|
| `DATABASE` | `"db"` |
| `KAFKA` | `"kafka"` |
| `HTTP` | `"http"` |
| `CONSUL` | `"consul"` |
| `VAULT` | `"vault"` |
| `VALKEY` | `"valkey"` |

### Dispatcher Resilience

**Dispatchers own their own resilience** - the `MessageDispatchEngine` does NOT wrap dispatchers with circuit breakers.

Each dispatcher should:
- Implement `MixinAsyncCircuitBreaker` for external service calls
- Configure thresholds appropriate to their transport type
- Raise `InfraUnavailableError` when circuit opens

**See**: `docs/patterns/dispatcher_resilience.md` for full pattern details
**See**: `docs/patterns/mixin_dependencies.md` for mixin composition and inheritance order

### Handler No-Publish Constraint

**Handlers MUST NOT have direct event bus access** - only orchestrators may publish events.

| Constraint | Verification |
|------------|--------------|
| No bus parameters | `__init__`, `handle()` signatures |
| No bus attributes | No `_bus`, `_event_bus`, `_publisher` |
| No publish methods | No `publish()`, `emit()`, `send_event()` |
| Protocol compliance | `ProtocolHandler` has no bus methods |

**Integration Tests**: `tests/integration/handlers/test_handler_no_publish_constraint.py`

| Test Class | Coverage |
|------------|----------|
| `TestHttpRestHandlerBusIsolation` | HTTP handler bus isolation |
| `TestHandlerNodeIntrospectedBusIsolation` | Introspection handler isolation |
| `TestHandlerProtocolCompliance` | Protocol-level constraints |
| `TestOrchestratorBusAccessVerification` | Orchestrator-only bus access |
| `TestHandlerNoPublishConstraintCrossValidation` | Cross-handler constraint validation |

### Node Introspection Security

The `MixinNodeIntrospection` mixin uses Python reflection for service discovery. This has security implications:

**What Gets Exposed**: Public method names, signatures, protocol implementations, FSM state
**What is NOT Exposed**: Private methods (`_` prefix), source code, configuration values, secrets

**Built-in Protections**:
- Private method exclusion
- Utility method filtering (`get_*`, `set_*`, etc.)
- Operation keyword matching

**Best Practices**:
- Prefix internal/sensitive methods with `_`
- Use generic parameter names (e.g., `data` not `user_credentials`)
- Configure Kafka topic ACLs for introspection topics

**See**: `docs/patterns/security_patterns.md#introspection-security` for complete threat model and deployment checklist
**See**: `docs/patterns/mixin_dependencies.md#mixinnodeintrospection` for initialization requirements

### Handler Plugin Loader Patterns

The runtime uses a **plugin-based handler loading** system instead of hardcoded registries. Handlers are discovered and loaded dynamically from YAML contracts.

**Why Plugin Pattern Over Hardcoded Registry:**

| Approach | Trade-off |
|----------|-----------|
| Hardcoded registry | Compile-time safety, but tight coupling and requires code changes to add handlers |
| Plugin pattern | Loose coupling, runtime discovery, but requires path validation |

**Benefits of Plugin Architecture:**
- **Contract-driven**: Handler configuration lives in `contract.yaml`, not Python code
- **Loose coupling**: Orchestrators don't import handler modules directly
- **Runtime discovery**: New handlers can be added without modifying orchestrator code
- **Testability**: Mock handlers can be injected via contract configuration

**Why YAML Contracts Over Python Decorators:**

| Aspect | YAML Contracts | Python Decorators |
|--------|---------------|-------------------|
| Tooling | Machine-readable, lintable, diffable | Requires AST parsing |
| Auditability | Changes visible in git, reviewable | Scattered across codebase |
| Non-Python access | CI/CD, dashboards can read contracts | Requires Python runtime |
| Separation of concerns | Configuration separate from logic | Mixes config with code |

**Contract-Based Handler Declaration:**
```yaml
# contract.yaml
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model: "ModelNodeIntrospectionEvent"
      handler_class: "HandlerNodeIntrospected"
      handler_module: "omnibase_infra.handlers.handler_node_introspected"
```

**Handler Resolution Precedence:**

The loader recognizes two contract file names:

| Filename | Purpose |
|----------|---------|
| `handler_contract.yaml` | Dedicated handler contract (preferred) |
| `contract.yaml` | General ONEX contract with handler fields |

**FAIL-FAST: Ambiguous Contract Configuration**

When **both** `handler_contract.yaml` **and** `contract.yaml` exist in the **same directory**, the loader **raises an error** (error code: `AMBIGUOUS_CONTRACT_CONFIGURATION` / `HANDLER_LOADER_040`). This fail-fast behavior prevents:

- Duplicate handler registrations if both files define similar handlers
- Confusion about which contract is the "source of truth"
- Unexpected runtime behavior if handlers conflict

**Error raised when ambiguous configuration detected:**
```
ProtocolConfigurationError: Ambiguous contract configuration in 'auth':
Found both 'handler_contract.yaml' and 'contract.yaml'.
Use only ONE contract file per handler directory to avoid conflicts.
```

**Best Practice**: Use only **ONE** contract file per handler directory.

```
# CORRECT: One contract per directory
nodes/auth/
    handler_contract.yaml   # Preferred: dedicated handler contract
    handler_auth.py

# INCORRECT: Both contract types in same directory (raises error)
nodes/auth/
    handler_contract.yaml   # Conflict detected!
    contract.yaml          # ProtocolConfigurationError raised
```

**Why Fail-Fast Over Warning**: The loader raises an error instead of warning because:
1. **Explicit is better than implicit**: Silent loading of both could mask configuration errors
2. **Fail-fast philosophy**: Early error detection prevents production issues
3. **No assumptions**: The loader cannot know which file the user intends to be authoritative

**See**: `docs/patterns/handler_plugin_loader.md#contract-file-precedence` for full resolution rules.

**Security Model:**

**CRITICAL**: YAML contracts are treated as **executable code**, not mere configuration. Dynamic imports via `importlib.import_module()` execute module-level code during import.

| Risk | Status | Description |
|------|--------|-------------|
| YAML deserialization attacks | **MITIGATED** | `yaml.safe_load()` blocks `!!python/object` tags |
| Memory exhaustion | **MITIGATED** | 10MB file size limit enforced |
| Arbitrary class loading | **MITIGATED** | Protocol validation requires 5 `ProtocolHandler` methods |
| Arbitrary code execution | **OPTIONALLY MITIGATED** | Enable `allowed_namespaces` to restrict imports |
| Path traversal in module paths | **OPTIONALLY MITIGATED** | Enable `allowed_namespaces` to restrict imports |
| Untrusted namespace imports | **OPTIONALLY MITIGATED** | Use `allowed_namespaces` parameter |

**Built-in Security Controls** (implemented in loader):
- **YAML safe loading**: `yaml.safe_load()` prevents deserialization attacks
- **File size limits**: Contracts exceeding 10MB are rejected
- **Protocol validation**: Classes must implement all 5 `ProtocolHandler` methods
- **Error containment**: Single bad contract doesn't crash the system
- **Correlation tracking**: All load operations logged with UUID correlation IDs
- **Namespace allowlisting** (optional): Use `allowed_namespaces` constructor parameter to restrict imports

**Optional Security Control - Namespace Allowlisting:**
```python
# Restrict to trusted namespaces only (recommended for production)
loader = HandlerPluginLoader(
    allowed_namespaces=["omnibase_infra.", "omnibase_core.", "myapp.handlers."]
)
# Untrusted modules will fail with NAMESPACE_NOT_ALLOWED (HANDLER_LOADER_013)
```

**NOT Implemented** (deployment-level controls):
- Import hook filtering (custom `MetaPathFinder`)
- Runtime isolation (subprocess/container isolation)

**Secure Deployment Checklist:**
1. **File permissions**: Contract directories readable only by runtime user
2. **Write protection**: Mount contract directories as read-only at runtime
3. **Source validation**: Contracts from version-controlled, reviewed sources only
4. **Namespace allowlisting**: Enable `allowed_namespaces` parameter (recommended for production)
5. **Audit logging**: Enable INFO-level logging for handler loader
6. **Contract validation**: Run `onex validate` in CI to catch malformed contracts

**Error Codes:**
- `MODULE_NOT_FOUND` (HANDLER_LOADER_010) - Handler module not found
- `CLASS_NOT_FOUND` (HANDLER_LOADER_011) - Class not found in module
- `IMPORT_ERROR` (HANDLER_LOADER_012) - Module import failed (syntax/dependency)
- `NAMESPACE_NOT_ALLOWED` (HANDLER_LOADER_013) - Handler module namespace not in allowlist
- `AMBIGUOUS_CONTRACT_CONFIGURATION` (HANDLER_LOADER_040) - Both contract types in same directory
- `PROTOCOL_NOT_IMPLEMENTED` (HANDLER_LOADER_006) - Class missing required `ProtocolHandler` methods

**See**: `docs/patterns/handler_plugin_loader.md` and `docs/decisions/adr-handler-plugin-loader-security.md` for complete security documentation.

### Service Integration Architecture
- **Adapter Pattern** - External services wrapped in ONEX adapters
- **Connection Pooling** - Database connections managed through pool managers
- **Event-Driven Communication** - Infrastructure events flow through Kafka adapters
- **Service Discovery** - Consul integration for dynamic service resolution
- **Secret Management** - Vault integration for secure credential handling

### Infrastructure 4-Node Pattern
- **EFFECT** - External service interactions (adapters)
- **COMPUTE** - Message processing and transformation
- **REDUCER** - State consolidation and decision making
- **ORCHESTRATOR** - Workflow coordination

## Node Structure

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

## Agent Architecture

| Category | Agents |
|----------|--------|
| Orchestration | `agent-onex-coordinator`, `agent-workflow-coordinator` |
| Development | `agent-contract-validator`, `agent-commit`, `agent-testing` |
| DevOps | `agent-devops-infrastructure`, `agent-security-audit` |
| Quality | `agent-pr-review`, `agent-pr-create` |

## Zero Tolerance

- `Any` types forbidden
- Direct coding without agent delegation
- Hand-written Pydantic models (must be contract-generated)
- Hardcoded service configurations
- Imperative nodes with custom routing logic

## Service Ports

| Service | Port |
|---------|------|
| Event Bus | 8083 |
| Consul | 8500 |
| Kafka | 9092 |
| Vault | 8200 |
| PostgreSQL | 5432 |

---

**Bottom Line**: Declarative nodes, container injection, agent-driven development. No backwards compatibility, no custom node logic.
