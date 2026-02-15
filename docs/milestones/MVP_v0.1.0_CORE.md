> **Navigation**: [Home](../index.md) > [Milestones](README.md) > MVP Core v0.1.0
>
> **Related**: [Next: Beta Hardening](./BETA_v0.2.0_HARDENING.md)

# MVP Core (v0.1.0) - Milestone Details

**Repository**: omnibase_infra
**Target Version**: v0.1.0
**Timeline**: Sprint 1-2
**Issue Count**: 24

---

## MVP Philosophy

**MVP (v0.1.0)**: Prove the architecture works end-to-end with minimal scope
- InMemoryEventBus only (no Kafka complexity)
- HTTP + DB handlers only (no Vault, no Consul)
- Simplified contract format
- Basic error handling
- Unit tests with mocks

**Terminology Clarification**:
- **"Minimal"** = Temporary scaffolding with known limitations. Will be expanded in Beta.
- **"Simple"** = Intentionally small feature set that may remain unchanged.
- When an issue says "MINIMAL", expect it to grow. When it says "SIMPLE", expect it to stay small.

---

## MVP Scope Boundaries (Do Not Implement)

The following are explicitly OUT OF SCOPE for MVP (v0.1.0):

- **No Vault, Consul, Kafka** - InMemoryEventBus only
- **No retries** - Single attempt per handler invocation
- **No rate limits** - No throttling in MVP
- **No multi-tenant isolation** - tenant_id deferred
- **No secrets resolution** - Direct config only
- **No graceful shutdown drain** - Basic shutdown only

**Explicitly Forbidden in MVP**:
- Multiple handlers of the same type (one HttpHandler, one HandlerDb only)
- Multi-topic subscription (single input topic, single output topic)
- Async background tasks in ANY MVP code
- Dynamic handler registration at runtime
- Handler hot-reload or reconfiguration
- Transactions in HandlerDb (individual queries only)

**Operations MUST Error Loudly**:
- Attempting to register duplicate handler types -> `ProtocolConfigurationError`
- Attempting forbidden operations (e.g., `db.transaction`) -> `InvalidOperationError`
- Missing required envelope fields -> `EnvelopeValidationError`

---

## Simplified Contract Format (MVP)

MVP uses a minimal contract format:

> **Stability Warning**: The MVP contract schema is intentionally incomplete and NOT considered stable. Breaking changes expected through v0.2.

```yaml
runtime:
  name: "my_runtime_host"
  version: "1.0.0"

  event_bus:
    kind: "inmemory"  # "kafka" in Beta

  handlers:
    - type: "http"
    - type: "db"

  nodes:
    - slug: "node_a"
    - slug: "node_b"
```

**MVP Contract Constraints**:
- **Handler uniqueness**: MVP supports one instance per handler type; multi-binding is added in Beta.
- **Operation format**: The `operation` field uses dotted-path notation (e.g., `http.get`, `db.query`). Enumeration of valid operations is handler-specific.

**MVP Prohibited Fields** (contract importer MUST reject):
- `tenant_id` - Reserved for Beta multi-tenancy
- `retry_policy` - Reserved for Beta resilience
- `rate_limit` - Reserved for Beta throttling
- `handlers.type: "local"` - LocalHandler forbidden in infra contracts
- `handlers.type: "vault"` - HandlerVault deferred to Beta
- `handlers.type: "consul"` - HandlerConsul deferred to Beta

---

## MVP Constraints Checklist

**If these 12 things work, MVP is guaranteed to run end-to-end.**

| # | Component | Validation | Pass? |
|---|-----------|------------|-------|
| 1 | `ModelOnexEnvelope` | Serializes to/from JSON without data loss | [ ] |
| 2 | `ModelRuntimeHostContract` | Loads from YAML, validates fields | [ ] |
| 3 | `ProtocolHandler` | Interface defined with `handler_type` returning enum | [ ] |
| 4 | `ProtocolEventBus` | Interface defined with `publish_envelope`, `subscribe` | [ ] |
| 5 | `NodeRuntime` | Routes envelope to correct handler by type | [ ] |
| 6 | `HandlerContractSource` | Discovers handler_contract.yaml files from filesystem | [ ] |
| 7 | `HttpHandler` | Executes GET/POST, returns response envelope | [ ] |
| 8 | `HandlerDb` | Executes query/execute, returns response envelope | [ ] |
| 9 | `InMemoryEventBus` | Publishes and consumes envelopes per topic | [ ] |
| 10 | `BaseRuntimeHostProcess` | Loads contract, wires handlers, starts consumption | [ ] |
| 11 | `wiring.py` | Registers handlers from config without duplicates | [ ] |
| 12 | E2E Flow | Envelope in -> handler executes -> response out | [ ] |

> **Note**: The checklist references core components. `omnibase_infra` provides `HandlerContractSource`
> for handler contract discovery from `handler_contract.yaml` files. `omnibase_core` will provide a general-purpose
> contract registry (see Issue 1.8) for loading node contracts from filesystem. These are complementary components:
> `HandlerContractSource` handles handler-specific discovery, while the core registry handles general contract loading.

**MVP Smoke Test** (single command to verify):
```bash
# This command MUST succeed for MVP to be complete
pytest tests/integration/test_e2e_envelope_flow.py -v
```

**Minimum Viable Configuration**:
- Event Bus: `InMemoryEventBus` only
- Handlers: `HttpHandler` + `HandlerDb` only
- Nodes: At least 1 node contract loaded
- Contract: Simplified MVP schema (no retry_policy, no tenant_id)

---

## Phase 0: Cross-Repo Guardrails & Invariants (MVP Issues)

**Repository**: All (CI Infrastructure)

### Blocking External Dependencies

> **CRITICAL**: `omnibase_infra` v0.1.0 MUST NOT begin implementation until these dependencies are met:

| Dependency | Repository | Version | Status |
|------------|------------|---------|--------|
| Core release | omnibase_core | v0.4.0 | **BLOCKING** |
| SPI release | omnibase_spi | v0.3.0 | **BLOCKING** |

**Release Sequencing Rule**: `omnibase_core` v0.4.0 MUST be cut and released BEFORE `omnibase_infra` v0.1.0 Runtime Host implementation begins. No exceptions. This ensures:
1. Declarative nodes are stable
2. Contract adapters and validators are finalized
3. Protocol surfaces in SPI are frozen

---

### Issue 0.1: Create core transport import check [MVP]

**Title**: Automated check for zero transport imports in omnibase_core
**Type**: Infrastructure
**Priority**: Critical
**Labels**: `architecture`, `ci`, `guardrails`
**Milestone**: v0.1.0 MVP

**Description**:
Create an automated CI check that verifies `omnibase_core` has NO imports of transport libraries.

**Blocked Libraries**:
- `kafka`, `aiokafka`
- `httpx`, `aiohttp`, `requests`
- `asyncpg`, `psycopg`
- `hvac` (Vault)
- `consul`
- `redis`, `valkey`

**Acceptance Criteria**:
- [ ] Shell script or pytest that greps for banned imports
- [ ] Runs in omnibase_core CI on every PR
- [ ] Clear error message listing violation locations
- [ ] Exit code 1 on any violation

---

### Issue 0.2: Enforce LocalHandler dev-only usage [MVP]

**Title**: Automated check that LocalHandler not used in infra
**Type**: Infrastructure
**Priority**: Critical
**Labels**: `architecture`, `ci`, `guardrails`
**Milestone**: v0.1.0 MVP

**Description**:
Ensure `LocalHandler` is ONLY used in `omnibase_core` tests, never imported or used in `omnibase_infra` production code.

**Production Fail-Fast**: BaseRuntimeHostProcess MUST fail fast if LocalHandler found in production startup. This is enforced at startup, not just CI.

**Acceptance Criteria**:
- [ ] Grep check in omnibase_infra CI
- [ ] Allowed: `tests/` directory imports for comparison tests
- [ ] Blocked: Any `src/` directory import of LocalHandler
- [ ] Clear error message on violation

---

## Phase 1: Core Types (omnibase_core) - MVP Issues

**Priority**: HIGH
**Dependencies**: Phase 0 CI in place
**Repository**: omnibase_core

---

### Issue 1.1: Add RUNTIME_HOST to EnumNodeKind [MVP]

**Title**: Add RUNTIME_HOST value to EnumNodeKind enum
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `enum`, `core`
**Milestone**: v0.1.0 MVP

**Description**:
Add `RUNTIME_HOST` value to `EnumNodeKind` enum to support runtime host contracts.

**File**: `src/omnibase_core/enums/enum_node_kind.py`

**Acceptance Criteria**:
- [ ] `EnumNodeKind.RUNTIME_HOST = "runtime_host"` added
- [ ] Serialization/deserialization works
- [ ] Exported from `omnibase_core.enums`
- [ ] mypy --strict passes

---

### Issue 1.2: Create EnumHandlerType enum [MVP]

**Title**: Create EnumHandlerType enum for protocol handlers
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `enum`, `core`
**Milestone**: v0.1.0 MVP

**Description**:
Create new `EnumHandlerType` enum for per-request handler types.

**File**: `src/omnibase_core/enums/enum_handler_type.py` (NEW)

**Values (MVP)**:
- `LOCAL = "local"` - Echo/test, dev only
- `HTTP = "http"` - HTTP REST calls
- `DB = "db"` - Database operations

**Values (Added in Beta)**:
- `VAULT = "vault"` - Vault secrets
- `CONSUL = "consul"` - Consul discovery
- `LLM = "llm"` - LLM API calls (future)

**Acceptance Criteria**:
- [ ] Enum created with MVP values
- [ ] Docstring explains Kafka is NOT a handler
- [ ] Exported from `omnibase_core.enums`
- [ ] mypy --strict passes

---

### Issue 1.3: Create ModelOnexEnvelope model (SIMPLIFIED) [MVP]

**Title**: Implement ModelOnexEnvelope unified message format
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `model`, `core`
**Milestone**: v0.1.0 MVP

**Description**:
Create the unified message envelope for all Runtime Host communication. **Simplified for MVP - no tenant_id yet.**

**File**: `src/omnibase_core/models/runtime/model_onex_envelope.py` (NEW)

**MVP Fields**:
- Identity: `envelope_id`, `envelope_version`
- Correlation: `correlation_id`, `causation_id`
- Routing: `source_node`, `target_node`, `handler_type`, `operation`
- Payload: `payload`, `metadata`
- Timing: `timestamp`
- Response: `is_response`, `success`, `error`

**Deferred to Beta**:
- `ttl_seconds`
- `metadata.tenant_id` reserved slot

**Acceptance Criteria**:
- [ ] Pydantic model with MVP fields defined
- [ ] JSON serialization with custom encoders (datetime, UUID)
- [ ] Unit tests for serialization/deserialization
- [ ] Exported from `omnibase_core.models`
- [ ] mypy --strict passes

---

### Issue 1.4: Create ModelRuntimeHostContract model (SIMPLIFIED) [MVP]

**Title**: Implement ModelRuntimeHostContract Pydantic model
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `model`, `core`
**Milestone**: v0.1.0 MVP

**Description**:
Create the runtime host contract model. **Simplified for MVP - no retry policies, no rate limits.**

**File**: `src/omnibase_core/models/contracts/model_runtime_host_contract.py` (NEW)

**MVP Components**:
- `ModelHandlerConfig` - Minimal: `type: EnumHandlerType`
- `ModelEventBusConfig` - Minimal: `kind: str`
- `ModelNodeRef` - Minimal: `slug: str`
- `ModelRuntimeHostContract` - Main contract

**Deferred to Beta**:
- Retry policies in handler config
- Rate limits in handler config
- Advanced event bus config

**Acceptance Criteria**:
- [ ] All sub-models created with minimal fields
- [ ] Clear separation between `handlers` and `event_bus`
- [ ] `from_yaml()` class method for loading
- [ ] Unit tests for YAML loading
- [ ] Exported from `omnibase_core.models`
- [ ] mypy --strict passes

---

### Issue 1.5: Verify ProtocolHandler export from SPI [MVP]

**Title**: Verify ProtocolHandler is available from omnibase_spi
**Type**: Task
**Priority**: High
**Labels**: `architecture`, `protocol`, `verification`
**Milestone**: v0.1.0 MVP

**Description**:
Verify that `ProtocolHandler` is properly exported from `omnibase_spi` and can be imported by `omnibase_infra`. The protocol is defined in SPI (not core) per architectural rules. Critical: `handler_type` must return `EnumHandlerType`, not `str`.

**Import**: `from omnibase_spi.protocols import ProtocolHandler`

**Methods**:
- `handler_type: EnumHandlerType` (property, abstract)
- `initialize(config: dict) -> None` (async)
- `shutdown() -> None` (async)
- `execute(envelope: ModelOnexEnvelope) -> ModelOnexEnvelope` (async)
- `health_check() -> dict` (async)
- `describe() -> dict` (NEW - MVP stub, full implementation in Beta)

**Handler Metadata via describe()**:
```python
def describe(self) -> dict:
    """Return handler metadata for debugging and observability."""
    return {
        "handler_type": self.handler_type.value,
        "supported_operations": ["get", "post"],  # handler-specific
        "required_payload_fields": ["url"],       # handler-specific
        "response_shape": "ModelHttpResponse",    # handler-specific
        "version": "1.0.0",
    }
```

**Handler Semantics**:
- **Async requirement**: Handlers MUST NOT perform blocking I/O; all operations must be async.
- **Lifecycle**: Handlers are singletons per BaseRuntimeHostProcess.
- **State management**: Handlers may not maintain global state except cached connections.

**Acceptance Criteria**:
- [ ] Abstract class with all methods defined
- [ ] `handler_type` returns `EnumHandlerType` (NOT `str`)
- [ ] Docstring explains handler vs event bus distinction
- [ ] Exported from `omnibase_core.protocols`
- [ ] mypy --strict passes
- [ ] `describe()` method returns handler metadata
- [ ] Supported operations are enumerable

---

### Issue 1.6: Implement NodeInstance class [MVP]

**Title**: Create NodeInstance execution wrapper
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `runtime`, `core`
**Milestone**: v0.1.0 MVP

**Description**:
Create the lightweight node instance wrapper that delegates to NodeRuntime for handler execution.

**File**: `src/omnibase_core/runtime/node_instance.py` (NEW)

**Properties**: `slug`, `node_type`, `contract`
**Methods**: `initialize()`, `shutdown()`, `handle(envelope)`

**Acceptance Criteria**:
- [ ] Class created with all properties/methods
- [ ] Delegates to `NodeRuntime.execute_with_handler()`
- [ ] NO I/O code in this class
- [ ] Docstring explains delegation pattern
- [ ] Unit tests with mock runtime
- [ ] mypy --strict passes

---

### Issue 1.7: Implement NodeRuntime class (MINIMAL) [MVP]

**Title**: Create NodeRuntime transport-agnostic orchestrator
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `runtime`, `core`
**Milestone**: v0.1.0 MVP

**Description**:
Create the core runtime that hosts multiple node instances. **MINIMAL for MVP - only essential methods.**

**File**: `src/omnibase_core/runtime/node_runtime.py` (NEW)

**MVP Methods**:
- `register_handler(handler)` - Takes `ProtocolHandler`
- `register_node(node)` - Takes `NodeInstance`
- `route_envelope(envelope)` - Routes to node
- `execute_with_handler(envelope)` - Routes to handler

**Deferred to Beta**:
- `initialize()`, `shutdown()` lifecycle methods
- `health_check()` aggregation
- `is_running` state management

**Acceptance Criteria**:
- [ ] NO `_event_bus_loop` method (transport-agnostic)
- [ ] NO Kafka/HTTP/DB imports
- [ ] Handlers stored by `EnumHandlerType` key
- [ ] Pure in-memory orchestration
- [ ] Unit tests with LocalHandler only
- [ ] mypy --strict passes

---

### Issue 1.8: Implement FileContractRegistry class (SIMPLE) [MVP]

**Title**: Create FileContractRegistry for contract loading
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `runtime`, `core`
**Milestone**: v0.1.0 MVP

> **Implementation Status**: This component is PLANNED for `omnibase_core` and does NOT yet exist.
> This issue defines the specification for future implementation.

> **Naming Convention**: Following ONEX naming standards (`Registry<Purpose>`), this class should be named
> `FileContractRegistry` (file: `registry_file_contract.py`). The name describes its purpose: a registry
> that loads contracts from files. Alternative names considered: `RegistryFileBased`, `FileRegistry`.

> **Relationship with HandlerContractSource**: `omnibase_infra` currently provides `HandlerContractSource`
> for handler-specific contract discovery from `handler_contract.yaml` files. Once `FileContractRegistry`
> is implemented in `omnibase_core`, `HandlerContractSource` should be evaluated for potential migration
> to use `FileContractRegistry` as its underlying file loading mechanism, while retaining handler-specific
> discovery logic. These remain complementary components with different scopes.

**Description**:
Create the file-based contract registry for loading node contracts from filesystem. **Simple for MVP - just load YAML to Pydantic, no skip rules.**

**File**: `src/omnibase_core/runtime/registry_file_contract.py` (NEW)

**MVP Methods**: `load(path) -> Model`, `load_all(directory) -> list[Model]`

**Path Resolution Rules**:
- Absolute paths: Used as-is
- Relative paths: Resolved from the contract file's parent directory
- Symbolic links: Followed (not resolved to canonical path)
- `~` expansion: NOT supported (must use absolute paths)

**Contract Loader Fail-Fast Rules**:

| Condition | Behavior | Exit Code |
|-----------|----------|-----------|
| Contract file not found | CRASH with clear error message | 1 |
| Invalid YAML syntax | CRASH with line number and error | 1 |
| Unknown fields in contract | CRASH (strict mode) or WARN (lenient mode) | 1 or 0 |
| Unknown handler type | CRASH - handler not in registry | 1 |
| Missing required fields | CRASH with list of missing fields | 1 |
| Schema version mismatch | CRASH with version comparison | 1 |
| YAML anchors/aliases | ALLOWED - standard YAML feature | N/A |
| Empty handlers list | WARN but continue (valid edge case) | 0 |
| Duplicate handler types | CRASH - one handler per type in MVP | 1 |

**MVP Default Mode**: Strict (crash on unknown fields)
**Beta Addition**: `--lenient` flag to warn instead of crash on unknown fields

**Error Message Format**:
```
ContractValidationError: Failed to load contract
  File: /path/to/contract.yaml
  Error: Unknown handler type 'kafka' at line 15
  Hint: Valid handler types are: http, db
  Action: Remove or replace the invalid handler type
```

**Deferred to Beta**:
- Skip files starting with `_`
- Pattern-based filtering
- Caching

**Acceptance Criteria**:
- [ ] Loads YAML contracts from directory
- [ ] Returns Pydantic models
- [ ] Error handling for invalid contracts
- [ ] Fail-fast on all conditions in the table above
- [ ] Clear, actionable error messages with file path and line number
- [ ] Unit tests with temp directories
- [ ] mypy --strict passes

---

### Issue 1.9: Implement LocalHandler (dev/test only) [MVP]

**Title**: Create LocalHandler echo handler for testing
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `handler`, `core`, `testing`
**Milestone**: v0.1.0 MVP

**Description**:
Create the local echo handler for testing. Must have clear warnings that it's NOT for production.

**File**: `src/omnibase_core/runtime/handlers/local_handler.py` (NEW)

**Operations**: `echo`, `transform`, `error`

**Acceptance Criteria**:
- [ ] Returns `EnumHandlerType.LOCAL` (NOT str)
- [ ] WARNING in docstring about dev/test only
- [ ] Logs warning on initialization
- [ ] `health_check` returns `dev_test_only: True`
- [ ] Unit tests for all operations
- [ ] mypy --strict passes

---

### Issue 1.10: Create dev/test CLI entry point (SIMPLE) [MVP]

**Title**: Add runtime-host CLI command for dev/test
**Type**: Feature
**Priority**: Medium
**Labels**: `cli`, `core`, `testing`
**Milestone**: v0.1.0 MVP

**Description**:
Add CLI entry point for testing Runtime Host with LocalHandler only. **Simple for MVP - just `omninode-runtime-host-dev CONTRACT.yaml`**

**File**: `src/omnibase_core/cli/runtime_host_cli.py` (NEW)

**Usage**: `omninode-runtime-host-dev CONTRACT_PATH`

**Acceptance Criteria**:
- [ ] Clearly named as dev/test only (`-dev` suffix)
- [ ] Only registers LocalHandler
- [ ] **HARD CHECK**: Refuses to start if `ENVIRONMENT=prod`
- [ ] Prints warning about production usage
- [ ] Entry point in pyproject.toml
- [ ] mypy --strict passes

---

### Issue 1.11: Create minimal error taxonomy (core) [MVP]

**Title**: Define minimal core error hierarchy for runtime
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `errors`, `core`
**Milestone**: v0.1.0 MVP

**Description**:
Create minimal base error classes. **MVP only includes essential errors.**

**File**: `src/omnibase_core/errors/runtime_errors.py` (NEW)

**MVP Classes**:
- `RuntimeHostError(OnexError)` - Base for all runtime errors
- `HandlerExecutionError(RuntimeHostError)` - Handler execution failed
- `EventBusError(RuntimeHostError)` - Event bus operation failed
- `InvalidOperationError(RuntimeHostError)` - Handler received unsupported operation (NEW)
- `ContractValidationError(RuntimeHostError)` - Contract failed validation (NEW)

**Error Invariants**:
- All errors MUST include `correlation_id` (auto-assigned if not present in source envelope)
- All errors MUST include `handler_type` when applicable
- All errors MUST include `operation` when applicable
- Raw exception stack traces MUST NOT appear in error envelopes (logged internally only)

**Error Envelope Requirements**:

When a handler or runtime produces an error, the response envelope MUST:

| Field | Requirement | Source |
|-------|-------------|--------|
| `envelope_id` | New UUID for the response | BaseRuntimeHostProcess generates |
| `correlation_id` | MUST echo from request envelope | Preserved from input |
| `causation_id` | MUST be set to request's `envelope_id` | Links response to request |
| `is_response` | MUST be `True` | Indicates this is a response |
| `success` | MUST be `False` | Indicates failure |
| `error.code` | Error type enum value | From `RuntimeHostError` subclass |
| `error.message` | Human-readable description | From exception message |
| `error.handler_type` | Handler that failed (if applicable) | From handler context |
| `error.operation` | Operation that failed (if applicable) | From envelope payload |

**Error Envelope Example**:
```json
{
  "envelope_id": "resp-uuid-5678",
  "correlation_id": "corr-uuid-1234",
  "causation_id": "req-uuid-1234",
  "is_response": true,
  "success": false,
  "error": {
    "code": "HANDLER_EXECUTION_ERROR",
    "message": "Database connection failed",
    "handler_type": "db",
    "operation": "query"
  },
  "payload": null,
  "metadata": {
    "original_target_node": "user_service"
  }
}
```

**Invariants**:
- Error envelopes MUST preserve `correlation_id` from the request (enables tracing)
- Error envelopes MUST NOT include raw stack traces (security risk)
- Error envelopes MUST include enough context for debugging without exposing internals
- `metadata.original_target_node` SHOULD be preserved to identify which node failed

**Deferred to Beta**:
- `HandlerNotFoundError`
- `NodeNotFoundError`
- `EnvelopeValidationError`
- `ProtocolConfigurationError`
- `SecretResolutionError`

**Acceptance Criteria**:
- [ ] All inherit from `OnexError`
- [ ] Structured fields for logging
- [ ] `correlation_id` optional field
- [ ] Unit tests for error construction
- [ ] mypy --strict passes

---

## Phase 2: SPI Protocol Updates (omnibase_spi) - MVP Issues

**Priority**: HIGH
**Dependencies**: Phase 1 complete
**Repository**: omnibase_spi

---

### Issue 2.1: Export ProtocolHandler from SPI [MVP]

**Title**: Export ProtocolHandler from omnibase_spi.protocols
**Type**: Task
**Priority**: High
**Labels**: `architecture`, `protocol`, `spi`
**Milestone**: v0.1.0 MVP

**Description**:
Export `ProtocolHandler` from SPI protocols module for infrastructure implementations.

**File**: `src/omnibase_spi/protocols/__init__.py` (UPDATE)

**Acceptance Criteria**:
- [ ] `ProtocolHandler` added to exports
- [ ] Import works: `from omnibase_spi.protocols import ProtocolHandler`
- [ ] No breaking changes to existing exports
- [ ] mypy passes

---

### Issue 2.2: Update ProtocolEventBus with envelope methods [MVP]

**Title**: Add envelope methods to ProtocolEventBus
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `protocol`, `spi`
**Milestone**: v0.1.0 MVP

**Description**:
Update `ProtocolEventBus` abstract class with `ModelOnexEnvelope` support.

**File**: `src/omnibase_spi/protocols/protocol_event_bus.py` (UPDATE)

**Methods**:
- `publish_envelope(envelope, topic)`
- `subscribe(topic, handler)` - handler takes/returns envelope
- `start_consuming()`
- `health_check()`

**Acceptance Criteria**:
- [ ] All methods added with proper typing
- [ ] Docstring explains handler vs event bus distinction
- [ ] Backward compatibility maintained
- [ ] mypy passes

---

### Issue 2.3: Document handler vs event bus distinction [MVP]

**Title**: Add documentation for handler vs event bus separation
**Type**: Documentation
**Priority**: Medium
**Labels**: `documentation`, `architecture`, `spi`
**Milestone**: v0.1.0 MVP

**Description**:
Add clear documentation explaining when to use ProtocolHandler vs ProtocolEventBus.

**File**: `src/omnibase_spi/protocols/README.md` (NEW or UPDATE)

**Acceptance Criteria**:
- [ ] Table comparing Handler vs EventBus
- [ ] Use cases documented
- [ ] Examples provided
- [ ] Cross-referenced in docstrings

---

## Phase 3: Infrastructure Handlers (omnibase_infra) - MVP Issues

**Priority**: HIGH
**Dependencies**: Phase 1 and Phase 2 complete
**Repository**: omnibase_infra

**Handler Implementation Guidelines**:

Every handler MUST define:
1. **Supported operations** - Enumerable list of valid `operation` values
2. **Required payload fields** - Fields that MUST be present in envelope payload
3. **Response shape** - Structure of successful response payload
4. **Error mapping** - How library exceptions map to `HandlerExecutionError`

**Code Quality Requirements**:
- Each handler file MUST be <300 lines of code (LOC)
- Docstrings MUST include: summary, parameters, returns, error modes
- All handlers MUST implement `health_check()` returning structured JSON

**Handler Operation Naming Convention**:

All handler operations MUST use lowercase with dots as separators:

| Pattern | Valid Examples | Invalid Examples |
|---------|----------------|------------------|
| `{domain}.{action}` | `http.get`, `db.query` | `HTTP_GET`, `dbQuery` |
| `{domain}.{action}` | `http.post`, `db.execute` | `http-post`, `DB.EXECUTE` |
| `{domain}.{action}_{modifier}` | `graph.get_neighbors` | `graph.getNeighbors` |

**Recommended: Define as Literal types**:
```python
# In handler implementation
HttpOperation = Literal["http.get", "http.post", "http.put", "http.delete"]
DbOperation = Literal["db.query", "db.execute", "db.transaction"]
```

**Contract Validation**:
Operations in contracts are validated against handler's supported operations.
Unknown operations -> `InvalidOperationError` at runtime.

---

### Handler Lifecycle State Machine

Every handler transitions through these states:

```
                    +---------------------------------------------+
                    |         HANDLER LIFECYCLE                   |
                    +---------------------------------------------+

    +----------+    initialize()    +-------------+
    | CREATED  | -----------------> | INITIALIZED |
    +----------+                    +-------------+
         |                                |
         | (constructor called)           | health_check() returns healthy
         |                                v
         |                          +-------------+
         |                          |   HEALTHY   | <----+
         |                          +-------------+      |
         |                                |              |
         |                                | execute()    |
         |                                v              |
         |                          +-------------+      |
         |                          |  EXECUTING  | -----+
         |                          +-------------+  (success)
         |                                |
         |                                | (error or shutdown signal)
         |                                v
         |                          +-------------+
         +------------------------->|  SHUTDOWN   |
              (any error during     +-------------+
               init or fatal)
```

**State Definitions**:

| State | Description | Valid Transitions |
|-------|-------------|-------------------|
| `CREATED` | Constructor called, no resources allocated | -> INITIALIZED, -> SHUTDOWN |
| `INITIALIZED` | `initialize()` completed, resources ready | -> HEALTHY, -> SHUTDOWN |
| `HEALTHY` | Ready to accept `execute()` calls | -> EXECUTING, -> SHUTDOWN |
| `EXECUTING` | Currently processing an envelope | -> HEALTHY, -> SHUTDOWN |
| `SHUTDOWN` | `shutdown()` called, resources released | (terminal) |

**Lifecycle Invariants**:
- `execute()` MUST only be called when handler is HEALTHY
- `shutdown()` MUST be called exactly once, regardless of current state
- `initialize()` MUST be called before any `execute()` calls
- Handler MUST NOT accept new `execute()` calls after `shutdown()` begins
- `health_check()` may be called in any state except SHUTDOWN

**MVP Simplification**:
MVP does not track state formally - handlers are assumed healthy after init.
Beta will add explicit state tracking for graceful shutdown and health degradation.

---

### Issue 3.1: Create handlers directory structure (SIMPLIFIED) [MVP]

**Title**: Create handlers/ directory with __init__.py
**Type**: Task
**Priority**: High
**Labels**: `architecture`, `infrastructure`
**Milestone**: v0.1.0 MVP

**Description**:
Create the handlers directory structure for protocol handler implementations. **Simplified for MVP.**

**MVP Structure**:
```
src/omnibase_infra/
|-- handlers/
|   |-- __init__.py
|   |-- http_handler.py
|   |-- db_handler.py
|-- event_bus/
|   |-- __init__.py
|   |-- inmemory_event_bus.py
|-- runtime/
|   |-- __init__.py
|   |-- runtime_host_process.py
|   |-- wiring.py
|-- errors/
|   |-- __init__.py
|   |-- infra_errors.py
```

**Deferred to Beta**:
- `handlers/handler_vault.py`
- `handlers/consul_handler.py`
- `event_bus/kafka_event_bus.py`
- `runtime/secret_resolver.py`
- `runtime/handler_config_resolver.py`
- `observability/` directory

**Acceptance Criteria**:
- [ ] MVP directories created
- [ ] `__init__.py` files with exports
- [ ] Clear separation between handlers/, event_bus/, runtime/

---

### Issue 3.2: Implement HttpHandler (MINIMAL) [MVP]

**Title**: Create HTTP REST protocol handler
**Type**: Feature
**Priority**: High
**Labels**: `handler`, `infrastructure`
**Milestone**: v0.1.0 MVP

**Description**:
Implement `HttpHandler` for HTTP REST operations using httpx. **Minimal for MVP - GET + POST only, no retries.**

**File**: `src/omnibase_infra/handlers/http_handler.py`

**MVP Operations**: GET, POST only

**Deferred to Beta**:
- PUT, DELETE, PATCH
- Retry logic
- Rate limiting
- Configurable timeout from `ModelHandlerBindingConfig`

**Handler Contract**:
- Supported operations: `get`, `post` (MVP); `put`, `delete`, `patch` (Beta)
- Required payload fields: `url` (required), `headers` (optional), `body` (optional for POST)
- Response shape: `{"status_code": int, "headers": dict, "body": str | dict}`

**Unsupported Operation Handling**:
- Attempting `put`, `delete`, `patch` in MVP -> raises `InvalidOperationError`
- Error message: "Operation '{op}' not supported in MVP. Available: get, post"

**Acceptance Criteria**:
- [ ] Returns `EnumHandlerType.HTTP` (not str)
- [ ] Uses httpx async client
- [ ] Fixed 30s timeout
- [ ] Proper error handling -> `HandlerExecutionError`
- [ ] Unit tests with mock responses
- [ ] mypy --strict passes

---

### Issue 3.3: Implement HandlerDb (MINIMAL) [MVP]

**Title**: Create PostgreSQL database protocol handler
**Type**: Feature
**Priority**: High
**Labels**: `handler`, `infrastructure`, `database`
**Milestone**: v0.1.0 MVP

**Description**:
Implement `HandlerDb` for PostgreSQL operations using asyncpg. **Minimal for MVP - query + execute only, no transactions.**

**File**: `src/omnibase_infra/handlers/db_handler.py`

**MVP Operations**: `query`, `execute`

**Deferred to Beta**:
- `transaction` operation
- Configurable pool size
- Connection pooling tuning

**Handler Contract**:
- Supported operations: `query`, `execute` (MVP); `transaction` (Beta)
- Required payload fields: `sql` (required), `parameters` (optional list)
- Response shape: `{"rows": list[dict], "row_count": int}`

**SQL Injection Protection**:
- All queries MUST use parameterized statements (no string interpolation)
- Parameters passed via `parameters` list, never embedded in SQL string
- HandlerDb MUST validate parameter count matches placeholder count
- Suspicious patterns (`;--`, `DROP`, `UNION` outside subqueries) trigger warning log

**Unsupported Operation Handling**:
- Attempting `transaction` in MVP -> raises `InvalidOperationError`
- Error message: "Operation 'transaction' not supported in MVP. Available: query, execute"

**Acceptance Criteria**:
- [ ] Returns `EnumHandlerType.DB` (not str)
- [ ] Uses asyncpg connection pool
- [ ] Fixed pool size (5)
- [ ] Maps `asyncpg.PostgresError` -> `HandlerExecutionError`
- [ ] Unit tests with mock pool
- [ ] mypy --strict passes

---

### Issue 3.4: Implement InMemoryEventBus [MVP]

**Title**: Create in-memory event bus for local development
**Type**: Feature
**Priority**: High
**Labels**: `event-bus`, `infrastructure`, `testing`
**Milestone**: v0.1.0 MVP

**Description**:
Implement `InMemoryEventBus` for local testing and CI without Kafka. **This is the PRIMARY event bus for MVP.**

**Ordering Guarantees**: Runtime MUST NOT assume global ordering. InMemoryEventBus provides FIFO per topic only.

**Response Pattern**: Event bus handlers must return envelopes; BaseRuntimeHostProcess is responsible for publishing responses.

**File**: `src/omnibase_infra/event_bus/inmemory_event_bus.py`

**Acceptance Criteria**:
- [ ] Implements `ProtocolEventBus`
- [ ] Uses `asyncio.Queue` per topic
- [ ] No networking required
- [ ] Unit tests
- [ ] BaseRuntimeHostProcess works with this bus
- [ ] mypy --strict passes

---

### Issue 3.5: Create wiring.py (SIMPLE) [MVP]

**Title**: Create wiring.py handler registration module
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `infrastructure`, `wiring`
**Milestone**: v0.1.0 MVP

**Description**:
Create the single source of truth for handler registration. **Simple for MVP - no describe_handlers, no validate_handler_config.**

**File**: `src/omnibase_infra/runtime/wiring.py`

**MVP Contents**:
- `HANDLER_REGISTRY: dict[EnumHandlerType, type[ProtocolHandler]]`
- `EVENT_BUS_REGISTRY: dict[str, type[ProtocolEventBus]]`
- `register_handlers_from_config(runtime, handler_configs)`
- `get_handler_class(handler_type)`
- `get_event_bus_class(kind: str)`

**Deferred to Beta**:
- `list_available_handlers()`
- `describe_handlers()` - Returns diagnostic info
- `validate_handler_config(handler_configs)` - Validates before registration

**Acceptance Criteria**:
- [ ] Single handler registry dictionary
- [ ] Single event bus registry dictionary
- [ ] Function to register from config
- [ ] NO duplicate logic elsewhere
- [ ] Docstring marks as single source of truth
- [ ] Unit tests for registration
- [ ] mypy --strict passes

---

### Issue 3.6: Implement BaseRuntimeHostProcess (MINIMAL) [MVP]

**Title**: Create BaseRuntimeHostProcess infrastructure wrapper
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `infrastructure`, `runtime`
**Milestone**: v0.1.0 MVP

**Description**:
Create the infrastructure-level process wrapper that owns the event bus and drives NodeRuntime. **Minimal for MVP - load contract, init handlers, consume/route/publish, shutdown without drain.**

**File**: `src/omnibase_infra/runtime/runtime_host_process.py`

**MVP Responsibilities**:
- Load contract
- Register handlers via `wiring.py`
- Own and manage event bus
- Call `runtime.route_envelope()` for each message
- Basic shutdown (no graceful drain)

**Critical Invariant**: NodeRuntime MUST NOT start background tasks. BaseRuntimeHostProcess exclusively drives consumption and dispatch.

**Error Handling**: MVP behavior: errors are returned as `success=False` envelopes published to the output topic; no retries.

**Concurrency Model**: MVP: BaseRuntimeHostProcess processes envelopes sequentially per subscription. Beta introduces parallel workers.

**Deferred to Beta**:
- Resolve secrets via `SecretResolver`
- Signal handling for SIGTERM/SIGINT
- Graceful shutdown with drain phase
- `SHUTDOWN_GRACE_SECONDS` configurable
- Single instance enforcement per process

**Missing Handler Behavior**:

When an envelope requests a handler type that is not registered:

| Scenario | MVP Behavior | Beta Behavior |
|----------|--------------|---------------|
| Handler type not in registry | Return error envelope | Same |
| Handler initialized but unhealthy | Return error envelope | Retry or circuit break |
| Handler type is "local" in infra | CRASH at startup | Same |

**Error Response for Missing Handler**:
```json
{
  "success": false,
  "error": {
    "code": "HANDLER_NOT_FOUND",
    "message": "No handler registered for type 'vault'",
    "handler_type": "vault",
    "available_handlers": ["http", "db"]
  }
}
```

**Fallback Rules**:
- MVP: No fallback. Missing handler = error envelope.
- LocalHandler fallback is FORBIDDEN in infra (dev-only in core tests).
- Future: Beta may add circuit breaker for degraded handlers.

**Acceptance Criteria**:
- [ ] Uses `wiring.py` for handler registration
- [ ] Owns event bus instance (InMemory for MVP)
- [ ] Subscribes event bus to call `runtime.route_envelope()`
- [ ] Basic shutdown
- [ ] Unit tests with mock event bus
- [ ] mypy --strict passes
- [ ] NodeRuntime does not start any background tasks or event loops
- [ ] Errors produce `success=False` response envelopes
- [ ] Sequential envelope processing (no parallelism in MVP)

---

### Issue 3.7: Create example runtime host contract (SIMPLE) [MVP]

**Title**: Create infra_runtime_host.yaml example contract
**Type**: Documentation
**Priority**: Medium
**Labels**: `documentation`, `infrastructure`, `contract`
**Milestone**: v0.1.0 MVP

**Description**:
Create an example runtime host contract showing minimal MVP configuration.

**File**: `src/omnibase_infra/contracts/runtime/infra_runtime_host.yaml`

**Acceptance Criteria**:
- [ ] Shows `handlers` section (http, db only)
- [ ] Shows `event_bus` section with `kind: inmemory`
- [ ] LocalHandler NOT enabled (commented with warning)
- [ ] Comments explaining structure
- [ ] Validates against simplified contract model

---

### Issue 3.8: Add production CLI entry point (SIMPLE) [MVP]

**Title**: Add omnibase-runtime-host CLI command
**Type**: Feature
**Priority**: Medium
**Labels**: `cli`, `infrastructure`
**Milestone**: v0.1.0 MVP

**Description**:
Add production CLI entry point for BaseRuntimeHostProcess. **Simple for MVP - just `omnibase-runtime-host CONTRACT.yaml`**

**Usage**: `omnibase-runtime-host CONTRACT_PATH`

**Deferred to Beta**:
- `--dry-run` mode
- `--describe-wiring` option
- Graceful shutdown on signals

**Acceptance Criteria**:
- [ ] Entry point in pyproject.toml
- [ ] Loads contract and starts BaseRuntimeHostProcess
- [ ] Proper logging configuration
- [ ] **Blocks LocalHandler** - fails if contract enables it
- [ ] Manual test instructions

---

## Phase 4: Integration & Testing - MVP Issues

**Priority**: HIGH/MEDIUM
**Dependencies**: Phase 3 complete
**Repository**: omnibase_infra

**Testing Infrastructure Requirements**:

**Deterministic Test Helpers** (MVP):
```python
# tests/helpers/deterministic.py
class DeterministicIdGenerator:
    """Generates predictable UUIDs for testing."""
    def __init__(self, seed: int = 42):
        self._counter = seed

    def next_uuid(self) -> UUID:
        self._counter += 1
        return UUID(int=self._counter)

class DeterministicClock:
    """Provides controllable timestamps for testing."""
    def __init__(self, start: datetime = datetime(2024, 1, 1)):
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: int) -> None:
        self._now += timedelta(seconds=seconds)
```

**Test Assertions Required**:
All test suites MUST assert:
- No unexpected log warnings (capture and check logs)
- No duplicate handler initialization (track init calls)
- No resource leaks (connections, file handles)
- Deterministic ordering of results

---

### Testing Without Docker

Many contributors prefer local testing without Docker. Here's the pure-Python strategy:

**Local Testing Stack**:

| Component | Docker Alternative | Setup |
|-----------|-------------------|-------|
| PostgreSQL | `testcontainers-python` | Auto-spawns container per test |
| PostgreSQL | SQLite (limited) | In-memory, schema differences |
| Kafka | `InMemoryEventBus` | No Kafka needed for MVP |
| Vault | Mock client | `hvac` mock in tests |
| Consul | Mock client | Dict-based mock |

**Recommended: testcontainers Approach**:
```python
# tests/conftest.py
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def postgres_container():
    """Spawn real PostgreSQL for integration tests."""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres.get_connection_url()

@pytest.fixture
def db_handler(postgres_container, mock_container):
    """HandlerDb with real PostgreSQL."""
    handler = HandlerDb(mock_container)
    await handler.initialize({"connection_string": postgres_container})
    yield handler
    await handler.shutdown()
```

**Pure In-Memory Testing** (no containers at all):
```python
# tests/conftest.py
@pytest.fixture
def inmemory_event_bus():
    """Event bus that needs no infrastructure."""
    bus = InMemoryEventBus()
    await bus.initialize({})
    yield bus
    await bus.shutdown()

@pytest.fixture
def mock_db_handler(mock_container):
    """HandlerDb with mocked pool."""
    handler = HandlerDb(mock_container)
    handler._pool = AsyncMock()
    handler._pool.execute.return_value = [{"id": 1}]
    return handler
```

**Test Categories by Infrastructure Need**:

| Test Type | Infrastructure | Run Command |
|-----------|----------------|-------------|
| Unit tests | None | `pytest tests/unit/` |
| Handler tests (mocked) | None | `pytest tests/unit/handlers/` |
| Integration (containers) | testcontainers | `pytest tests/integration/ -m "not docker"` |
| Full E2E | Docker Compose | `docker-compose up -d && pytest tests/e2e/` |

**CI Strategy**:
- PR checks: Unit tests only (fast, no containers)
- Merge to main: Full integration with testcontainers
- Nightly: E2E with Docker Compose

---

### Issue 4.1: Unit tests for handlers (mocked) [MVP]

**Title**: Unit test coverage for MVP handlers
**Type**: Testing
**Priority**: High
**Labels**: `testing`, `handlers`
**Milestone**: v0.1.0 MVP

**Description**:
Unit tests for HttpHandler and HandlerDb with mocked dependencies.

**Files**:
- `tests/unit/handlers/test_http_handler.py`
- `tests/unit/handlers/test_db_handler.py`

**Acceptance Criteria**:
- [ ] Each handler has dedicated test file
- [ ] Mock external dependencies
- [ ] Test all MVP operations
- [ ] Test error handling -> mapped errors
- [ ] >80% coverage per handler
- [ ] Uses DeterministicIdGenerator for all UUIDs
- [ ] Uses DeterministicClock for all timestamps
- [ ] Asserts no unexpected warnings in logs
- [ ] Asserts handler initialized exactly once

---

### Issue 4.2: Unit tests for InMemoryEventBus [MVP]

**Title**: Unit tests for InMemoryEventBus
**Type**: Testing
**Priority**: High
**Labels**: `testing`, `event-bus`
**Milestone**: v0.1.0 MVP

**Description**:
Unit tests for InMemoryEventBus implementation.

**File**: `tests/unit/event_bus/test_inmemory_event_bus.py`

**Acceptance Criteria**:
- [ ] Test envelope publishing
- [ ] Test subscription and consumption
- [ ] Test error handling
- [ ] Test health check
- [ ] >80% coverage

---

### Issue 4.3: Single E2E flow test with InMemoryEventBus [MVP]

**Title**: E2E test: InMemoryEventBus -> Runtime -> Handler
**Type**: Testing
**Priority**: High
**Labels**: `testing`, `e2e`
**Milestone**: v0.1.0 MVP

**Description**:
Single end-to-end test verifying the complete flow using InMemoryEventBus.

**File**: `tests/integration/test_e2e_envelope_flow.py`

**Determinism Requirements**:
- E2E tests must override envelope_id for deterministic assertions.
- All envelope fields that contain UUIDs or timestamps must be mockable/injectable.

**Clock/Time Mocking**:
- All E2E tests MUST use `DeterministicClock`
- `envelope.timestamp` MUST be injectable
- Time-based assertions MUST NOT use `time.sleep()`
- Use `asyncio.wait_for()` with timeout instead of sleeps

**Acceptance Criteria**:
- [ ] Publish envelope to InMemoryEventBus
- [ ] BaseRuntimeHostProcess receives it
- [ ] Routes to correct node
- [ ] Handler executes operation
- [ ] Response published to response topic
- [ ] Verify response content
- [ ] Envelope IDs are deterministic (injected, not generated)
- [ ] Test assertions do not depend on random values

---

### Issue 4.4: Architecture compliance verification (SIMPLIFIED) [MVP]

**Title**: Verify architectural invariants
**Type**: Task
**Priority**: High
**Labels**: `architecture`, `validation`
**Milestone**: v0.1.0 MVP

**Description**:
Create simplified verification that architectural invariants are maintained. **Just grep checks for MVP.**

**Checks**:
- [ ] `grep -r "kafka" src/omnibase_core/` returns empty
- [ ] `grep -r "httpx" src/omnibase_core/` returns empty (except requirements)
- [ ] `grep -r "asyncpg" src/omnibase_core/` returns empty

**Deferred to Beta**:
- All handlers return `EnumHandlerType` check
- `wiring.py` is only handler registration location check
- No `os.getenv` in handlers check

**Acceptance Criteria**:
- [ ] Shell script or pytest for verification
- [ ] Runs in CI
- [ ] Clear error messages on failure

---

## Phase 5: Deployment & Migration - MVP Issues

**Priority**: MEDIUM
**Dependencies**: Phase 4 complete
**Repository**: omnibase_infra

---

### Issue 5.1: Create Dockerfile for runtime host (BASIC) [MVP]

**Title**: Create basic Dockerfile
**Type**: DevOps
**Priority**: Medium
**Labels**: `deployment`, `docker`
**Milestone**: v0.1.0 MVP

**Description**:
Create basic Dockerfile for BaseRuntimeHostProcess.

**File**: `docker/Dockerfile.runtime-host`

**Version Pinning Requirements**:
- Base image MUST use specific tag (e.g., `python:3.12.1-slim`), never `latest`
- All apt/apk packages MUST pin versions where possible
- All pip dependencies MUST be locked via requirements.txt or uv.lock

**Resource Targets**:
- Minimum container memory: 128MB RAM
- Minimum container CPU: 0.25 vCPU
- Recommended for production: 256MB RAM, 0.5 vCPU

**Entrypoint Configuration**:
```dockerfile
# Use ENTRYPOINT, not CMD
ENTRYPOINT ["omnibase-runtime-host"]
# CMD is overridable; ENTRYPOINT ensures correct binary always runs
# Users can still pass args: docker run image contract.yaml
```

**Why ENTRYPOINT over CMD**:
- ENTRYPOINT cannot be accidentally overridden
- Ensures the runtime-host binary always executes
- Arguments passed at runtime append to ENTRYPOINT
- Prevents misuse like `docker run image /bin/bash`

**Acceptance Criteria**:
- [ ] Single-stage build (multi-stage in Beta)
- [ ] All dependencies included
- [ ] Runs BaseRuntimeHostProcess
- [ ] Base image uses specific version tag
- [ ] No `latest` tags in Dockerfile
- [ ] Dependencies version-locked
- [ ] Uses ENTRYPOINT, not CMD
- [ ] Healthcheck configured with appropriate intervals

---

### Issue 5.2: Create docker-compose for local development [MVP]

**Title**: docker-compose.yaml for local development
**Type**: DevOps
**Priority**: Medium
**Labels**: `deployment`, `docker`, `development`
**Milestone**: v0.1.0 MVP

**Description**:
Create docker-compose.yaml with minimal services for local development.

> **Production Warning**: docker-compose.yaml MUST NOT be used for production; K8s manifests are the only supported deployment path.

**MVP Services**:
- runtime-host
- postgresql

**Deferred to Beta**:
- kafka (redpanda)
- vault (dev mode)
- consul (dev mode)

**Acceptance Criteria**:
- [ ] MVP services defined
- [ ] Health checks configured
- [ ] Environment variables documented
- [ ] README with usage instructions

---

### Issue 5.3: Update CLAUDE.md with new architecture (MINIMAL) [MVP]

**Title**: Update CLAUDE.md for Runtime Host
**Type**: Documentation
**Priority**: Medium
**Labels**: `documentation`
**Milestone**: v0.1.0 MVP

**Description**:
Update CLAUDE.md with basic Runtime Host architecture patterns.

**MVP Sections**:
- Runtime Host Overview
- Handler vs Event Bus distinction
- Simplified contract structure
- Basic CLI usage

**Deferred to Beta**:
- Development workflow
- CLIs and Environments (dev vs prod)
- Agent delegation patterns

**Acceptance Criteria**:
- [ ] Runtime Host section added
- [ ] Basic patterns documented
- [ ] Example contracts linked

---

## MVP Execution Order

```
Phase 0 (CI Guardrails - MVP)
    |
    +-- 0.1 Transport import check [MVP]
    +-- 0.2 LocalHandler usage check [MVP]
    |
    v
Phase 1 (Core Types - omnibase_core)
    |
    +-- 1.1 EnumNodeKind [MVP]
    +-- 1.2 EnumHandlerType [MVP]
    +-- 1.3 ModelOnexEnvelope (simplified) [MVP]
    +-- 1.4 ModelRuntimeHostContract (simplified) [MVP]
    +-- 1.5 ProtocolHandler [MVP]
    +-- 1.6 NodeInstance [MVP]
    +-- 1.7 NodeRuntime (minimal) [MVP]
    +-- 1.8 FileContractRegistry (simple) [MVP]
    +-- 1.9 LocalHandler [MVP]
    +-- 1.10 Dev CLI (simple) [MVP]
    +-- 1.11 Minimal error taxonomy [MVP]
    |
    v
Phase 2 (SPI Updates - omnibase_spi)
    |
    +-- 2.1 Export ProtocolHandler [MVP]
    +-- 2.2 Update ProtocolEventBus [MVP]
    +-- 2.3 Documentation [MVP]
    |
    v
Phase 3 (Infra - MVP)
    |
    +-- 3.1 Directory structure (simplified) [MVP]
    +-- 3.2 HttpHandler (minimal) [MVP]
    +-- 3.3 HandlerDb (minimal) [MVP]
    +-- 3.4 InMemoryEventBus [MVP]
    +-- 3.5 wiring.py (simple) [MVP]
    +-- 3.6 BaseRuntimeHostProcess (minimal) [MVP]
    +-- 3.7 Example contract (simple) [MVP]
    +-- 3.8 Production CLI (simple) [MVP]
    |
    v
Phase 4 (Testing - MVP)
    |
    +-- 4.1 Handler unit tests (mocked) [MVP]
    +-- 4.2 InMemoryEventBus unit tests [MVP]
    +-- 4.3 Single E2E flow test [MVP]
    +-- 4.4 Architecture compliance (grep checks) [MVP]
    |
    v
Phase 5 (Deployment - MVP)
    |
    +-- 5.1 Dockerfile (basic) [MVP]
    +-- 5.2 docker-compose (minimal) [MVP]
    +-- 5.3 Update CLAUDE.md (minimal) [MVP]
    |
    v
v0.1.0 Release
```

---

## MVP Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| E2E flow works | Yes | Single test passes |
| Core Kafka imports | 0 | grep verification |
| Handler unit test coverage | >80% | pytest-cov |
| MVP issue count | 24 | Linear tracking |

---

## Expected First PRs (Contributor Onboarding)

New contributors should tackle issues in this order to build momentum:

| PR # | Component | Repository | Complexity | Dependencies |
|------|-----------|------------|------------|--------------|
| 1 | `EnumHandlerType` | omnibase_core | S | None |
| 2 | `EnumNodeKind.RUNTIME_HOST` | omnibase_core | S | None |
| 3 | `ModelOnexEnvelope` | omnibase_core | M | Enums |
| 4 | `ProtocolHandler` | omnibase_core | M | Enums, Envelope |
| 5 | `ModelRuntimeHostContract` | omnibase_core | M | Enums |
| 6 | `ProtocolEventBus` export | omnibase_spi | S | Core protocols |
| 7 | `InMemoryEventBus` | omnibase_infra | M | SPI protocols |
| 8 | `HttpHandler` (skeleton) | omnibase_infra | M | ProtocolHandler |
| 9 | `HandlerDb` (skeleton) | omnibase_infra | M | ProtocolHandler |
| 10 | `wiring.py` | omnibase_infra | M | Handlers |
| 11 | `BaseRuntimeHostProcess` (skeleton) | omnibase_infra | L | All above |
| 12 | E2E integration test | omnibase_infra | L | All above |

**Complexity Legend**: S = Small (<50 LOC), M = Medium (50-200 LOC), L = Large (200+ LOC)

**First Week Goals**:
- Day 1-2: PRs 1-3 (enums and envelope)
- Day 3-4: PRs 4-6 (protocols)
- Day 5-7: PRs 7-9 (handlers)
- Week 2: PRs 10-12 (wiring and integration)

**Tip**: Each PR should include unit tests. Don't batch PRs - small, focused changes review faster.

---

## Minimum Reference Contract

The smallest working Runtime Host contract. Copy this to start a new runtime.

```yaml
# minimum_runtime_host.yaml
# The absolute minimum contract for a working MVP Runtime Host
# NO comments in production - this is for reference only

schema_version: "1.0.0"

runtime:
  name: "minimum_runtime_host"
  version: "1.0.0"

  event_bus:
    kind: "inmemory"

  handlers:
    - type: "http"
    - type: "db"

  nodes:
    - slug: "example_node"
```

**What this contract does**:
1. Uses `InMemoryEventBus` (MVP default)
2. Registers `HttpHandler` and `HandlerDb`
3. Loads one node contract (`example_node`)

**What this contract does NOT have** (intentionally):
- No `retry_policy` (Beta feature)
- No `rate_limit` (Beta feature)
- No `tenant_id` (Beta feature)
- No Kafka configuration (Beta feature)
- No Vault/Consul handlers (Beta feature)

**Startup command**:
```bash
omnibase-runtime-host minimum_runtime_host.yaml
```

**Expected startup logs**:
```
INFO  BaseRuntimeHostProcess initializing...
INFO  Loading contract: minimum_runtime_host.yaml
INFO  Registering handler: HttpHandler
INFO  Registering handler: HandlerDb
INFO  Loading node: example_node
INFO  Starting InMemoryEventBus...
INFO  BaseRuntimeHostProcess ready. Listening on topic: onex.*.*.cmd.v1
```

---

## Example Node Contracts

Minimal contracts for effect and compute nodes to contextualize handler usage.

**Effect Node Contract** (external I/O):
```yaml
# contracts/nodes/user_service_effect.yaml
schema_version: "1.0.0"
node_version: "1.0.0"
name: "user_service_effect"
node_type: "EFFECT_GENERIC"  # Use _GENERIC variants in contracts
description: "Fetches user data from external API"

input_model:
  name: "ModelUserRequest"
  fields:
    - name: "user_id"
      type: "string"
      required: true

output_model:
  name: "ModelUserResponse"
  fields:
    - name: "user_id"
      type: "string"
    - name: "email"
      type: "string"
    - name: "name"
      type: "string"

io_operations:
  - type: "http"
    operations: ["http.get"]
    target: "https://api.example.com/users"

dependencies: []
```

**Compute Node Contract** (data transformation):
```yaml
# contracts/nodes/order_calculator_compute.yaml
schema_version: "1.0.0"
node_version: "1.0.0"
name: "order_calculator_compute"
node_type: "COMPUTE_GENERIC"  # Use _GENERIC variants in contracts
description: "Calculates order totals and discounts"

input_model:
  name: "ModelOrderInput"
  fields:
    - name: "items"
      type: "list[OrderItem]"
      required: true
    - name: "discount_code"
      type: "string"
      required: false

output_model:
  name: "ModelOrderOutput"
  fields:
    - name: "subtotal"
      type: "decimal"
    - name: "discount"
      type: "decimal"
    - name: "total"
      type: "decimal"

# Compute nodes have NO io_operations (pure logic)
io_operations: []

dependencies: []
```

**Key Differences**:
| Aspect | Effect Node | Compute Node |
|--------|-------------|--------------|
| `node_type` | `EFFECT_GENERIC` | `COMPUTE_GENERIC` |
| `io_operations` | Lists external I/O | Empty (pure) |
| Handler usage | Uses HttpHandler, HandlerDb | No handlers (pure logic) |
| Side effects | Yes (external calls) | No (deterministic) |

---

> **Navigation**: [Back to Overview](../MVP_PLAN.md) | [Next: Beta Hardening](./BETA_v0.2.0_HARDENING.md)

**Last Updated**: 2025-12-03
