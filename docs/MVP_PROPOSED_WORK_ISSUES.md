# MVP Proposed Work Issues - omnibase_infra

> **Why This Document Exists**: This is the canonical infrastructure plan for shipping ONEX Runtime Host. If a feature, requirement, or constraint is not in this document, it is not happening in MVP. This document is the single source of truth for scope, architecture, and acceptance criteria. When in doubt, check this document.

**Repository**: omnibase_infra
**Generated**: 2025-12-03
**Updated**: 2025-12-03 (Milestone Split - Scope Reduction)
**Linear Project**: MVP - ONEX Runtime Host Infrastructure

---

## Milestone Overview

This document organizes issues into three milestones with progressive scope:

> **Core Principle**: Core is **pure orchestrator**; infra is **pure transport**. This boundary is inviolable.

| Milestone | Version | Focus | Issue Count | Timeline |
|-----------|---------|-------|-------------|----------|
| **MVP Core** | v0.1.0 | Minimal working runtime with InMemoryEventBus | 24 | Sprint 1-2 |
| **Beta Hardening** | v0.2.0 | Production handlers, Kafka, observability | 22 | Sprint 3-4 |
| **Production** | v0.3.0 | Full deployment, chaos testing, advanced features | 8 | Sprint 5 |

### MVP vs Beta Philosophy

**MVP (v0.1.0)**: Prove the architecture works end-to-end with minimal scope
- InMemoryEventBus only (no Kafka complexity)
- HTTP + DB handlers only (no Vault, no Consul)
- Simplified contract format
- Basic error handling
- Unit tests with mocks

**Beta (v0.2.0)**: Harden for production use
- KafkaEventBus with backpressure
- Vault + Consul handlers
- Full retry/rate-limit policies
- Integration tests with real services
- Observability layer

**Production (v0.3.0)**: Deploy and validate at scale
- Kubernetes manifests
- Chaos testing
- Performance benchmarks
- Complete documentation

**Terminology Clarification**:
- **"Minimal"** = Temporary scaffolding with known limitations. Will be expanded in Beta.
- **"Simple"** = Intentionally small feature set that may remain unchanged.
- When an issue says "MINIMAL", expect it to grow. When it says "SIMPLE", expect it to stay small.

---

## Summary Statistics

| Phase | MVP (v0.1.0) | Beta (v0.2.0) | Production (v0.3.0) |
|-------|--------------|---------------|---------------------|
| Phase 0: CI Guardrails | 2 | 2 | 0 |
| Phase 1: Core Types | 9 | 2 | 0 |
| Phase 2: SPI Updates | 3 | 0 | 0 |
| Phase 3: Infrastructure | 8 | 10 | 0 |
| Phase 4: Testing | 2 | 5 | 3 |
| Phase 5: Deployment | 2 | 3 | 2 |
| **Total** | **24** | **22** | **8** |

---

## Map of Abstractions

Understanding which component lives where is critical. This diagram shows the dependency flow:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ONEX ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     omnibase_infra (YOU ARE HERE)                │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │  │
│  │  │ RuntimeHost    │  │ Handlers       │  │ EventBus         │   │  │
│  │  │ Process        │  │ • HttpHandler  │  │ • InMemoryBus    │   │  │
│  │  │                │  │ • DbHandler    │  │ • KafkaBus (Beta)│   │  │
│  │  │ Owns event     │  │ • VaultHandler │  │                  │   │  │
│  │  │ loop + wiring  │  │   (Beta)       │  │ Transport layer  │   │  │
│  │  └────────────────┘  └────────────────┘  └──────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼ implements                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         omnibase_spi                              │  │
│  │  ┌────────────────────┐  ┌────────────────────────────────────┐  │  │
│  │  │ ProtocolHandler    │  │ ProtocolEventBus                   │  │  │
│  │  │ (abstract)         │  │ (abstract)                         │  │  │
│  │  └────────────────────┘  └────────────────────────────────────┘  │  │
│  │  Defines interfaces only. Zero implementations.                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼ depends on                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         omnibase_core                             │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │  │
│  │  │ NodeRuntime    │  │ NodeInstance   │  │ ModelOnex        │   │  │
│  │  │                │  │                │  │ Envelope         │   │  │
│  │  │ Routes         │  │ Wraps node     │  │                  │   │  │
│  │  │ envelopes to   │  │ contracts      │  │ Unified message  │   │  │
│  │  │ handlers       │  │                │  │ format           │   │  │
│  │  └────────────────┘  └────────────────┘  └──────────────────┘   │  │
│  │  Pure orchestration. ZERO I/O. ZERO transport imports.           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

DEPENDENCY RULE: infra → spi → core (never reverse)
```

**Repository Ownership Summary**:

| Component | Repository | Responsibility |
|-----------|------------|----------------|
| `NodeRuntime` | `omnibase_core` | Routes envelopes to handlers. NO I/O. |
| `NodeInstance` | `omnibase_core` | Wraps node contracts. Delegates to runtime. |
| `ModelOnexEnvelope` | `omnibase_core` | Unified message format. JSON serializable. |
| `ProtocolHandler` | `omnibase_spi` | Abstract interface for handlers. |
| `ProtocolEventBus` | `omnibase_spi` | Abstract interface for event buses. |
| `BaseBaseRuntimeHostProcess` | `omnibase_infra` | Owns event loop. Wires handlers. Drives runtime. Base class for app-specific hosts. |
| `HttpHandler`, `DbHandler` | `omnibase_infra` | Concrete handler implementations. |
| `InMemoryEventBus` | `omnibase_infra` | Concrete event bus for MVP. |

**The Golden Rule**:
```
CORE defines models and orchestration.
SPI defines abstract protocols.
INFRA implements handlers + event bus + BaseBaseRuntimeHostProcess bootstrap.
```

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
- `handlers.type: "vault"` - VaultHandler deferred to Beta
- `handlers.type: "consul"` - ConsulHandler deferred to Beta

**Cross-Version Compatibility**:
- v0.2 infra MUST accept v0.1 contracts but MUST warn on deprecated fields
- v0.1 contracts MUST NOT use any Beta-only fields
- Contract version mismatch MUST produce clear error message with upgrade instructions

**Beta Contract Additions:**
```yaml
runtime:
  name: "my_runtime_host"
  version: "1.0.0"

  event_bus:
    kind: "kafka"
    config:
      bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS}"
      consumer_group: "runtime-host-group"
      topics:
        input: "onex.tenant.domain.cmd.v1"
        output: "onex.tenant.domain.evt.v1"

  handlers:
    - type: "http"
      config:
        timeout_ms: 30000
        retry_policy:
          max_retries: 3
          backoff_strategy: "exponential"
    - type: "db"
      config:
        pool_size: 10
    - type: "vault"
      config:
        address: "${VAULT_ADDR}"
    - type: "consul"
      config:
        address: "${CONSUL_ADDR}"

  nodes:
    - slug: "node_a"
    - slug: "node_b"
```

**Topic Naming Schema**:

Topics follow the pattern: `onex.<tenant>.<domain>.<direction>.v<version>`

**Required Directions**:
- `.cmd` - Command/input topic (requests INTO the runtime)
- `.evt` - Event/output topic (responses FROM the runtime)

**MVP Validation Rules** (enforced by contract importer):
- No invalid characters (alphanumeric, dots, hyphens only)
- Version segment must be numeric (e.g., `v1`, `v2`)
- Tenant and domain segments required

**Examples**:
- `onex.acme.orders.cmd.v1` - Input commands
- `onex.acme.orders.evt.v1` - Output events

**Reserved Namespaces** (MUST NOT be used by user contracts):
- `onex.internal.*` - System-internal communication
- `onex.debug.*` - Debug and diagnostics
- `onex.metrics.*` - Metrics and telemetry
- `onex.health.*` - Health check signals
- `onex.admin.*` - Administrative commands

**Allowed Characters in Segments**: `[a-z0-9_-]` (lowercase alphanumeric, underscore, hyphen)

**Topic Naming Examples**:

| Example | Valid? | Reason |
|---------|--------|--------|
| `onex.acme.orders.cmd.v1` | Valid | Correct format |
| `onex.acme.orders.evt.v1` | Valid | Correct format |
| `onex.acme.user-service.cmd.v1` | Valid | Hyphen allowed |
| `onex.acme.user_service.cmd.v1` | Valid | Underscore allowed |
| `onex.ACME.orders.cmd.v1` | Invalid | Uppercase not allowed |
| `onex.acme.orders.command.v1` | Invalid | Must be `cmd` or `evt` |
| `onex.acme.orders.cmd.1` | Invalid | Version must have `v` prefix |
| `onex.acme.orders.cmd` | Invalid | Missing version |
| `acme.orders.cmd.v1` | Invalid | Missing `onex.` prefix |
| `onex..orders.cmd.v1` | Invalid | Empty segment |
| `onex.acme.orders.query.v1` | Invalid | `query` not a valid direction |
| `onex.internal.debug.evt.v1` | Invalid | `internal` is reserved |

**Validation Regex**:
```python
TOPIC_PATTERN = re.compile(
    r"^onex\.[a-z0-9][a-z0-9_-]*\.[a-z0-9][a-z0-9_-]*\.(cmd|evt)\.v[0-9]+$"
)

def validate_topic(topic: str) -> bool:
    if topic.startswith("onex.internal.") or topic.startswith("onex.debug."):
        raise ValueError(f"Reserved namespace: {topic}")
    return bool(TOPIC_PATTERN.match(topic))
```

---

## Non-Goals (All Milestones)

To prevent scope creep, the following are explicitly **NOT** in scope:

- **No multi-region failover** - Single cluster only
- **No automatic topic creation** - Assumes infra bootstrap handled separately
- **No dynamic handler discovery** - Static wiring from contract only
- **No auto-migration of legacy nodes** - Handled by higher-level repos
- **No LLM handler** - Deferred to future milestone
- **No Consul-based service mesh** - Basic discovery only (Beta)

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
- Multiple handlers of the same type (one HttpHandler, one DbHandler only)
- Multi-topic subscription (single input topic, single output topic)
- Async background tasks in ANY MVP code
- Dynamic handler registration at runtime
- Handler hot-reload or reconfiguration
- Transactions in DbHandler (individual queries only)

**Operations MUST Error Loudly**:
- Attempting to register duplicate handler types → `HandlerConfigurationError`
- Attempting forbidden operations (e.g., `db.transaction`) → `InvalidOperationError`
- Missing required envelope fields → `EnvelopeValidationError`

---

## Beta Scope Boundaries (Do Not Implement in Beta)

The following are explicitly OUT OF SCOPE until Production (v0.3.0):

- **No multi-region failover** - Single cluster only
- **No automatic topic creation** - Manual infra bootstrap
- **No dynamic handler discovery** - Static wiring only

---

## Architectural Invariants Checklist

Before starting any issue, review these invariants:

- [ ] **Core is transport-agnostic**: `omnibase_core` has NO Kafka/HTTP/DB/Vault imports
- [ ] **NodeRuntime is pure in-memory**: No event loop, no bus consumer
- [ ] **Event bus is infra concern**: `BaseRuntimeHostProcess` + `ProtocolEventBus` drives runtime
- [ ] **Handlers use strong typing**: `handler_type` returns `EnumHandlerType`, not `str`
- [ ] **LocalHandler is dev/test only**: Never in production contracts
- [ ] **Single source of truth**: `wiring.py` for handler registration
- [ ] **No secrets in contracts**: All secrets via `secret_ref` or resolver
- [ ] **Single BaseRuntimeHostProcess per OS process**: No nested runtimes
- [ ] **Error boundary**: Infra MUST map all raw exceptions into core RuntimeHostError types before returning to NodeRuntime
- [ ] **No threads in handlers**: No handler may create threads or executors. All concurrency must be async + event bus
- [ ] **Payload opacity**: Infra never inspects payload interpretation; all semantics live in core
- [ ] **Correlation tracking**: If an incoming envelope has no correlation_id, the runtime MUST assign one
- [ ] **Correlation ID immutability**: Once assigned, correlation_id MUST NOT be modified by any component
- [ ] **No dynamic imports**: Handlers MUST NOT use `importlib` or dynamic module loading
- [ ] **No blocking I/O**: All I/O operations MUST be async. No `time.sleep()`, no synchronous HTTP/DB calls
- [ ] **No top-level awaits**: Module-level code MUST NOT await; all async work happens in lifecycle methods
- [ ] **No task creation in handlers**: Handlers MUST NOT call `asyncio.create_task()`. All concurrency is managed by BaseRuntimeHostProcess
- [ ] **Single-threaded MVP**: BaseRuntimeHostProcess MUST be single-threaded in MVP. No thread pools, no multiprocessing
- [ ] **Envelope immutability in core**: NodeRuntime MUST NOT mutate envelopes. Only handlers or event bus may enrich envelopes
- [ ] **Fail-fast contracts**: BaseRuntimeHostProcess MUST crash fast on malformed contracts. Fail early, fail loudly
- [ ] **No raw exceptions on bus**: BaseRuntimeHostProcess MUST NOT emit raw exceptions to the bus. All errors wrapped in response envelopes

---

## Blocking I/O Rules

**Blocking I/O Rules** (Binary Yes/No):

| Operation | Allowed in Handlers? | Allowed in Core? | Notes |
|-----------|---------------------|------------------|-------|
| `await asyncio.sleep()` | YES | NO | Use for backoff, rate limiting |
| `time.sleep()` | NO | NO | Blocks event loop - FORBIDDEN |
| `asyncpg` queries | YES | NO | Infra handler only |
| `httpx` async requests | YES | NO | Infra handler only |
| `requests.get()` | NO | NO | Synchronous HTTP - FORBIDDEN |
| `open()` file read | LIMITED | NO | Only during initialization |
| `os.environ.get()` | LIMITED | YES | Only during initialization |
| `logging.info()` | YES | YES | Sync logging is acceptable |
| `print()` | NO | NO | Use structured logging |
| `subprocess.run()` | NO | NO | Blocks - FORBIDDEN |
| `json.dumps()` | YES | YES | CPU-bound, not I/O |
| DNS resolution | ASYNC ONLY | NO | Use `aiodns` or let `httpx` handle |

**Initialization Exception**:
During `initialize()` lifecycle method, synchronous operations ARE allowed:
- Reading config files
- Environment variable access
- Initial connection pool creation (async preferred)
- One-time setup operations

After `initialize()` completes, ALL I/O must be async.

**Detection in CI**:
```bash
# CI check for forbidden synchronous patterns
grep -r "time.sleep\|requests\.\|subprocess\." src/omnibase_infra/handlers/
# Should return empty (excluding test files)
```

---

## Correlation ID Ownership Rules

**Correlation ID Ownership Rules**:

| Scenario | Behavior |
|----------|----------|
| Request envelope has `correlation_id` | MUST preserve exactly as-is |
| Request envelope missing `correlation_id` | BaseRuntimeHostProcess MUST assign new UUID |
| Handler receives envelope | MUST NOT modify `correlation_id` |
| Event bus receives envelope | MUST NOT modify `correlation_id` |
| Response envelope creation | MUST echo `correlation_id` from request |
| Error envelope creation | MUST echo `correlation_id` from request |

**Assignment Authority**:
- Only `BaseRuntimeHostProcess` may assign a new `correlation_id` (and only when missing)
- No other component (handler, event bus, NodeRuntime) may create or modify it
- All log entries MUST include `correlation_id` for traceability

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
| 6 | `FileRegistry` | Loads contracts from directory, fails on invalid | [ ] |
| 7 | `HttpHandler` | Executes GET/POST, returns response envelope | [ ] |
| 8 | `DbHandler` | Executes query/execute, returns response envelope | [ ] |
| 9 | `InMemoryEventBus` | Publishes and consumes envelopes per topic | [ ] |
| 10 | `BaseRuntimeHostProcess` | Loads contract, wires handlers, starts consumption | [ ] |
| 11 | `wiring.py` | Registers handlers from config without duplicates | [ ] |
| 12 | E2E Flow | Envelope in → handler executes → response out | [ ] |

**MVP Smoke Test** (single command to verify):
```bash
# This command MUST succeed for MVP to be complete
pytest tests/integration/test_e2e_envelope_flow.py -v
```

**Minimum Viable Configuration**:
- Event Bus: `InMemoryEventBus` only
- Handlers: `HttpHandler` + `DbHandler` only
- Nodes: At least 1 node contract loaded
- Contract: Simplified MVP schema (no retry_policy, no tenant_id)

---

## Failure Examples (What NOT to Do)

Concrete examples of invariant violations. These patterns MUST fail CI or cause startup errors.

### FAIL: Core importing transport library

```python
# FILE: omnibase_core/runtime/node_runtime.py
# THIS MUST FAIL CI

import asyncpg  # VIOLATION: Core cannot import transport libraries

class NodeRuntime:
    async def route_envelope(self, envelope):
        # This code should never exist in core
        conn = await asyncpg.connect()  # VIOLATION: I/O in core
```

**Why it fails**: `omnibase_core` must remain transport-agnostic. CI grep check blocks this.

---

### FAIL: Handler creating background tasks

```python
# FILE: omnibase_infra/handlers/http_handler.py
# THIS MUST FAIL CODE REVIEW

class HttpHandler(ProtocolHandler):
    async def execute(self, envelope):
        # VIOLATION: Handler spawning background task
        asyncio.create_task(self._background_work())  # FORBIDDEN
        return response_envelope
```

**Why it fails**: Handlers MUST NOT create tasks. BaseRuntimeHostProcess owns all concurrency.

---

### FAIL: Handler returning raw exception

```python
# FILE: omnibase_infra/handlers/db_handler.py
# THIS MUST FAIL RUNTIME

class DbHandler(ProtocolHandler):
    async def execute(self, envelope):
        try:
            result = await self._pool.execute(query)
        except asyncpg.PostgresError as e:
            raise e  # VIOLATION: Raw exception escaping handler

        # CORRECT:
        # except asyncpg.PostgresError as e:
        #     raise HandlerExecutionError(
        #         message=str(e),
        #         handler_type=self.handler_type,
        #         correlation_id=envelope.correlation_id,
        #     ) from e
```

**Why it fails**: Raw exceptions leak internal details. All errors must be `RuntimeHostError` subclasses.

---

### FAIL: NodeRuntime mutating envelope

```python
# FILE: omnibase_core/runtime/node_runtime.py
# THIS MUST FAIL CODE REVIEW

class NodeRuntime:
    def route_envelope(self, envelope):
        envelope.metadata["routed_at"] = datetime.now()  # VIOLATION
        envelope.correlation_id = uuid4()  # VIOLATION
        return self._dispatch(envelope)
```

**Why it fails**: NodeRuntime MUST NOT mutate envelopes. Only handlers or event bus may enrich.

---

### FAIL: LocalHandler in production contract

```yaml
# FILE: contracts/production_runtime.yaml
# THIS MUST FAIL STARTUP

runtime:
  name: "production_runtime"
  handlers:
    - type: "local"  # VIOLATION: LocalHandler forbidden in infra
    - type: "http"
    - type: "db"
```

**Why it fails**: BaseRuntimeHostProcess MUST fail fast if LocalHandler detected in production.

---

### FAIL: Blocking I/O in handler

```python
# FILE: omnibase_infra/handlers/http_handler.py
# THIS MUST FAIL CODE REVIEW

import requests  # VIOLATION: Synchronous HTTP library

class HttpHandler(ProtocolHandler):
    async def execute(self, envelope):
        # VIOLATION: Blocking call in async handler
        response = requests.get(url)  # Blocks event loop!

        # CORRECT: Use httpx (async)
        # async with httpx.AsyncClient() as client:
        #     response = await client.get(url)
```

**Why it fails**: Blocking I/O freezes the entire BaseRuntimeHostProcess. All I/O must be async.

---

## Phase 0: Cross-Repo Guardrails & Invariants

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

### MVP Issues (v0.1.0)

#### Issue 0.1: Create core transport import check [MVP]

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

#### Issue 0.2: Enforce LocalHandler dev-only usage [MVP]

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

### Beta Issues (v0.2.0)

#### Issue 0.3: Protocol ownership verification [BETA]

**Title**: Verify infra never declares new protocols
**Type**: Infrastructure
**Priority**: High
**Labels**: `architecture`, `ci`, `guardrails`
**Milestone**: v0.2.0 Beta

**Description**:
Ensure `omnibase_infra` only IMPLEMENTS protocols defined in `omnibase_spi`, never declares new abstract protocols.

**Acceptance Criteria**:
- [ ] AST or grep check for `class.*Protocol.*ABC` in infra
- [ ] Only allow concrete implementations of SPI protocols
- [ ] Clear error message on violation
- [ ] Runs in CI

---

#### Issue 0.4: Version compatibility matrix check [BETA]

**Title**: Runtime version compatibility verification
**Type**: Infrastructure
**Priority**: High
**Labels**: `ci`, `versioning`
**Milestone**: v0.2.0 Beta

**Description**:
On startup, verify that `omnibase_core` and `omnibase_spi` versions meet minimum requirements.

**Compatibility Matrix**:
- Infra v0.1.0 -> Core >=0.4.0, SPI >=0.3.0

**Acceptance Criteria**:
- [ ] `BaseRuntimeHostProcess` logs resolved versions on startup
- [ ] Fails fast with clear message if incompatible
- [ ] Version matrix documented in README
- [ ] CI test for version check logic

---

## Phase 1: Core Types (omnibase_core)

**Priority**: HIGH
**Dependencies**: Phase 0 CI in place
**Repository**: omnibase_core

### MVP Issues (v0.1.0)

#### Issue 1.1: Add RUNTIME_HOST to EnumNodeKind [MVP]

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

#### Issue 1.2: Create EnumHandlerType enum [MVP]

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

#### Issue 1.3: Create ModelOnexEnvelope model (SIMPLIFIED) [MVP]

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

#### Issue 1.4: Create ModelRuntimeHostContract model (SIMPLIFIED) [MVP]

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

#### Issue 1.5: Verify ProtocolHandler export from SPI [MVP]

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

#### Issue 1.6: Implement NodeInstance class [MVP]

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

#### Issue 1.7: Implement NodeRuntime class (MINIMAL) [MVP]

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

#### Issue 1.8: Implement FileRegistry class (SIMPLE) [MVP]

**Title**: Create FileRegistry for contract loading
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `runtime`, `core`
**Milestone**: v0.1.0 MVP

**Description**:
Create the file-based contract registry for loading node contracts from filesystem. **Simple for MVP - just load YAML to Pydantic, no skip rules.**

**File**: `src/omnibase_core/runtime/file_registry.py` (NEW)

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

#### Issue 1.9: Implement LocalHandler (dev/test only) [MVP]

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

#### Issue 1.10: Create dev/test CLI entry point (SIMPLE) [MVP]

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

#### Issue 1.11: Create minimal error taxonomy (core) [MVP]

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
- `HandlerConfigurationError`
- `SecretResolutionError`

**Acceptance Criteria**:
- [ ] All inherit from `OnexError`
- [ ] Structured fields for logging
- [ ] `correlation_id` optional field
- [ ] Unit tests for error construction
- [ ] mypy --strict passes

---

### Beta Issues (v0.2.0)

#### Issue 1.12: Create ModelHandlerBindingConfig model [BETA]

**Title**: Implement formalized handler config schema
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `model`, `core`
**Milestone**: v0.2.0 Beta

**Description**:
Create a formal schema for handler binding configuration with validation and defaults.

**File**: `src/omnibase_core/models/runtime/model_handler_binding_config.py` (NEW)

**Fields**:
- `handler_type: EnumHandlerType`
- `name: str` (optional, defaults to handler_type)
- `enabled: bool = True`
- `priority: int = 0`
- `config_ref: str | None` (reference to external config)
- `retry_policy: ModelRetryPolicy | None`
- `timeout_ms: int = 30000`
- `rate_limit_per_second: float | None`

**Sub-model** `ModelRetryPolicy`:
- `max_retries: int = 3`
- `backoff_strategy: Literal["fixed", "exponential"] = "exponential"`
- `base_delay_ms: int = 100`
- `max_delay_ms: int = 5000`

**Acceptance Criteria**:
- [ ] Full validation with Pydantic
- [ ] Sensible defaults for all optional fields
- [ ] Used by `ModelRuntimeHostContract.handlers`
- [ ] Unit tests for validation edge cases
- [ ] mypy --strict passes

---

#### Issue 1.13: Extend error taxonomy (core) [BETA]

**Title**: Complete core error hierarchy for runtime
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `errors`, `core`
**Milestone**: v0.2.0 Beta

**Description**:
Complete the error hierarchy with additional error types.

**File**: `src/omnibase_core/errors/runtime_errors.py` (UPDATE)

**Additional Classes**:
- `HandlerNotFoundError(RuntimeHostError)`
- `NodeNotFoundError(RuntimeHostError)`
- `EnvelopeValidationError(RuntimeHostError)`
- `HandlerConfigurationError(RuntimeHostError)`
- `SecretResolutionError(RuntimeHostError)`

**Acceptance Criteria**:
- [ ] All inherit from `RuntimeHostError`
- [ ] Structured fields: `handler_type`, `operation`, `correlation_id`
- [ ] NodeRuntime only sees abstract errors, not library exceptions
- [ ] Unit tests for each error class
- [ ] mypy --strict passes

---

## Phase 2: SPI Protocol Updates (omnibase_spi)

**Priority**: HIGH
**Dependencies**: Phase 1 complete
**Repository**: omnibase_spi

### MVP Issues (v0.1.0)

#### Issue 2.1: Export ProtocolHandler from SPI [MVP]

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

#### Issue 2.2: Update ProtocolEventBus with envelope methods [MVP]

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

#### Issue 2.3: Document handler vs event bus distinction [MVP]

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

## Phase 3: Infrastructure Handlers (omnibase_infra)

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
                    ┌─────────────────────────────────────────┐
                    │         HANDLER LIFECYCLE               │
                    └─────────────────────────────────────────┘

    ┌──────────┐    initialize()    ┌─────────────┐
    │ CREATED  │ ─────────────────► │ INITIALIZED │
    └──────────┘                    └─────────────┘
         │                                │
         │ (constructor called)           │ health_check() returns healthy
         │                                ▼
         │                          ┌─────────────┐
         │                          │   HEALTHY   │ ◄────┐
         │                          └─────────────┘      │
         │                                │              │
         │                                │ execute()    │ (success)
         │                                ▼              │
         │                          ┌─────────────┐      │
         │                          │  EXECUTING  │ ─────┘
         │                          └─────────────┘
         │                                │
         │                                │ (error or shutdown signal)
         │                                ▼
         │                          ┌─────────────┐
         └─────────────────────────►│  SHUTDOWN   │
              (any error during     └─────────────┘
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

### MVP Issues (v0.1.0)

#### Issue 3.1: Create handlers directory structure (SIMPLIFIED) [MVP]

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
- `handlers/vault_handler.py`
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

#### Issue 3.2: Implement HttpHandler (MINIMAL) [MVP]

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

#### Issue 3.3: Implement DbHandler (MINIMAL) [MVP]

**Title**: Create PostgreSQL database protocol handler
**Type**: Feature
**Priority**: High
**Labels**: `handler`, `infrastructure`, `database`
**Milestone**: v0.1.0 MVP

**Description**:
Implement `DbHandler` for PostgreSQL operations using asyncpg. **Minimal for MVP - query + execute only, no transactions.**

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
- DbHandler MUST validate parameter count matches placeholder count
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

#### Issue 3.4: Implement InMemoryEventBus [MVP]

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

#### Issue 3.5: Create wiring.py (SIMPLE) [MVP]

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

#### Issue 3.6: Implement BaseRuntimeHostProcess (MINIMAL) [MVP]

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

---

#### Issue 3.7: Create example runtime host contract (SIMPLE) [MVP]

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

#### Issue 3.8: Add production CLI entry point (SIMPLE) [MVP]

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

### Beta Issues (v0.2.0)

#### Issue 3.9: Implement VaultHandler [BETA]

**Title**: Create Vault secrets management protocol handler
**Type**: Feature
**Priority**: High
**Labels**: `handler`, `infrastructure`, `secrets`
**Milestone**: v0.2.0 Beta

**Description**:
Implement `VaultHandler` for HashiCorp Vault operations using hvac.

**File**: `src/omnibase_infra/handlers/vault_handler.py`

**Operations**: `get_secret`, `set_secret`, `delete_secret`, `list_secrets`

**Acceptance Criteria**:
- [ ] Returns `EnumHandlerType.VAULT` (not str)
- [ ] Uses hvac client
- [ ] KV v2 support
- [ ] Maps `hvac.VaultError` -> `HandlerExecutionError`
- [ ] Unit tests with mock client
- [ ] Integration test with dev Vault
- [ ] mypy --strict passes

---

#### Issue 3.10: Implement ConsulHandler [BETA]

**Title**: Create Consul service discovery protocol handler
**Type**: Feature
**Priority**: Medium
**Labels**: `handler`, `infrastructure`, `discovery`
**Milestone**: v0.2.0 Beta

**Description**:
Implement `ConsulHandler` for Consul service discovery operations.

**File**: `src/omnibase_infra/handlers/consul_handler.py`

**Operations**: `register_service`, `deregister_service`, `get_service`, `list_services`, `health_check_service`

**Acceptance Criteria**:
- [ ] Returns `EnumHandlerType.CONSUL` (not str)
- [ ] Uses python-consul or httpx
- [ ] Service registration/deregistration
- [ ] Health check integration
- [ ] Unit tests with mock responses
- [ ] Integration test with dev Consul
- [ ] mypy --strict passes

---

#### Issue 3.11: Implement KafkaEventBus [BETA]

**Title**: Create Kafka event bus implementation
**Type**: Feature
**Priority**: High
**Labels**: `event-bus`, `infrastructure`, `messaging`
**Milestone**: v0.2.0 Beta

**Description**:
Implement `KafkaEventBus` that implements `ProtocolEventBus` (NOT ProtocolHandler).

**File**: `src/omnibase_infra/event_bus/kafka_event_bus.py`

**Methods**: `initialize`, `shutdown`, `publish_envelope`, `subscribe`, `start_consuming`, `health_check`

**Backpressure Config**:
- `max_inflight_envelopes: int = 100`
- `pause_consumption_threshold: int = 80`
- `circuit_breaker_threshold: int = 5` (consecutive failures)

**Ordering Guarantees**: Runtime MUST NOT assume global ordering. Kafka provides partition-level ordering only.

**Response Pattern**: Event bus handlers must return envelopes; BaseRuntimeHostProcess is responsible for publishing responses.

**Acceptance Criteria**:
- [ ] Implements `ProtocolEventBus` (NOT ProtocolHandler)
- [ ] NO `handler_type` property (event bus, not handler)
- [ ] Uses aiokafka
- [ ] Proper envelope serialization/deserialization
- [ ] Configurable topics and consumer group
- [ ] Backpressure: pauses consumption when queue full
- [ ] Circuit breaker: stops after N consecutive handler failures
- [ ] Unit tests with mock Kafka
- [ ] Integration test with test Kafka
- [ ] mypy --strict passes

---

#### Issue 3.12: Create SecretResolver [BETA]

**Title**: Implement centralized secret resolution
**Type**: Feature
**Priority**: High
**Labels**: `infrastructure`, `secrets`
**Milestone**: v0.2.0 Beta

**Description**:
Create a centralized secret resolver so handlers never call `os.getenv` directly.

**File**: `src/omnibase_infra/runtime/secret_resolver.py`

**Interface**:
```python
class SecretResolver:
    async def get_secret(self, logical_name: str) -> str: ...
    async def get_secrets(self, logical_names: list[str]) -> dict[str, str]: ...
```

**Sources** (priority order):
1. Vault (if configured)
2. Environment variables
3. File-based secrets (K8s secrets volume)

**Acceptance Criteria**:
- [ ] Typed interface for secret requests
- [ ] Handlers never call `os.getenv` directly
- [ ] Vault integration optional
- [ ] Unit tests with mocked sources
- [ ] mypy --strict passes

---

#### Issue 3.13: Create HandlerConfigResolver [BETA]

**Title**: Implement handler config resolution layer
**Type**: Feature
**Priority**: High
**Labels**: `infrastructure`, `configuration`
**Milestone**: v0.2.0 Beta

**Description**:
Create a resolver that normalizes handler configs from multiple sources.

**File**: `src/omnibase_infra/runtime/handler_config_resolver.py`

**Sources**:
- Contract YAML
- Environment variables (override)
- Vault (secrets only via SecretResolver)
- Files (config refs)

**Acceptance Criteria**:
- [ ] Resolves `config_ref` to actual config dict
- [ ] Merges environment overrides
- [ ] Returns fully validated `ModelHandlerBindingConfig`
- [ ] Clear error messages for missing required fields
- [ ] Unit tests for resolution logic
- [ ] mypy --strict passes

---

#### Issue 3.14: Extend infra error hierarchy [BETA]

**Title**: Complete infrastructure error taxonomy
**Type**: Feature
**Priority**: High
**Labels**: `infrastructure`, `errors`
**Milestone**: v0.2.0 Beta

**Description**:
Complete structured error hierarchy for all infra components.

**File**: `src/omnibase_infra/errors/infra_errors.py` (UPDATE)

**Additional Classes**:
- `HandlerConfigurationError(RuntimeHostError)`
- `SecretResolutionError(RuntimeHostError)`

**Acceptance Criteria**:
- [ ] All inherit from core `RuntimeHostError`
- [ ] Handlers map raw exceptions to these types
- [ ] Structured fields: `handler_type`, `operation`, `correlation_id`
- [ ] Unit tests for each error class
- [ ] mypy --strict passes

---

#### Issue 3.15: Implement observability layer [BETA]

**Title**: Create structured logging and metrics infrastructure
**Type**: Feature
**Priority**: High
**Labels**: `infrastructure`, `observability`
**Milestone**: v0.2.0 Beta

**Description**:
Create centralized observability configuration owned by BaseRuntimeHostProcess.

**Files**:
- `src/omnibase_infra/observability/logging_config.py`
- `src/omnibase_infra/observability/metrics.py`

**Logging Requirements**:
- JSON structured logs
- Standard fields: `runtime_id`, `node_id`, `handler_type`, `envelope_id`, `correlation_id`
- Handlers use shared logger (no ad-hoc loggers)

**Metrics Hooks**:
- Envelope count in/out per handler
- Handler latency histogram
- Event bus lag metrics (consumer group offset)

**Acceptance Criteria**:
- [ ] `BaseRuntimeHostProcess` sets up global logging config
- [ ] Handlers get loggers from centralized factory
- [ ] All logs include correlation_id when available
- [ ] Metrics exposed for collection
- [ ] Unit tests for log formatting
- [ ] mypy --strict passes

---

#### Issue 3.16: Implement health HTTP endpoint [BETA]

**Title**: Create HTTP health endpoint server
**Type**: Feature
**Priority**: Medium
**Labels**: `infrastructure`, `health`
**Milestone**: v0.2.0 Beta

**Description**:
Optional HTTP server exposing health endpoints for K8s probes.

**File**: `src/omnibase_infra/runtime/health_server.py`

**Endpoints**:
- `/health/live` - Is process alive
- `/health/ready` - Can process envelopes
- `/health/handlers` - Per-handler status snapshot

**Acceptance Criteria**:
- [ ] Optional (can be disabled in config)
- [ ] Configurable port
- [ ] Minimal dependencies (use `aiohttp` or built-in)
- [ ] Integrates with Docker healthchecks
- [ ] Integrates with K8s probes
- [ ] Unit tests
- [ ] mypy --strict passes

---

#### Issue 3.17: Implement handler retry wrapper [BETA]

**Title**: Create generic retry/rate-limit wrapper for handlers
**Type**: Feature
**Priority**: Medium
**Labels**: `infrastructure`, `handlers`
**Milestone**: v0.2.0 Beta

**Description**:
Create a decorator/wrapper that adds retry and rate limiting to any handler.

**File**: `src/omnibase_infra/handlers/handler_wrapper.py`

**Features**:
- Retry with configurable backoff (from `ModelRetryPolicy`)
- Token bucket rate limiting
- Circuit breaker pattern

**Acceptance Criteria**:
- [ ] Reads config from `ModelHandlerBindingConfig`
- [ ] Wraps `handler.execute()` transparently
- [ ] Logs retry attempts
- [ ] Rate limit enforced per handler instance
- [ ] Circuit breaker trips after threshold
- [ ] Unit tests for retry logic
- [ ] mypy --strict passes

---

#### Issue 3.18: Add contract schema and linting [BETA]

**Title**: Create JSON Schema for runtime host contracts
**Type**: Feature
**Priority**: Medium
**Labels**: `infrastructure`, `validation`
**Milestone**: v0.2.0 Beta

**Description**:
Create validation tooling for runtime host contracts.

**Files**:
- `src/omnibase_infra/contracts/schema/runtime_host_contract.schema.json`
- `src/omnibase_infra/cli/validate_contract.py`

**Lint Rules**:
- No embedded secrets (require `secret_ref`)
- No `LOCAL` handler in production contracts
- Topic names follow naming schema

**CLI**: `omnibase-runtime-validate-contract PATH`

**Acceptance Criteria**:
- [ ] JSON Schema created
- [ ] CLI validates contracts
- [ ] Lint rules enforced
- [ ] CI integration
- [ ] Unit tests for validation
- [ ] mypy --strict passes

---

## Phase 4: Integration & Testing

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

## Testing Without Docker

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
def db_handler(postgres_container):
    """DbHandler with real PostgreSQL."""
    handler = DbHandler()
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
def mock_db_handler():
    """DbHandler with mocked pool."""
    handler = DbHandler()
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

### MVP Issues (v0.1.0)

#### Issue 4.1: Unit tests for handlers (mocked) [MVP]

**Title**: Unit test coverage for MVP handlers
**Type**: Testing
**Priority**: High
**Labels**: `testing`, `handlers`
**Milestone**: v0.1.0 MVP

**Description**:
Unit tests for HttpHandler and DbHandler with mocked dependencies.

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

#### Issue 4.2: Unit tests for InMemoryEventBus [MVP]

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

#### Issue 4.3: Single E2E flow test with InMemoryEventBus [MVP]

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

#### Issue 4.4: Architecture compliance verification (SIMPLIFIED) [MVP]

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

### Beta Issues (v0.2.0)

#### Issue 4.5: Unit tests for KafkaEventBus [BETA]

**Title**: Unit tests for KafkaEventBus
**Type**: Testing
**Priority**: High
**Labels**: `testing`, `event-bus`
**Milestone**: v0.2.0 Beta

**Description**:
Unit tests for KafkaEventBus with mocked Kafka.

**File**: `tests/unit/event_bus/test_kafka_event_bus.py`

**Mock Requirements**: MockKafka MUST match aiokafka interface exactly. This prevents Beta teams from mocking the wrong subset of APIs.

**Acceptance Criteria**:
- [ ] Test envelope publishing
- [ ] Test subscription and consumption
- [ ] Test backpressure behavior
- [ ] Test circuit breaker
- [ ] Test error handling
- [ ] Test health check
- [ ] >90% coverage
- [ ] MockKafka implements complete aiokafka interface
- [ ] No aiokafka methods missing from mock

---

#### Issue 4.6: Integration tests with Docker [BETA]

**Title**: Integration tests with real services
**Type**: Testing
**Priority**: High
**Labels**: `testing`, `integration`, `docker`
**Milestone**: v0.2.0 Beta

**Description**:
Integration tests using docker-compose with real PostgreSQL, Kafka, Vault.

**File**: `tests/integration/test_runtime_host_integration.py`

**Tests**:
- DbHandler with real PostgreSQL
- VaultHandler with real Vault
- KafkaEventBus with real Kafka
- Full envelope flow

**Test Performance Requirements**:
- Integration tests must complete within 3 seconds per test case.
- Tests MUST NOT contain sleeps >50ms (use proper async waiting).
- Flaky tests are not acceptable - deterministic behavior required.

**Acceptance Criteria**:
- [ ] docker-compose.test.yaml with all services
- [ ] pytest fixtures for service setup
- [ ] Tests pass in CI
- [ ] Cleanup after tests
- [ ] Each test completes in <3 seconds
- [ ] No sleeps >50ms in test code
- [ ] Tests pass consistently (0 flaky tests)

---

#### Issue 4.7: Graceful shutdown tests [BETA]

**Title**: Test graceful shutdown behavior
**Type**: Testing
**Priority**: High
**Labels**: `testing`, `shutdown`
**Milestone**: v0.2.0 Beta

**Description**:
Verify shutdown semantics under load.

**File**: `tests/integration/test_graceful_shutdown.py`

**Scenarios**:
- SIGTERM while 50 envelopes in progress
- Verify no envelopes dropped
- Verify handlers' `shutdown()` called exactly once
- Verify exit within grace period

**Acceptance Criteria**:
- [ ] Simulate SIGTERM under load
- [ ] Assert no silent envelope drops
- [ ] Assert all handlers shutdown
- [ ] Assert exit within configured timeout

---

#### Issue 4.8: Backpressure and overload tests [BETA]

**Title**: Test backpressure under overload
**Type**: Testing
**Priority**: Medium
**Labels**: `testing`, `performance`
**Milestone**: v0.2.0 Beta

**Description**:
Verify backpressure behavior when Kafka produces faster than handlers consume.

**File**: `tests/integration/test_backpressure.py`

**Acceptance Criteria**:
- [ ] Simulate high-volume Kafka production
- [ ] Verify event bus pauses consumption
- [ ] Verify no unbounded memory growth
- [ ] Verify recovery when backlog clears

---

#### Issue 4.9: Topic naming validation tests [BETA]

**Title**: Validate Kafka topic naming schema
**Type**: Testing
**Priority**: Medium
**Labels**: `testing`, `kafka`
**Milestone**: v0.2.0 Beta

**Description**:
Verify all configured topics follow naming schema.

**Schema**: `onex.<tenant>.<domain>.<signal>.v<version>`
**Signals**: `cmd`, `evt`, `state`, `error`, `log`

**Acceptance Criteria**:
- [ ] Parse topics from all runtime contracts
- [ ] Validate tenant segment exists
- [ ] Validate signal is in allowed set
- [ ] CI integration

---

### Production Issues (v0.3.0)

#### Issue 4.10: Chaos and failure-mode tests [PROD]

**Title**: Create chaos test suite
**Type**: Testing
**Priority**: Medium
**Labels**: `testing`, `chaos`
**Milestone**: v0.3.0 Production

**Description**:
Test failure modes and recovery behavior.

**Directory**: `tests/chaos/`

**Scenarios**:
- Kafka connection loss (drop broker)
- Postgres unavailable for 5 seconds
- Vault network flakiness
- Handler returning errors continuously

**Acceptance Criteria**:
- [ ] BaseRuntimeHostProcess does not crash
- [ ] Retries within configured budgets
- [ ] Health endpoints reflect degraded state
- [ ] Circuit breaker trips appropriately
- [ ] Recovery after service restoration

---

#### Issue 4.11: Performance benchmarks [PROD]

**Title**: Establish performance baselines
**Type**: Testing
**Priority**: Medium
**Labels**: `testing`, `performance`
**Milestone**: v0.3.0 Production

**Description**:
Create benchmark suite to verify performance targets.

**Targets**:
- Memory per 10 nodes: <200MB
- Envelope throughput: >100/sec
- Handler latency (local): <1ms p99
- Handler latency (http): <100ms p99
- Handler latency (db): <50ms p99

**Acceptance Criteria**:
- [ ] Benchmark script created
- [ ] Results logged and tracked
- [ ] CI integration for regression detection
- [ ] Documentation of baselines

---

#### Issue 4.12: Complete architecture compliance [PROD]

**Title**: Full architectural invariant verification
**Type**: Task
**Priority**: High
**Labels**: `architecture`, `validation`
**Milestone**: v0.3.0 Production

**Description**:
Complete verification that all architectural invariants are maintained.

**Additional Checks**:
- [ ] All handlers return `EnumHandlerType`
- [ ] `wiring.py` is only handler registration location
- [ ] No `os.getenv` in handlers (uses SecretResolver)
- [ ] Single BaseRuntimeHostProcess per process enforcement

**Acceptance Criteria**:
- [ ] Shell script or pytest for verification
- [ ] Runs in CI
- [ ] Clear error messages on failure

---

## Phase 5: Deployment & Migration

**Priority**: MEDIUM
**Dependencies**: Phase 4 complete
**Repository**: omnibase_infra

### MVP Issues (v0.1.0)

#### Issue 5.1: Create Dockerfile for runtime host (BASIC) [MVP]

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
- All pip dependencies MUST be locked via requirements.txt or poetry.lock

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

#### Issue 5.2: Create docker-compose for local development [MVP]

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

#### Issue 5.3: Update CLAUDE.md with new architecture (MINIMAL) [MVP]

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

### Beta Issues (v0.2.0)

#### Issue 5.4: Expand docker-compose with full services [BETA]

**Title**: Add Kafka, Vault, Consul to docker-compose
**Type**: DevOps
**Priority**: Medium
**Labels**: `deployment`, `docker`, `development`
**Milestone**: v0.2.0 Beta

**Description**:
Expand docker-compose.yaml with all dependent services.

**Additional Services**:
- kafka (redpanda)
- vault (dev mode)
- consul (dev mode)

**Acceptance Criteria**:
- [ ] All services defined
- [ ] Health checks configured
- [ ] Volumes for persistence
- [ ] Service dependencies correct

---

#### Issue 5.5: Document CLIs and environments [BETA]

**Title**: Create CLIs and Environments documentation
**Type**: Documentation
**Priority**: Medium
**Labels**: `documentation`
**Milestone**: v0.2.0 Beta

**Description**:
Document the separation between dev and prod CLIs.

**File**: `docs/CLI_ENVIRONMENTS.md`

**Content**:
- `omninode-runtime-host-dev` (core, dev only, LocalHandler)
- `omnibase-runtime-host` (infra, production, no LocalHandler)
- Environment variables
- When to use each

**Acceptance Criteria**:
- [ ] Clear distinction documented
- [ ] Examples for each scenario
- [ ] Warning about prod CLI restrictions

---

#### Issue 5.6: Create version compatibility documentation [BETA]

**Title**: Document version compatibility matrix
**Type**: Documentation
**Priority**: Low
**Labels**: `documentation`, `versioning`
**Milestone**: v0.2.0 Beta

**Description**:
Document which versions of core/spi work with which infra versions.

**File**: `docs/VERSION_COMPATIBILITY.md`

**Acceptance Criteria**:
- [ ] Compatibility matrix table
- [ ] Minimum version requirements
- [ ] How to check versions at runtime
- [ ] Upgrade path documentation

---

### Production Issues (v0.3.0)

#### Issue 5.7: Kubernetes manifests [PROD]

**Title**: Create Kubernetes deployment manifests
**Type**: DevOps
**Priority**: Medium
**Labels**: `deployment`, `kubernetes`
**Milestone**: v0.3.0 Production

**Description**:
Create Kubernetes manifests for production deployment.

**Files**:
- `k8s/deployment.yaml`
- `k8s/service.yaml`
- `k8s/configmap.yaml`
- `k8s/secrets.yaml` (template)

**Acceptance Criteria**:
- [ ] Deployment with resource limits
- [ ] Service for internal access
- [ ] ConfigMap for contract
- [ ] Secret references documented
- [ ] Liveness probe: `/health/live`
- [ ] Readiness probe: `/health/ready`

---

#### Issue 5.8: Migration guide documentation [PROD]

**Title**: Create migration guide from legacy architecture
**Type**: Documentation
**Priority**: Medium
**Labels**: `documentation`, `migration`
**Milestone**: v0.3.0 Production

**Description**:
Document the migration path from 1-container-per-node to Runtime Host model.

**File**: `docs/MIGRATION_GUIDE.md`

**Acceptance Criteria**:
- [ ] Before/after architecture comparison
- [ ] Step-by-step migration process
- [ ] Rollback procedures
- [ ] Common issues and solutions

---

## Execution Order

### MVP (v0.1.0) - Sprint 1-2

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
    +-- 1.8 FileRegistry (simple) [MVP]
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
    +-- 3.3 DbHandler (minimal) [MVP]
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

### Beta (v0.2.0) - Sprint 3-4

```
Phase 0 (CI Guardrails - Beta)
    |
    +-- 0.3 Protocol ownership verification [BETA]
    +-- 0.4 Version compatibility check [BETA]
    |
    v
Phase 1 (Core Types - Beta)
    |
    +-- 1.12 ModelHandlerBindingConfig [BETA]
    +-- 1.13 Extended error taxonomy [BETA]
    |
    v
Phase 3 (Infra - Beta)
    |
    +-- 3.9 VaultHandler [BETA]
    +-- 3.10 ConsulHandler [BETA]
    +-- 3.11 KafkaEventBus [BETA]
    +-- 3.12 SecretResolver [BETA]
    +-- 3.13 HandlerConfigResolver [BETA]
    +-- 3.14 Extended error hierarchy [BETA]
    +-- 3.15 Observability layer [BETA]
    +-- 3.16 Health HTTP endpoint [BETA]
    +-- 3.17 Handler retry wrapper [BETA]
    +-- 3.18 Contract schema/linting [BETA]
    |
    v
Phase 4 (Testing - Beta)
    |
    +-- 4.5 KafkaEventBus unit tests [BETA]
    +-- 4.6 Integration tests with Docker [BETA]
    +-- 4.7 Graceful shutdown tests [BETA]
    +-- 4.8 Backpressure tests [BETA]
    +-- 4.9 Topic naming validation [BETA]
    |
    v
Phase 5 (Deployment - Beta)
    |
    +-- 5.4 Expand docker-compose [BETA]
    +-- 5.5 CLI environments doc [BETA]
    +-- 5.6 Version compatibility doc [BETA]
    |
    v
v0.2.0 Release
```

### Production (v0.3.0) - Sprint 5

```
Phase 4 (Testing - Production)
    |
    +-- 4.10 Chaos tests [PROD]
    +-- 4.11 Performance benchmarks [PROD]
    +-- 4.12 Complete architecture compliance [PROD]
    |
    v
Phase 5 (Deployment - Production)
    |
    +-- 5.7 Kubernetes manifests [PROD]
    +-- 5.8 Migration guide [PROD]
    |
    v
v0.3.0 Release
```

---

## Success Metrics

### MVP (v0.1.0) Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| E2E flow works | Yes | Single test passes |
| Core Kafka imports | 0 | grep verification |
| Handler unit test coverage | >80% | pytest-cov |
| MVP issue count | 24 | Linear tracking |

### Beta (v0.2.0) Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test coverage | >90% | pytest-cov |
| Integration tests pass | Yes | CI |
| Graceful shutdown drain | 100% | No dropped envelopes |
| Circuit breaker trigger | <10s | After threshold failures |
| Architecture violations | 0 | CI checks |

### Production (v0.3.0) Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Memory per 10 nodes | <200MB | tracemalloc |
| Envelope throughput | >100/sec | Benchmark suite |
| Handler latency (local) | <1ms | p99 latency |
| Handler latency (http) | <100ms | p99 latency |
| Handler latency (db) | <50ms | p99 latency |
| Chaos test survival | 100% | No crashes |

---

## Issue Creation Guidelines

When creating these issues in Linear:

1. **Team**: Omninode
2. **Project**: MVP - ONEX Runtime Host Infrastructure
3. **Labels**: Apply as indicated + milestone tag (`mvp`, `beta`, `production`)
4. **Priority**:
   - 1 = Urgent
   - 2 = High
   - 3 = Normal
   - 4 = Low
5. **Dependencies**: Link related issues where indicated
6. **Repository**: Tag with appropriate repo (omnibase_core, omnibase_spi, omnibase_infra)
7. **Milestone**: v0.1.0 MVP, v0.2.0 Beta, or v0.3.0 Production

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
2. Registers `HttpHandler` and `DbHandler`
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
INFO  Registering handler: DbHandler
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
node_type: "EFFECT"
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
node_type: "COMPUTE"
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
| `node_type` | `EFFECT` | `COMPUTE` |
| `io_operations` | Lists external I/O | Empty (pure) |
| Handler usage | Uses HttpHandler, DbHandler | No handlers (pure logic) |
| Side effects | Yes (external calls) | No (deterministic) |

---

## How to Read This Document

Different roles need different sections:

| Role | Start Here | Focus On |
|------|------------|----------|
| **New Engineer** | Executive Summary -> Glossary -> Map of Abstractions | Phase 1 (Core Types) |
| **Infra Engineer** | Map of Abstractions -> Phase 3 (Handlers) | BaseRuntimeHostProcess, Handlers |
| **QA Engineer** | Testing Infrastructure -> Phase 4 | Test requirements, E2E flows |
| **DevOps** | Phase 5 (Deployment) | Dockerfile, K8s, docker-compose |
| **Architect** | Architecture Invariants -> Failure Examples | Invariants, Risk Items |
| **Product Manager** | Executive Summary -> Milestone Overview | Success Metrics |

**Quick Navigation**:
- "What is this?" -> Executive Summary
- "What do the terms mean?" -> Glossary
- "What goes where?" -> Map of Abstractions
- "What MUST work?" -> MVP Constraints Checklist
- "What MUST NOT happen?" -> Failure Examples
- "What's the smallest working thing?" -> Minimum Reference Contract

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
| 9 | `DbHandler` (skeleton) | omnibase_infra | M | ProtocolHandler |
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

**Last Updated**: 2025-12-03
**Document Owner**: OmniNode Architecture Team
**Linear Project URL**: TBD
