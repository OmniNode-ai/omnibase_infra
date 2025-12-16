# MVP Proposed Work Issues - omnibase_infra

> **Why This Document Exists**: This is the canonical infrastructure plan for shipping ONEX Runtime Host. If a feature, requirement, or constraint is not in this document, it is not happening in MVP. This document is the single source of truth for scope, architecture, and acceptance criteria. When in doubt, check this document.

**Repository**: omnibase_infra
**Generated**: 2025-12-03
**Updated**: 2025-12-04 (Split into milestone-specific documents)
**Linear Project**: MVP - ONEX Runtime Host Infrastructure

---

## Document Structure

This document has been split into milestone-specific files for easier navigation:

| Milestone | Version | Document | Issue Count |
|-----------|---------|----------|-------------|
| **MVP Core** | v0.1.0 | [MVP_v0.1.0_CORE.md](./milestones/MVP_v0.1.0_CORE.md) | 24 |
| **Beta Hardening** | v0.2.0 | [BETA_v0.2.0_HARDENING.md](./milestones/BETA_v0.2.0_HARDENING.md) | 22 |
| **Production** | v0.3.0 | [PRODUCTION_v0.3.0.md](./milestones/PRODUCTION_v0.3.0.md) | 8 |

---

## Milestone Overview

This document organizes issues into three milestones with progressive scope:

> **Core Principle**: Core is **pure orchestrator**; infra is **pure transport**. This boundary is inviolable.

| Milestone | Version | Focus | Issue Count | Timeline |
|-----------|---------|-------|-------------|----------|
| **MVP Core** | v0.1.0 | Minimal working runtime with InMemoryEventBus | 24 | Sprint 1-2 |
| **Beta Hardening** | v0.2.0 | Production handlers, Kafka, observability | 22 | Sprint 3-4 |
| **Production** | v0.3.0 | Full deployment, chaos testing, advanced features | 8 | Sprint 5 |

### MVP vs Beta vs Production Philosophy

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
+-------------------------------------------------------------------------+
|                           ONEX ARCHITECTURE                             |
+-------------------------------------------------------------------------+
|                                                                         |
|  +------------------------------------------------------------------+  |
|  |                     omnibase_infra (YOU ARE HERE)                |  |
|  |  +----------------+  +----------------+  +------------------+   |  |
|  |  | RuntimeHost    |  | Handlers       |  | EventBus         |   |  |
|  |  | Process        |  | - HttpHandler  |  | - InMemoryBus    |   |  |
|  |  |                |  | - DbHandler    |  | - KafkaBus (Beta)|   |  |
|  |  | Owns event     |  | - VaultAdapter |  |                  |   |  |
|  |  | loop + wiring  |  |   (Beta)       |  | Transport layer  |   |  |
|  |  +----------------+  +----------------+  +------------------+   |  |
|  +------------------------------------------------------------------+  |
|                                    |                                    |
|                                    v implements                         |
|  +------------------------------------------------------------------+  |
|  |                         omnibase_spi                              |  |
|  |  +--------------------+  +------------------------------------+  |  |
|  |  | ProtocolHandler    |  | ProtocolEventBus                   |  |  |
|  |  | (abstract)         |  | (abstract)                         |  |  |
|  |  +--------------------+  +------------------------------------+  |  |
|  |  Defines interfaces only. Zero implementations.                   |  |
|  +------------------------------------------------------------------+  |
|                                    |                                    |
|                                    v depends on                         |
|  +------------------------------------------------------------------+  |
|  |                         omnibase_core                             |  |
|  |  +----------------+  +----------------+  +------------------+   |  |
|  |  | NodeRuntime    |  | NodeInstance   |  | ModelOnex        |   |  |
|  |  |                |  |                |  | Envelope         |   |  |
|  |  | Routes         |  | Wraps node     |  |                  |   |  |
|  |  | envelopes to   |  | contracts      |  | Unified message  |   |  |
|  |  | handlers       |  |                |  | format           |   |  |
|  |  +----------------+  +----------------+  +------------------+   |  |
|  |  Pure orchestration. ZERO I/O. ZERO transport imports.           |  |
|  +------------------------------------------------------------------+  |
|                                                                         |
+-------------------------------------------------------------------------+

DEPENDENCY RULE: infra -> spi -> core (never reverse)
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

See [MVP_v0.1.0_CORE.md](./milestones/MVP_v0.1.0_CORE.md) for full contract format details and constraints.

See [BETA_v0.2.0_HARDENING.md](./milestones/BETA_v0.2.0_HARDENING.md) for Beta contract additions including Kafka, retry policies, and topic naming.

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

---

## Correlation ID Ownership Rules

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

## Success Metrics by Milestone

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

## How to Read This Document

Different roles need different sections:

| Role | Start Here | Focus On |
|------|------------|----------|
| **New Engineer** | This Overview -> Map of Abstractions | [MVP_v0.1.0_CORE.md](./milestones/MVP_v0.1.0_CORE.md) |
| **Infra Engineer** | Map of Abstractions | [MVP_v0.1.0_CORE.md](./milestones/MVP_v0.1.0_CORE.md) Phase 3 (Handlers) |
| **QA Engineer** | Testing sections in milestone docs | [BETA_v0.2.0_HARDENING.md](./milestones/BETA_v0.2.0_HARDENING.md) Phase 4 |
| **DevOps** | Deployment sections | [PRODUCTION_v0.3.0.md](./milestones/PRODUCTION_v0.3.0.md) Phase 5 |
| **Architect** | Architecture Invariants -> Failure Examples | All milestone docs |
| **Product Manager** | This Overview -> Success Metrics | Success Metrics sections |

**Quick Navigation**:
- "What is this?" -> This document (Overview)
- "What goes where?" -> Map of Abstractions (above)
- "What MUST work in MVP?" -> [MVP_v0.1.0_CORE.md](./milestones/MVP_v0.1.0_CORE.md)
- "What's added in Beta?" -> [BETA_v0.2.0_HARDENING.md](./milestones/BETA_v0.2.0_HARDENING.md)
- "What's needed for Production?" -> [PRODUCTION_v0.3.0.md](./milestones/PRODUCTION_v0.3.0.md)
- "What MUST NOT happen?" -> Failure Examples (above)

---

## Milestone Documents

### [MVP Core (v0.1.0)](./milestones/MVP_v0.1.0_CORE.md)

Contains:
- MVP philosophy and scope boundaries
- Simplified contract format details
- All 24 MVP issues with full details
- MVP execution order
- Minimum reference contract
- Example node contracts

### [Beta Hardening (v0.2.0)](./milestones/BETA_v0.2.0_HARDENING.md)

Contains:
- Beta philosophy and dependencies on MVP
- Extended contract format with Kafka, retry policies
- Topic naming schema and validation
- All 22 Beta issues with full details
- Beta execution order
- Considerations for future work

### [Production (v0.3.0)](./milestones/PRODUCTION_v0.3.0.md)

Contains:
- Production philosophy and dependencies
- All 8 Production issues with full details
- Chaos test scenarios (detailed)
- Kubernetes deployment specifications
- Migration checklist

---

**Last Updated**: 2025-12-04
**Document Owner**: OmniNode Architecture Team
**Linear Project URL**: TBD
