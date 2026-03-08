> **Navigation**: [Home](../index.md) > [Architecture](README.md) > Handler Classification Rules

# ONEX Handler Classification Rules (Epic 3 — Design Pre-Ticket 3.0)

**Ticket**: OMN-4004
**Epic**: OMN-4014 — Epic 3: Mixin/Service -> Handler Refactoring
**Status**: Final — governs all 3.x execution tickets
**Last Updated**: 2026-03-08

---

## Purpose

This document defines the classification rules used to determine whether a mixin or service
qualifies for refactoring to the ONEX handler pattern. It is the authoritative gate for all
3.x execution tickets (OMN-4005 through OMN-4011).

**Handlerization is successful only if** it preserves existing behavior AND produces a
clearer operational boundary for I/O ownership and dispatch. Passing tests alone are necessary
but not sufficient.

---

## 1. What Is a Handler?

A handler in ONEX is a first-class I/O boundary unit with:

1. **Single-purpose I/O ownership** — one external system per handler (postgres, kafka, LLM, filesystem)
2. **Defined lifecycle** — `initialize()` and `shutdown()` that containers can manage
3. **Contract-declared interface** — behavior declared in `handler_contract.yaml`, not implicit via inheritance
4. **No publish access** — handlers receive and process; they do not emit events
5. **Injectable and mockable** — handlers are injected at construction time, not inherited or composed via mixin

The canonical handler shape:

```python
class HandlerPostgresHeartbeat(ProtocolHandler):
    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def handle(self, payload: ModelPayload, correlation_id: str) -> ModelBackendResult: ...
```

---

## 2. Classification Rubric

Evaluate each mixin or service against the following five criteria. Score each as YES / NO / AMBIGUOUS.

### Criterion 1: I/O Ownership

**Question**: Does this code own a direct connection to an external system (database, HTTP endpoint,
filesystem path, message broker)?

- YES → strong signal for handler
- NO (pure computation, data transformation, state machine) → keep as mixin or utility
- AMBIGUOUS → proceed to Criterion 2

**Examples**:
- `MixinPostgresOpExecutor` — YES (owns PostgreSQL execute path)
- `MixinAsyncCircuitBreaker` — NO (pure state machine over any transport)
- `ServiceTopicCatalogPostgres` — YES (owns Postgres connection for topic catalog)

### Criterion 2: Lifecycle Manageability

**Question**: Does this code need `initialize()` / `shutdown()` lifecycle that the container
should manage?

- YES → strong signal for handler (connection pools, file handles, HTTP clients need cleanup)
- NO → mixin is appropriate (stateless logic, no teardown needed)

**Examples**:
- `MixinLlmHttpTransport` — YES (manages httpx.AsyncClient lifecycle)
- `MixinRetryExecution` — NO (stateless retry loop, no owned resources)

### Criterion 3: Dispatch Entry Point Clarity

**Question**: Is there a single, clear dispatch entry point that callers should use, vs inherited
methods scattered across the inheritance chain?

- Single entry point → handler is clearer
- Multiple inherited helpers legitimately composed by subclasses → mixin is correct

**Examples**:
- `MixinPostgresOpExecutor._execute_postgres_op()` — single entry point, callers always use it
- `MixinAsyncCircuitBreaker` — exposes multiple integration points (`call_with_breaker`,
  `get_circuit_status`, `reset_circuit`) intentionally used in varied compositions

### Criterion 4: Testability Without Subclassing

**Question**: Can this code be tested by injecting a mock/stub without requiring subclassing
or test fixtures that inherit from it?

- YES (inject + mock is natural) → handler is better
- NO (must subclass to test) → mixin composition may be intentional

### Criterion 5: Cross-Layer Leakage Risk

**Question**: Does inheriting this mixin grant the subclass capabilities that violate ONEX
layer boundaries (e.g., does a compute node gain I/O capabilities via mixin inheritance)?

- YES → handler refactoring is justified to prevent leakage
- NO → mixin composition is safe

---

## 3. Classification Decision Matrix

| Criteria Met (YES) | Recommendation |
|--------------------|---------------|
| 4–5 | **CONVERT to handler** — refactoring is clearly justified |
| 3 | **AMBIGUOUS** — requires proof-of-concept (3.1 pattern) before committing |
| 1–2 | **KEEP as mixin** — mixin composition is intentional and correct |
| 0 | **KEEP as mixin** — no case for handler pattern |

---

## 4. Canonical Mixin → Handler Migration Pattern

When classification yields CONVERT, follow this migration pattern:

**4a. Extract the Handler**

Create a new handler class in `src/omnibase_infra/handlers/` (or appropriate subpackage):

```python
# Before (mixin):
class MixinPostgresOpExecutor:
    async def _execute_postgres_op(self, op, ...) -> ModelBackendResult: ...

# After (handler):
class HandlerPostgresOpExecutor(ProtocolHandler):
    def __init__(self, pool: asyncpg.Pool, ...) -> None: ...
    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def execute(self, op: Callable, ...) -> ModelBackendResult: ...
```

**4b. Remove Mixin Inheritance from Callers**

Replace `class HandlerFoo(MixinX):` with `class HandlerFoo(ProtocolHandler):` and inject
the new handler:

```python
class HandlerPostgresHeartbeat(ProtocolHandler):
    def __init__(self, executor: HandlerPostgresOpExecutor) -> None:
        self._executor = executor

    async def handle(self, payload, correlation_id) -> ModelBackendResult:
        return await self._executor.execute(self._heartbeat_op, correlation_id)
```

**4c. Wire the Injection Point**

Update the handler contract YAML and wiring layer to inject the new handler dependency.

**4d. Verify Behavior Preservation**

Run the existing test suite. The behavior preservation test requires:
1. All unit tests passing without modification
2. At least **two observable signals** from the ambiguity rubric (Section 5) demonstrating
   reduced ambiguity — not just passing tests

**4e. No Shims**

Do not create backwards-compatible aliases (`MixinPostgresOpExecutor = HandlerPostgresOpExecutor`).
If callers must be updated, update them in the same PR.

---

## 5. Ambiguity Rubric — Observable Signals of Reduced Ambiguity

Handlerization is only net-positive if at least **two** of the following signals are demonstrable
after the refactoring:

| Signal | Description |
|--------|-------------|
| **S1: Hidden I/O eliminated** | Inherited I/O methods are no longer accessible to subclasses; only the injected handler's public API is available |
| **S2: Dispatch ownership clarified** | Single defined entry point replaces scattered inherited methods |
| **S3: Lifecycle enforced** | `initialize()` / `shutdown()` are called by the container; mixin teardown was previously ad-hoc |
| **S4: Testability improved** | Test can inject a mock handler without subclassing; previously required `super().__init__()` plumbing |
| **S5: Layer boundary respected** | Compute or orchestrator node no longer inherits I/O capability via mixin; handler is injected instead |

---

## 6. Explicit Keep-As-Mixin Decisions

The following patterns are **not candidates** for handlerization regardless of criteria score,
because their cross-cutting composition is intentional:

### MixinAsyncCircuitBreaker + MixinRetryExecution

**Decision: KEEP AS MIXIN**

Rationale:
- These are cross-cutting behavioral policies, not I/O boundaries
- They apply to *any* transport (postgres, kafka, HTTP, filesystem) via composition
- Converting to a handler would require one handler per transport type, creating N handlers
  where one mixin currently serves all
- Their state (circuit state, retry count) is per-caller-instance, not per-external-system
- No layer boundary violation — composing a circuit breaker into a handler does not grant
  the handler new I/O capabilities

If OMN-4006 (classification ticket) confirms this decision, OMN-4007 (execution ticket) is
closed without action.

### MixinDictLikeAccessors, MixinNodeIntrospection, MixinEnvelopeExtraction

**Decision: KEEP AS MIXIN**

Rationale: Pure computation, no I/O, no lifecycle, no layer boundary concerns.

---

## 7. Abort Conditions

Stop refactoring and revert if any of the following are true:

1. **Dispatch integration is more complex than expected** — if wiring the injected handler
   requires more scaffolding than the mixin it replaces, the net architectural benefit is negative
2. **Behavior preservation requires excessive shim layers** — if `__init__` keyword arguments,
   default parameter compatibility, or `super()` call chains must be preserved via shims
3. **Handler boundaries distort ownership** — if extracting the handler causes callers to
   expose their internal operation logic that was previously encapsulated in the mixin
4. **Test suite requires structural changes** — if more than 10 test files require modification
   solely due to the mixin-to-handler refactoring (not counting test additions)

---

## 8. Success Criteria

A handlerization is complete and successful when:

1. All pre-existing tests pass without modification (or with minimal fixture updates)
2. At least two ambiguity signals from Section 5 are demonstrably satisfied
3. No backwards-compatibility shims are introduced
4. The handler contract YAML exists and is validated by `onex-contract-validation`
5. The PR description includes the ambiguity signal evidence section

---

## 9. Ticket Dependency Map

```
OMN-4004 (this doc) — must complete first
  ├── OMN-4005 (3.1 POC: postgres mixins → handler) — gating ticket for all 3.x
  │     ├── OMN-4008 (LLM transport mixin → handler)
  │     ├── OMN-4009 (projector mixins → handlers)
  │     └── OMN-4011 (ServiceTopicCatalogPostgres → handler)
  ├── OMN-4006 (3.2a: classify circuit breaker + retry) — read-only investigation
  │     └── OMN-4007 (execute if 3.2a confirms refactoring justified)
  └── OMN-4010 (omnibase_core file I/O services → handlers) — contingent on 3.1
```

**Gate rule**: If OMN-4005 (3.1 POC) demonstrates negative net-architectural-clarity
(abort conditions triggered), all contingent tickets (OMN-4008, OMN-4009, OMN-4010, OMN-4011)
are closed without action.

---

## 10. References

- [Handler Protocol-Driven Architecture](HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md)
- [Current Node Architecture](CURRENT_NODE_ARCHITECTURE.md)
- `src/omnibase_infra/mixins/` — existing mixin implementations
- `src/omnibase_infra/services/` — existing service implementations
- OMN-4014 — parent epic
