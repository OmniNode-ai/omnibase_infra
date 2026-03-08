> **Navigation**: [Home](../index.md) > [Architecture](README.md) > Handler Classification POC 3.1

# POC 3.1: Postgres Mixin Handlerization Assessment

**Ticket**: OMN-4005
**Epic**: OMN-4014 — Epic 3: Mixin/Service -> Handler Refactoring
**Status**: Complete — KEEP AS MIXIN (abort conditions apply)
**Last Updated**: 2026-03-08

---

## Summary

This document records the outcome of POC 3.1: applying the OMN-4004 handler classification
rubric to `MixinPostgresOpExecutor` and `MixinPostgresErrorResponse`.

**Outcome**: Both mixins score 0–1 on the classification rubric. Per the OMN-4004 decision
matrix, the correct decision is **KEEP AS MIXIN**. Multiple abort conditions apply.
Execution of dependent tickets (OMN-4008, OMN-4009, OMN-4011) that depend on a net-positive
POC outcome should be reassessed individually against the rubric.

---

## Classification: MixinPostgresOpExecutor

**Location**: `src/omnibase_infra/mixins/mixin_postgres_op_executor.py`
**Callers**: 9 handler files in `node_contract_persistence_effect`, `node_delta_bundle_effect`,
`node_decision_store_effect`, `node_delta_metrics_effect`

### Rubric Scores

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| C1: I/O Ownership | **NO** | Mixin does NOT own any connection. It wraps `fn: Callable[[], Awaitable[T]]` provided by the calling handler. The caller (handler) owns the pool. The mixin has no `_pool`, no connection, no external system reference. |
| C2: Lifecycle Manageability | **NO** | No `initialize()` / `shutdown()` needed. Mixin is stateless — no resources to acquire or release. |
| C3: Dispatch Entry Point Clarity | **NO** | `_execute_postgres_op()` is a single entry point, but it receives the operation as a callable parameter from the handler. The handler already IS the dispatch owner. This is a helper function, not a dispatch boundary. |
| C4: Testability Without Subclassing | **NO** | The test suite already demonstrates the correct pattern: `class ConcreteExecutor(MixinPostgresOpExecutor): pass` — bare subclass with no pool. The mixin's behavior is fully testable via inheritance today, no injection needed. |
| C5: Cross-Layer Leakage Risk | **NO** | Inheriting this mixin grants no I/O capabilities. The handler still needs its own `pool` injected. The mixin cannot be used to perform I/O — it only processes exceptions and constructs results. |

**Total YES: 0 / 5** → **KEEP AS MIXIN**

### Why Handlerization Would Be Net-Negative

Converting `MixinPostgresOpExecutor` to a handler (e.g., `HandlerPostgresOpExecutor`) would require:

1. **Constructor change** on all 9 callers: `HandlerPostgresHeartbeat(pool)` → `HandlerPostgresHeartbeat(pool, executor: HandlerPostgresOpExecutor)`
2. **Wiring complexity**: 9 injection sites in the runtime wiring layer
3. **No behavioral gain**: The executor would still need to receive `fn` as a parameter — the calling handler would still own the pool. The injection adds a layer without clarifying ownership.
4. **Distorted ownership** (Abort Condition 3): `HandlerPostgresHeartbeat` would need to pass `self._execute_heartbeat` to an external `HandlerPostgresOpExecutor`, exposing its internal method to an outside object. This is a layering violation, not an improvement.

### What the Mixin Actually Is

`MixinPostgresOpExecutor` is a **utility function packaged as a mixin** for Python's
limitation that free functions can't be easily mixed into class hierarchies without
injection. Its single method `_execute_postgres_op` is essentially:

```python
async def execute_postgres_op(fn, op_error_code, correlation_id, log_context) -> ModelBackendResult:
    # timing + error classification + result construction
```

This is pure computation (timing + exception → result mapping). No I/O. No ownership.
The handler pattern is designed for I/O boundary units, not for computation helpers.

---

## Classification: MixinPostgresErrorResponse

**Location**: `src/omnibase_infra/mixins/mixin_postgres_error_response.py`
**Callers**: Subset of the same handler files (uses `_build_error_response`)

### Rubric Scores

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| C1: I/O Ownership | **NO** | Pure computation: exception → ModelBackendResult. No connections. |
| C2: Lifecycle Manageability | **NO** | Stateless. No resources. |
| C3: Dispatch Entry Point Clarity | **NO** | `_build_error_response` is called from catch blocks — it's error handling plumbing, not a dispatch point. |
| C4: Testability Without Subclassing | **NO** | Already testable via `class ConcreteHandler(MixinPostgresErrorResponse): pass`. |
| C5: Cross-Layer Leakage Risk | **NO** | No I/O capabilities granted. |

**Total YES: 0 / 5** → **KEEP AS MIXIN**

---

## Abort Conditions Assessment

Per OMN-4004 Section 7, the following abort conditions apply:

| Condition | Triggered? | Evidence |
|-----------|-----------|---------|
| Dispatch integration more complex than expected | YES | 9 callers, all needing constructor changes + wiring updates |
| Handler boundaries distort ownership | YES | Handler would need to pass its own method to the executor — layering violation |
| Test suite requires structural changes | YES | `test_mixin_postgres_op_executor.py` uses bare subclass pattern; converting would require redesigning the test approach |

---

## Impact on Wave 2 Tickets

Per the OMN-4004 dependency map, the following tickets are contingent on 3.1 demonstrating
net-positive handlerization:

| Ticket | Status | Reasoning |
|--------|--------|-----------|
| OMN-4008 (LLM transport mixin → handler) | **Reassess independently** | LLM transport owns an httpx client — C1=YES, C2=YES. May qualify independently. |
| OMN-4009 (projector mixins → handlers) | **Reassess independently** | Projector mixins may own Kafka/Postgres connections. Evaluate each separately. |
| OMN-4011 (ServiceTopicCatalogPostgres → handler) | **Proceed** | SERVICE → HANDLER is a different classification path. Services (not mixins) with owned connections are stronger handler candidates. |
| OMN-4010 (omnibase_core file I/O services → handlers) | **Reassess independently** | Evaluate each service individually against the rubric. |

**The 3.1 POC finding does NOT block Wave 2 tickets** — it only means the postgres-specific
mixin pattern is correctly a mixin. Other mixins and services with genuine I/O ownership
(LLM transport, projector notification publishing, service-level postgres) should be evaluated
independently.

---

## References

- [Handler Classification Rules](HANDLER_CLASSIFICATION_RULES.md) — OMN-4004 rubric
- `src/omnibase_infra/mixins/mixin_postgres_op_executor.py` — mixin under test
- `src/omnibase_infra/mixins/mixin_postgres_error_response.py` — mixin under test
- `tests/unit/mixins/test_mixin_postgres_op_executor.py` — existing test suite (passes)
- OMN-4014 — parent epic
