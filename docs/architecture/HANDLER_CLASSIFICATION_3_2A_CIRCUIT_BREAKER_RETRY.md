> **Navigation**: [Home](../index.md) > [Architecture](README.md) > Handler Classification 3.2a — Circuit Breaker + Retry Mixins

# Handler Classification 3.2a — MixinAsyncCircuitBreaker + MixinRetryExecution

**Ticket**: OMN-4006
**Epic**: OMN-4014 — Epic 3: Mixin/Service -> Handler Refactoring
**Rubric**: [Handler Classification Rules](HANDLER_CLASSIFICATION_RULES.md) (OMN-4004)
**Status**: Final — KEEP AS MIXIN
**Last Updated**: 2026-03-08

---

## Purpose

This document provides the formal rubric-based classification of `MixinAsyncCircuitBreaker`
and `MixinRetryExecution` against the 3.0 handler rubric. It confirms or challenges the
pre-existing decision in `HANDLER_CLASSIFICATION_RULES.md` Section 6.

**Outcome of this classification determines whether OMN-4007 is executed or closed.**

---

## Callers Surveyed

`MixinAsyncCircuitBreaker` is used across **37+ production classes** across the codebase:

- **Dispatchers** (6): `DispatcherRuntimeTick`, `DispatcherTopicCatalogQuery`,
  `DispatcherNodeIntrospected`, `DispatcherCatalogRequest`, `DispatcherNodeHeartbeat`,
  `DispatcherNodeRegistrationAcked`
- **Runtime** (3): `RuntimeScheduler`, `RequestResponseWiring`, `TransitionNotificationPublisher`
- **Handlers** (4): `HandlerFileSystem`, `HandlerDb`, `HandlerMCP`, `HandlerQdrant`,
  `HandlerRegistrationStoragePostgres`
- **Services** (10+): `ServiceRetryWorker`, `ServiceCapabilityQuery`, `ServiceDlqTracking`,
  `EndpointCircuitBreaker`, multiple observability writers
- **Projectors** (3): `ProjectionReaderContract`, `SnapshotPublisherRegistration`,
  `ProjectionReaderRegistration`
- **Adapters** (2): `AdapterPipelineAlertBridge`, `AdapterONEXToolExecution`
- **Mixins** (1): `MixinLlmHttpTransport` (itself composing both mixins)

`MixinRetryExecution` is used by `MixinLlmHttpTransport` and `HandlerPostgres` pattern classes.

This breadth is a key signal: these mixins serve cross-cutting resilience concerns across
every I/O boundary type in the codebase.

---

## Rubric Scoring

### Criterion 1: I/O Ownership

**Question**: Does this code own a direct connection to an external system?

**MixinAsyncCircuitBreaker**:
- Owns only: `asyncio.Lock`, counters, timestamps — pure state machine
- It wraps *any* external call passed to it by the caller
- It has no knowledge of what transport it guards (Kafka, HTTP, Postgres, filesystem)
- Score: **NO**

**MixinRetryExecution**:
- Owns only: retry loop logic, backoff calculations, error classification dispatch
- It calls `func: Callable` provided by the caller — it does not hold any connection
- The `_executor: ThreadPoolExecutor` attribute is optional and caller-provided
- Score: **NO**

**Criterion 1 result**: NO for both. No case for handler based on this criterion alone.

---

### Criterion 2: Lifecycle Manageability

**Question**: Does this code need `initialize()` / `shutdown()` lifecycle the container should manage?

**MixinAsyncCircuitBreaker**:
- `_init_circuit_breaker()` is a regular init method, not a container lifecycle hook
- `cancel_active_recovery()` exists for shutdown cleanup, but it is called by the *caller*
  (the enclosing handler's `shutdown()`), not managed independently
- The mixin itself holds no connection pool, no file handle, no HTTP client
- Score: **NO**

**MixinRetryExecution**:
- Stateless retry logic — no owned resources requiring teardown
- Score: **NO**

**Criterion 2 result**: NO for both. No lifecycle that the container should own independently.

---

### Criterion 3: Dispatch Entry Point Clarity

**Question**: Is there a single, clear dispatch entry point vs inherited helpers composed by subclasses?

**MixinAsyncCircuitBreaker**:
- Exposes a deliberate multi-method API: `_check_circuit_breaker()`, `_record_circuit_failure()`,
  `_reset_circuit_breaker()`, `cancel_active_recovery()`, `get_circuit_breaker_state()`
- These are intentionally called at different points in the caller's try/except structure:
  check before, record on failure, reset on success
- A handler wrapping this would have to expose the same multi-step API — no net simplification
- Score: **NO** (multiple integration points are intentional)

**MixinRetryExecution**:
- Also multi-method: `_check_circuit_if_enabled()`, `_record_circuit_failure_if_enabled()`,
  `_reset_circuit_if_enabled()`, `_handle_retriable_error()`, `_handle_non_retriable_error()`
- These are composed per-operation in the calling class
- Score: **NO**

**Criterion 3 result**: NO for both. Scattered integration points are intentional composition, not a smell.

---

### Criterion 4: Testability Without Subclassing

**Question**: Can this code be tested by injecting a mock/stub without subclassing?

**MixinAsyncCircuitBreaker**:
- Currently tested by creating a minimal concrete subclass in test fixtures — this is normal
  for a mixin
- Converting to a handler would enable mock injection, but the testing ergonomics are already
  acceptable: the mixin's behavior is independently testable via a thin concrete class
- Score: **AMBIGUOUS** (handler would improve testability marginally)

**MixinRetryExecution**:
- Tested via concrete subclasses implementing the abstract methods
- Same assessment as above
- Score: **AMBIGUOUS**

**Criterion 4 result**: AMBIGUOUS, but marginally. Does not reach YES threshold required for
conversion recommendation.

---

### Criterion 5: Cross-Layer Leakage Risk

**Question**: Does inheriting this mixin grant the subclass capabilities that violate ONEX
layer boundaries?

**MixinAsyncCircuitBreaker**:
- It grants resilience policy capabilities, not I/O capabilities
- A compute node inheriting `MixinAsyncCircuitBreaker` does NOT gain database access, Kafka
  access, or any I/O capability — it gains only state-machine behavior
- The circuit breaker wraps operations the caller already performs; it does not add new ones
- No ONEX layer boundary violation is possible via this mixin
- Score: **NO**

**MixinRetryExecution**:
- Same — retry logic does not grant I/O capability; it wraps existing caller I/O operations
- Score: **NO**

**Criterion 5 result**: NO for both. No layer boundary leakage risk.

---

## Score Summary

| Criterion | MixinAsyncCircuitBreaker | MixinRetryExecution |
|-----------|--------------------------|---------------------|
| C1: I/O Ownership | NO | NO |
| C2: Lifecycle Manageability | NO | NO |
| C3: Dispatch Entry Point Clarity | NO | NO |
| C4: Testability Without Subclassing | AMBIGUOUS | AMBIGUOUS |
| C5: Cross-Layer Leakage Risk | NO | NO |
| **YES count** | **0** | **0** |
| **Recommendation** | **KEEP AS MIXIN** | **KEEP AS MIXIN** |

Per the decision matrix: 0 YES → **KEEP AS MIXIN**.

---

## Ambiguity Analysis (Addressing Ticket Risk Flag)

The ticket marked this as HIGH risk because these mixins are used in event bus, handlers, and
dispatchers. This risk flag refers to the *blast radius of a conversion* — not to whether
conversion is architecturally justified.

The specific ambiguity criteria from the rubric are addressed:

**"Cross-cutting composition adds no I/O boundary violation if the composition model is intentional"**

Confirmed: The composition model IS intentional. Evidence:

1. **Transport-agnostic design**: `MixinAsyncCircuitBreaker` accepts `EnumInfraTransportType`
   as a parameter at init time. It is explicitly designed to serve any transport.

2. **Caller-provided callable pattern**: Both mixins wrap `Callable` arguments passed by the
   caller. They never own the I/O operation. They add resilience policy around it.

3. **Per-instance state, not per-external-system state**: Circuit breaker state (`_circuit_breaker_failures`,
   `_circuit_breaker_open`) lives on the caller instance. If converted to a handler, one handler
   instance per external system would be needed, losing the ability to track per-caller failure
   patterns where the same external system has multiple callers with different thresholds.

4. **Breadth of use**: 37+ classes across 8 subsystems compose `MixinAsyncCircuitBreaker`.
   This is not accident — it is evidence of intentional cross-cutting policy application.

5. **OMN-4004 rubric pre-decision**: The classification rubric document itself (authored as the
   authoritative gate for all 3.x tickets) explicitly lists these mixins in Section 6
   "Explicit Keep-As-Mixin Decisions" with full rationale. This classification confirms that
   pre-decision is correct based on independent scoring.

---

## Decision

**MixinAsyncCircuitBreaker**: **KEEP AS MIXIN** — Score 0/5. Cross-cutting resilience policy
with no I/O ownership, no lifecycle, no layer boundary risk. Conversion would create N transport-specific
handlers where one universal mixin currently serves all.

**MixinRetryExecution**: **KEEP AS MIXIN** — Score 0/5. Stateless retry algorithm composed
into callers that own the I/O. No conversion value; conversion would fragment retry policy
across N handler types.

---

## Impact on OMN-4007

Per the ticket dependency rule:

> OMN-4007 (execute if 3.2a confirms refactoring justified) is executed **only** if this
> classification confirms refactoring is justified.

**This classification does NOT confirm refactoring is justified.**

**Decision: OMN-4007 is CLOSED WITHOUT ACTION.**

---

## Impact on Wave 2 Tickets (OMN-4008, OMN-4009, OMN-4011, OMN-4010)

This classification finding is specific to cross-cutting resilience policy mixins. It does NOT
generalize to the remaining wave 2 tickets, which address mixins/services with actual I/O
ownership. Those tickets should be assessed independently per the rubric.

Consistent with OMN-4005 finding: OMN-4005 found that `MixinPostgresOpExecutor` scored 0/5
(utility function mixin, not an I/O boundary owner). This 3.2a finding independently scores
0/5 for a different structural reason (cross-cutting policy, not I/O ownership).

The wave 2 tickets (`MixinLlmHttpTransport`, projector mixins, `ServiceTopicCatalogPostgres`,
file I/O services) address components with genuine I/O ownership and lifecycle and should be
evaluated on their own merits.

---

## References

- [Handler Classification Rules](HANDLER_CLASSIFICATION_RULES.md) (OMN-4004) — authoritative rubric
- `src/omnibase_infra/mixins/mixin_async_circuit_breaker.py` — implementation
- `src/omnibase_infra/mixins/mixin_retry_execution.py` — implementation
- OMN-4005 — 3.1 POC (postgres mixins classified as KEEP AS MIXIN — different structural reason)
- OMN-4007 — closed without action per this classification
- OMN-4014 — parent epic
