> **Navigation**: [Home](../index.md) > [Architecture](README.md) > Handler Classification 3.3 — LLM HTTP Transport Mixin

# Handler Classification 3.3 — MixinLlmHttpTransport

**Ticket**: OMN-4008
**Epic**: OMN-4014 — Epic 3: Mixin/Service -> Handler Refactoring
**Rubric**: [Handler Classification Rules](HANDLER_CLASSIFICATION_RULES.md) (OMN-4004)
**Status**: Final — ABORT CONDITION TRIGGERED (defer to protocol-first approach)
**Last Updated**: 2026-03-08

---

## Purpose

This document provides the formal rubric-based classification of `MixinLlmHttpTransport`
against the 3.0 handler rubric, and records the abort condition that prevents full
execution in this ticket.

---

## Callers Surveyed

`MixinLlmHttpTransport` is used by 5 production classes:

| Caller | Pattern | Location |
|--------|---------|---------|
| `HandlerLlmOllama` | Inherits mixin, calls `_execute_llm_http_call` in `handle()` | `nodes/node_llm_inference_effect/handlers/` |
| `HandlerEmbeddingOllama` | Inherits mixin, calls `_execute_llm_http_call` | `nodes/node_llm_embedding_effect/handlers/` |
| `HandlerEmbeddingOpenaiCompatible` | Inherits mixin, calls `_execute_llm_http_call` | `nodes/node_llm_embedding_effect/handlers/` |
| `TransportHolderLlmHttp` | Inherits mixin, wraps `_execute_llm_http_call` + `_get_http_client` | `adapters/llm/` |
| `TransportInstance` (inline class) | Inherits mixin as inline factory pattern | `nodes/node_llm_inference_effect/registry/` |

All 5 callers use `_execute_llm_http_call` as the primary dispatch point and
`_close_http_client()` for lifecycle cleanup.

---

## Rubric Scoring

### Criterion 1: I/O Ownership

**Question**: Does this code own a direct connection to an external system?

**Finding**: YES — `MixinLlmHttpTransport` owns an `httpx.AsyncClient` instance.

Evidence:
- `_http_client: httpx.AsyncClient | None` — stores a live HTTP client
- `_owns_http_client: bool` — tracks ownership for lifecycle management
- Lazy client creation via `_get_http_client()` using `asyncio.Lock`
- `_close_http_client()` calls `self._http_client.aclose()` for resource cleanup

Score: **YES**

---

### Criterion 2: Lifecycle Manageability

**Question**: Does this code need `initialize()` / `shutdown()` lifecycle the container should manage?

**Finding**: YES — the mixin manages an HTTP client with explicit teardown.

Evidence:
- All 5 callers implement a `close()` method that delegates to `_close_http_client()`
- The client is created lazily and released explicitly — this IS a lifecycle
- Container should manage this rather than ad-hoc per-caller cleanup

Score: **YES**

---

### Criterion 3: Dispatch Entry Point Clarity

**Question**: Is there a single, clear dispatch entry point?

**Finding**: YES — `_execute_llm_http_call()` is the single dispatch point used by all callers.

Evidence:
- All 5 callers call exactly `_execute_llm_http_call(url, payload, correlation_id, ...)`
- The HMAC signing, CIDR validation, circuit breaker, retry logic are all encapsulated here
- No caller bypasses this to call lower-level methods directly (except `TransportHolderLlmHttp`
  which calls `_get_http_client()` for a health-check path — an outlier)

Score: **YES** (one well-defined primary entry point)

---

### Criterion 4: Testability Without Subclassing

**Question**: Can this code be tested by injecting a mock/stub without subclassing?

**Finding**: YES — injection would enable direct mock substitution.

Evidence:
- Current tests must create a concrete subclass implementing abstract methods from
  `MixinRetryExecution` to test LLM transport behavior
- An injected `HandlerLlmHttpTransport` could be mocked as a simple interface:
  `mock_transport.execute(url, payload, correlation_id) -> dict`
- Tests for `HandlerLlmOllama` would no longer need to exercise retry/circuit logic

Score: **YES**

---

### Criterion 5: Cross-Layer Leakage Risk

**Question**: Does inheriting this mixin grant capabilities that violate ONEX layer boundaries?

**Finding**: YES — inheriting this mixin grants LLM HTTP I/O capability to any class.

Evidence:
- Any node or orchestrator inheriting `MixinLlmHttpTransport` gains the ability to
  make authenticated, signed HTTP calls to LLM endpoints
- This is a significant capability — CIDR allowlist enforcement, HMAC signing, circuit breaker
  are all attached to the class without explicit injection
- Injected handler makes this capability explicit and auditable

Score: **YES**

---

## Score Summary

| Criterion | MixinLlmHttpTransport |
|-----------|----------------------|
| C1: I/O Ownership | YES |
| C2: Lifecycle Manageability | YES |
| C3: Dispatch Entry Point Clarity | YES |
| C4: Testability Without Subclassing | YES |
| C5: Cross-Layer Leakage Risk | YES |
| **YES count** | **5** |
| **Recommendation** | **CONVERT to handler** |

Score 5/5 → per the decision matrix: **CONVERT to handler** — refactoring is clearly justified.

---

## Abort Condition: Protocol Interface Mismatch

Per [HANDLER_CLASSIFICATION_RULES.md](HANDLER_CLASSIFICATION_RULES.md) Section 7, abort if:

> **Abort Condition 1**: "if dispatch integration is more complex than expected — if wiring
> the injected handler requires more scaffolding than the mixin it replaces, the net
> architectural benefit is negative"

**This abort condition is triggered.**

### Analysis

The canonical ONEX handler interface (`ProtocolHandler`) defines:

```python
async def execute(
    self,
    request: ModelProtocolRequest,
    operation_config: ModelOperationConfig,
) -> ModelProtocolResponse:
```

`_execute_llm_http_call` takes domain-specific parameters:

```python
async def _execute_llm_http_call(
    self,
    url: str,
    payload: dict[str, JsonType],
    correlation_id: UUID,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
) -> dict[str, JsonType]:
```

To convert `MixinLlmHttpTransport` to a full `ProtocolHandler`, one of two paths is required:

**Path A: Wrap domain types in ModelProtocolRequest**
```python
# In each caller:
request = ModelProtocolRequest(
    payload={"url": url, "payload": payload, "correlation_id": str(correlation_id), ...}
)
response = await self._transport.execute(request, operation_config)
result = response.data  # unpack dict
```
This is pure scaffolding — the domain structure is discarded and re-parsed at the handler
boundary. Net architectural clarity: **negative** (more code, no new invariants).

**Path B: Define a domain-specific protocol**
```python
class ProtocolLlmHttpTransportHandler(Protocol):
    async def execute_llm_http_call(
        self,
        url: str,
        payload: dict[str, JsonType],
        correlation_id: UUID,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ) -> dict[str, JsonType]: ...
```
This preserves the domain interface and enables injection. This is the correct path.

**Why Path B requires a separate ticket**: Defining `ProtocolLlmHttpTransportHandler` in
`omnibase_spi` requires:
1. A new protocol file in `omnibase_spi/src/omnibase_spi/protocols/handlers/`
2. Cross-repo alignment (omnibase_spi is a separate repo)
3. Type validation across both repos

This is a proper multi-repo change, not scoped to OMN-4008 (omnibase_infra only).

---

## Decision

**MixinLlmHttpTransport**: Rubric score 5/5 — refactoring IS architecturally justified, but
the abort condition prevents execution in this ticket.

**Root cause**: The generic `ProtocolHandler.execute()` interface does not match the domain-specific
LLM transport signature. Full refactoring requires first defining `ProtocolLlmHttpTransportHandler`
in `omnibase_spi`.

**Action**:
1. This ticket (OMN-4008) is **DEFERRED** — classification complete, abort condition documented
2. A follow-up ticket should be created: "Define ProtocolLlmHttpTransportHandler in omnibase_spi + refactor MixinLlmHttpTransport to handler"

---

## Ambiguity Signals Already Satisfiable

For reference, when the follow-up ticket executes Path B, the following ambiguity signals
from Section 5 of the rubric will be demonstrable:

| Signal | Evidence |
|--------|---------|
| **S1: Hidden I/O eliminated** | Callers no longer inherit `_http_client`; only the injected handler's `execute_llm_http_call()` is accessible |
| **S2: Dispatch ownership clarified** | `execute_llm_http_call()` on the protocol replaces `_execute_llm_http_call()` inherited from mixin |
| **S3: Lifecycle enforced** | `initialize()` / `shutdown()` on the handler replace the ad-hoc `close()` delegation pattern |
| **S4: Testability improved** | Tests inject a mock `ProtocolLlmHttpTransportHandler`; no mixin subclassing needed |
| **S5: Layer boundary respected** | LLM HTTP capability is explicit in the constructor signature, not inherited silently |

At least 2 of 5 signals (S1, S2, S4) are strongly demonstrable → net-positive refactoring
when the protocol is defined.

---

## References

- [Handler Classification Rules](HANDLER_CLASSIFICATION_RULES.md) (OMN-4004) — authoritative rubric
- `src/omnibase_infra/mixins/mixin_llm_http_transport.py` — implementation (1059 lines)
- `omnibase_spi/src/omnibase_spi/protocols/handlers/protocol_handler.py` — ProtocolHandler interface
- OMN-4014 — parent epic
