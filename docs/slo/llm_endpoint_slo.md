# LLM Endpoint SLO Definitions

**Ticket**: OMN-2249
**Last Updated**: 2026-02-16
**Status**: Initial baseline (targets pending first profiling run)

---

## Overview

This document defines Service Level Objectives (SLOs) for the five local LLM
inference endpoints used by the OmniNode platform. These SLOs cover transport
latency (HTTP round-trip + minimal 1-token generation), concurrency degradation,
and backpressure strategy.

All measurements use a minimal payload (`max_tokens=1`) to isolate transport and
inference startup overhead from variable-length generation. Real workloads will
have higher latencies proportional to output token count.

---

## Endpoint Inventory

| Endpoint | Model | Hardware | URL | Primary Use |
|----------|-------|----------|-----|-------------|
| **Coder-14B** | Qwen2.5-Coder-14B | RTX 5090 (32 GB) | `http://192.168.86.201:8000` | Code generation, analysis, enforcement |
| **Embedding** | GTE-Qwen2-1.5B | RTX 4090 (24 GB) | `http://192.168.86.201:8002` | RAG embeddings, semantic search |
| **72B** | Qwen2.5-72B | Mac Studio M2 Ultra | `http://192.168.86.200:8100` | Documentation, summarization, complex analysis |
| **Vision** | Qwen2-VL | Mac Studio M2 Ultra | `http://192.168.86.200:8102` | Vision, multimodal, screenshot analysis |
| **14B** | Qwen2.5-14B | Mac Mini M2 Pro | `http://192.168.86.100:8200` | Agent routing, general purpose |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_CODER_URL` | `http://192.168.86.201:8000` | Coder endpoint base URL |
| `LLM_EMBEDDING_URL` | `http://192.168.86.201:8002` | Embedding endpoint base URL |
| `LLM_QWEN_72B_URL` | `http://192.168.86.200:8100` | 72B endpoint base URL |
| `LLM_VISION_URL` | `http://192.168.86.200:8102` | Vision endpoint base URL |
| `LLM_QWEN_14B_URL` | `http://192.168.86.100:8200` | 14B routing endpoint base URL |

---

## SLO Targets

### Baseline Latency (1 Concurrent Request, max_tokens=1)

| Endpoint | P95 Target | P99 Budget | Rationale |
|----------|-----------|------------|-----------|
| **Qwen2.5-14B** (routing) | < 400 ms | < 800 ms | Transport + minimal inference (max_tokens=1) on M2 Pro. 14B model on Apple Silicon has typical TTFT of 100-300ms; target includes headroom for thermal/load variance. |
| **Qwen2.5-Coder-14B** (analysis) | < 200 ms | < 400 ms | Used for code analysis and enforcement. RTX 5090 GPU inference is fast; moderate latency acceptable since these are background tasks. |
| **Qwen2.5-72B** (summarization) | < 800 ms | < 1500 ms | 72B model on M2 Ultra unified memory. Higher latency expected; used for documentation and analysis where latency is less critical. |
| **GTE-Qwen2-1.5B** (embedding) | < 100 ms | < 200 ms | Small model on GPU. Embeddings must be fast for real-time RAG queries. |
| **Qwen2-VL** (vision) | < 600 ms | < 1200 ms | Vision model on M2 Ultra. Multimodal processing is inherently slower. Acceptable for async screenshot/diagram analysis. |

### Cold Start Budget

Cold start is the latency of the first request after server idle or model swap.

| Endpoint | Cold Start Budget | Notes |
|----------|------------------|-------|
| **Qwen2.5-14B** | < 2x warm | Always-on; minimal cold start expected |
| **Qwen2.5-Coder-14B** | < 3x warm | Dedicated GPU; model stays loaded |
| **Qwen2.5-72B** | < 5x warm | Large model; first inference may trigger memory mapping |
| **GTE-Qwen2-1.5B** | < 2x warm | Small model; fast to warm |
| **Qwen2-VL** | < 5x warm | Multimodal model; larger initialization |

---

## Concurrency Degradation Expectations

### Expected Behavior at Increasing Concurrency

These are expected ranges based on hardware characteristics. The profiling test
(`test_llm_endpoint_slo.py`) measures actual values.

#### GPU Endpoints (RTX 5090 / RTX 4090)

GPU inference servers (vLLM) batch concurrent requests efficiently using
continuous batching. Expected degradation is sub-linear up to the batch
saturation point.

| Concurrency | Expected P95 Multiplier | Notes |
|-------------|------------------------|-------|
| 1x (baseline) | 1.0x | Single request, no contention |
| 2x | 1.0-1.3x | GPU batching absorbs second request |
| 5x | 1.2-2.0x | Batch queue starts filling |
| 10x | 2.0-4.0x | Approaching VRAM bandwidth saturation |

#### Apple Silicon Endpoints (M2 Ultra / M2 Pro)

Apple Silicon uses unified memory with Metal acceleration. Concurrency
degradation is more linear because there is no hardware batching.

| Concurrency | Expected P95 Multiplier | Notes |
|-------------|------------------------|-------|
| 1x (baseline) | 1.0x | Single request, no contention |
| 2x | 1.5-2.0x | Serialized inference on single accelerator |
| 5x | 3.0-5.0x | Queue depth causes latency spike |
| 10x | 5.0-10.0x | Severe degradation; backpressure needed |

### Maximum Recommended Concurrency

| Endpoint | Max Concurrent Before SLO Violation | Reasoning |
|----------|-------------------------------------|-----------|
| **Qwen2.5-14B** | 2 | 400ms SLO on M2 Pro; limited throughput for 14B model |
| **Qwen2.5-Coder-14B** | 5 | RTX 5090 continuous batching handles moderate load |
| **Qwen2.5-72B** | 2 | Large model on M2 Ultra; serialized inference |
| **GTE-Qwen2-1.5B** | 10+ | Small model; GPU handles high concurrency for embeddings |
| **Qwen2-VL** | 2 | Shares M2 Ultra with 72B (when both active) |

---

## Backpressure Strategy

### Principle

When concurrency exceeds the recommended maximum, the platform must **shed load
gracefully** rather than allowing unbounded queue growth that degrades all
requests.

### Three-Layer Defense

```text
Layer 1: Circuit Breaker (per-endpoint)
    |
    v
Layer 2: Concurrency Limiter (asyncio.Semaphore)
    |
    v
Layer 3: Queue with Timeout (bounded asyncio.Queue)
```

#### Layer 1: Circuit Breaker

Already implemented in `MixinAsyncCircuitBreaker` and used by
`MixinLlmHttpTransport`.

- **Threshold**: 5 consecutive failures -> OPEN
- **Reset timeout**: 60 seconds -> HALF_OPEN
- **Behavior when OPEN**: Immediately raise `InfraUnavailableError` (fast fail)
- **Prevents**: Cascading failures when an endpoint is down

#### Layer 2: Concurrency Limiter (Recommended)

Add an `asyncio.Semaphore` per endpoint to cap in-flight requests:

```python
# Per-endpoint concurrency limits
CONCURRENCY_LIMITS = {
    "qwen-14b":     2,   # M2 Pro, limited throughput for 14B
    "coder-14b":    5,   # GPU batching handles moderate load
    "qwen-72b":     2,   # Large model, serialized inference
    "gte-qwen2":   10,   # Small model, high throughput
    "qwen2-vl":     2,   # Shared accelerator
}

semaphore = asyncio.Semaphore(CONCURRENCY_LIMITS[endpoint_name])

async with semaphore:
    result = await self._execute_llm_http_call(url, payload, correlation_id)
```

- **Behavior at limit**: Requests queue behind the semaphore
- **Prevents**: Overloading the inference server beyond its batching capacity

#### Layer 3: Queue with Timeout (Recommended for Production)

Wrap the semaphore acquisition with a timeout to prevent unbounded waiting:

```python
try:
    async with asyncio.timeout(5.0):  # 5s max wait
        async with semaphore:
            result = await self._execute_llm_http_call(...)
except TimeoutError:
    raise InfraRateLimitedError(
        f"Request queued too long for {endpoint_name}",
        context=ctx,
        retry_after_seconds=2.0,
    )
```

- **Behavior on timeout**: Return 429-equivalent error with retry-after hint
- **Prevents**: Clients waiting indefinitely during traffic spikes

### Degradation at Scale

| Load Level | Behavior | User Impact |
|------------|----------|-------------|
| **1x** (normal) | All requests served within SLO | None |
| **2x** (moderate) | Semaphore queues excess requests; P95 may increase 50-100% | Slight latency increase; within P99 budget |
| **5x** (spike) | Queue timeout starts rejecting requests; circuit breaker may trigger | Some requests receive 429 with retry-after; clients should retry with backoff |
| **10x** (overload) | Most requests rejected at queue layer; circuit breaker likely OPEN | Majority of requests fail fast; auto-recovery when load subsides |

### RTX 4090 Hot-Swap Consideration

The RTX 4090 is shared between GTE-Qwen2 (embeddings), Qwen2.5-7B (function
calling), and DeepSeek-V2-Lite. Only one model runs at a time.

**Impact on SLOs**:
- When switching models, the endpoint at the old port becomes unreachable
- Circuit breaker will open after 5 failed connection attempts
- After model swap completes (~30-60s), circuit breaker resets in HALF_OPEN
- First request after swap has elevated cold start latency

**Recommendation**: Coordinate model swaps during low-traffic periods. Emit a
`model.swap.initiated` event so upstream callers can pre-emptively pause
requests to the affected port.

---

## Profiling Test Reference

### Running the Baseline

```bash
# Full profiling suite (requires all endpoints running)
uv run pytest tests/performance/test_llm_endpoint_slo.py -v -s

# Single endpoint
uv run pytest tests/performance/test_llm_endpoint_slo.py::TestCoder14BSlo -v -s

# Summary table only (quick check)
uv run pytest tests/performance/test_llm_endpoint_slo.py::TestCrossEndpointSummary -v -s

# By marker
uv run pytest -m llm -v -s
```

### Test Structure

| Test Class | Endpoint | Tests |
|------------|----------|-------|
| `TestCoder14BSlo` | Qwen2.5-Coder-14B | reachable, cold start, baseline latency, concurrency sweep |
| `TestEmbeddingSlo` | GTE-Qwen2-1.5B | reachable, cold start, baseline latency, concurrency sweep |
| `TestQwen72BSlo` | Qwen2.5-72B | reachable, cold start, baseline latency, concurrency sweep |
| `TestVisionSlo` | Qwen2-VL | reachable, cold start, baseline latency, concurrency sweep |
| `TestQwen14BSlo` | Qwen2.5-14B | reachable, cold start, baseline latency, concurrency sweep |
| `TestCrossEndpointSummary` | All | Summary table across all reachable endpoints |

### Test Behavior

- All tests are skipped in CI (no real infrastructure)
- Unreachable endpoints are skipped with `pytest.skip()` (not failures)
- Warm-up iterations (3 by default) are not included in measurements
- Baseline profiling uses 20 sequential requests
- Concurrency sweep uses 5 requests per worker at each level
- SLO assertions only apply to baseline (concurrency=1) tests

---

## Updating SLO Targets

After collecting baseline measurements:

1. Run the full profiling suite: `uv run pytest tests/performance/test_llm_endpoint_slo.py -v -s`
2. Review the summary table output
3. If measured P95 exceeds the target:
   - Investigate whether the target is too aggressive for the hardware
   - Check for external factors (thermal throttling, background load)
   - Adjust the target in both this document and the test file
4. Store baseline numbers in this document for future regression comparison

### Baseline Measurements

> **TODO**: Fill in after first profiling run.

| Endpoint | P50 | P95 | P99 | Mean | Date |
|----------|-----|-----|-----|------|------|
| Qwen2.5-Coder-14B | -- | -- | -- | -- | -- |
| GTE-Qwen2-1.5B | -- | -- | -- | -- | -- |
| Qwen2.5-72B | -- | -- | -- | -- | -- |
| Qwen2-VL | -- | -- | -- | -- | -- |
| Qwen2.5-14B | -- | -- | -- | -- | -- |

---

## Related Documents

- [Circuit Breaker Implementation](../patterns/circuit_breaker_implementation.md)
- [Error Recovery Patterns](../patterns/error_recovery_patterns.md)
- [LLM HTTP Transport Mixin](../../src/omnibase_infra/mixins/mixin_llm_http_transport.py)
- [Multi-Server LLM Architecture](../../CLAUDE.md) (see `~/.claude/CLAUDE.md` LLM section)
