# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""SLO profiling and load tests for local LLM inference endpoints.

This test suite measures per-endpoint latency characteristics including:
- P50/P95/P99 latency at baseline (1 concurrent request)
- Cold start penalty (first request vs warm steady-state)
- Concurrency degradation at 1, 2, 5, 10 concurrent requests
- Maximum concurrency before SLO violation

SLO Targets (P95 at 1 concurrent request):
    - Qwen2.5-14B  (routing):            P95 < 400ms (transport + minimal inference)
    - Qwen2.5-Coder-14B (analysis):      P95 < 200ms
    - Qwen2.5-72B  (summarization/docs):  P95 < 800ms
    - GTE-Qwen2    (embedding):           P95 < 100ms
    - Qwen2-VL     (vision):              P95 < 600ms

    NOTE: These SLO targets measure transport + minimal inference (max_tokens=1).
    Real workloads with longer generation will have higher latencies proportional
    to output token count.

CI Behavior:
    ALL tests in this module are skipped in CI environments because they require
    real LLM inference servers running on the local network. These are local-only
    profiling tests.

Endpoint Configuration:
    Endpoints are read from environment variables with defaults matching the
    standard OmniNode multi-server LLM architecture:

    - LLM_CODER_URL:    http://192.168.86.201:8000  (RTX 5090)
    - LLM_EMBEDDING_URL: http://192.168.86.201:8002 (RTX 4090)
    - LLM_QWEN_72B_URL: http://192.168.86.200:8100  (Mac Studio M2 Ultra)
    - LLM_VISION_URL:   http://192.168.86.200:8102   (Mac Studio M2 Ultra)
    - LLM_QWEN_14B_URL: http://192.168.86.100:8200   (Mac Mini M2 Pro)

Usage:
    Run all LLM SLO tests locally:
        uv run pytest tests/performance/test_llm_endpoint_slo.py -v -s

    Run only LLM-tagged tests:
        uv run pytest -m llm -v -s

    Run a specific endpoint class:
        uv run pytest tests/performance/test_llm_endpoint_slo.py::TestCoder14BSlo -v -s

Related:
    - OMN-2249: Local endpoint SLO profiling and load test
    - docs/slo/llm_endpoint_slo.md: SLO document with targets and backpressure strategy
    - src/omnibase_infra/mixins/mixin_llm_http_transport.py: Production HTTP transport
"""

from __future__ import annotations

import asyncio
import os
import time
import warnings
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from statistics import mean, median, quantiles, stdev
from typing import Any

import httpx
import pytest
import pytest_asyncio

from omnibase_infra.testing import is_ci_environment
from omnibase_infra.utils.util_error_sanitization import (
    sanitize_error_message,
    sanitize_error_string,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IS_CI = is_ci_environment()

# Skip ALL tests in this module when running in CI (no real LLM servers).
pytestmark = [
    pytest.mark.performance,
    pytest.mark.llm,
    pytest.mark.skipif(
        IS_CI, reason="Requires real LLM inference endpoints on local network"
    ),
]

# Endpoint URLs from env with OmniNode defaults
CODER_14B_URL = os.getenv("LLM_CODER_URL", "http://192.168.86.201:8000")
EMBEDDING_URL = os.getenv("LLM_EMBEDDING_URL", "http://192.168.86.201:8002")
QWEN_72B_URL = os.getenv("LLM_QWEN_72B_URL", "http://192.168.86.200:8100")
VISION_URL = os.getenv("LLM_VISION_URL", "http://192.168.86.200:8102")
QWEN_14B_URL = os.getenv("LLM_QWEN_14B_URL", "http://192.168.86.100:8200")

# Timeouts: chat endpoints can be slow on first load; embeddings are fast.
CHAT_TIMEOUT = 60.0
EMBEDDING_TIMEOUT = 15.0

# Number of warm-up iterations (not measured) before profiling.
WARMUP_ITERATIONS = 3

# Number of measured iterations for single-concurrency profiling.
PROFILE_ITERATIONS = 20

# Concurrency levels to sweep.
CONCURRENCY_LEVELS = [1, 2, 5, 10]

# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

# Minimal chat payload: single-token generation to measure transport + overhead.
# NOTE: "model" is intentionally omitted here; each test class injects its own
# MODEL class attribute via ``_payload_with_model()`` so that the correct
# vLLM-registered model name is sent to each endpoint.
MINIMAL_CHAT_PAYLOAD: dict[str, Any] = {
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 1,
    "temperature": 0.0,
}

# Minimal embedding payload: single short string.
MINIMAL_EMBEDDING_PAYLOAD: dict[str, Any] = {
    "input": "hello",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyProfile:
    """Aggregated latency statistics for a set of measurements."""

    latencies: tuple[float, ...] = field(default_factory=tuple)

    @property
    def count(self) -> int:
        """Return the number of latency samples."""
        return len(self.latencies)

    @property
    def p50_ms(self) -> float:
        """Return the 50th percentile (median) latency in milliseconds."""
        if self.count < 2:
            return self.latencies[0] * 1000.0 if self.count == 1 else 0.0
        return quantiles(self.latencies, n=100)[49] * 1000.0

    @property
    def p95_ms(self) -> float:
        """Return the 95th percentile latency in milliseconds."""
        if self.count < 2:
            return self.latencies[0] * 1000.0 if self.count == 1 else 0.0
        return quantiles(self.latencies, n=100)[94] * 1000.0

    @property
    def p99_ms(self) -> float:
        """Return the 99th percentile latency in milliseconds."""
        if self.count < 2:
            return self.latencies[0] * 1000.0 if self.count == 1 else 0.0
        return quantiles(self.latencies, n=100)[98] * 1000.0

    @property
    def mean_ms(self) -> float:
        """Return the arithmetic mean latency in milliseconds."""
        return mean(self.latencies) * 1000.0 if self.latencies else 0.0

    @property
    def median_ms(self) -> float:
        """Return the median latency in milliseconds."""
        return median(self.latencies) * 1000.0 if self.latencies else 0.0

    @property
    def stdev_ms(self) -> float:
        """Return the sample standard deviation in milliseconds.

        Returns 0.0 when fewer than 2 samples are available.
        """
        if self.count < 2:
            return 0.0
        return stdev(self.latencies) * 1000.0

    @property
    def min_ms(self) -> float:
        """Return the minimum latency in milliseconds."""
        return min(self.latencies) * 1000.0 if self.latencies else 0.0

    @property
    def max_ms(self) -> float:
        """Return the maximum latency in milliseconds."""
        return max(self.latencies) * 1000.0 if self.latencies else 0.0


def _print_profile(label: str, profile: LatencyProfile) -> None:
    """Print a formatted latency profile table row."""
    print(f"\n  {label} ({profile.count} samples):")
    print(f"    Mean:   {profile.mean_ms:8.1f} ms")
    print(f"    Median: {profile.median_ms:8.1f} ms")
    print(f"    Stdev:  {profile.stdev_ms:8.1f} ms")
    print(f"    Min:    {profile.min_ms:8.1f} ms")
    print(f"    Max:    {profile.max_ms:8.1f} ms")
    print(f"    P50:    {profile.p50_ms:8.1f} ms")
    print(f"    P95:    {profile.p95_ms:8.1f} ms")
    print(f"    P99:    {profile.p99_ms:8.1f} ms")


def _print_concurrency_table(
    endpoint_name: str,
    results: dict[int, LatencyProfile],
) -> None:
    """Print a concurrency sweep results table."""
    print(f"\n{'=' * 72}")
    print(f"  Concurrency Sweep: {endpoint_name}")
    print(f"{'=' * 72}")
    print(
        f"  {'Concurrency':>12} {'Mean ms':>10} {'P50 ms':>10} {'P95 ms':>10} {'P99 ms':>10}"
    )
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for level in sorted(results.keys()):
        p = results[level]
        print(
            f"  {level:>12} {p.mean_ms:>10.1f} {p.p50_ms:>10.1f} {p.p95_ms:>10.1f} {p.p99_ms:>10.1f}"
        )


async def _check_endpoint_reachable(url: str, timeout: float = 5.0) -> bool:
    """Quick health check: attempt a GET on /health or /v1/models.

    Returns True if the endpoint responds with a non-error status (< 400),
    False on connection error or client/server error responses. This threshold
    aligns with ``_measure_single_request`` which raises on status >= 400,
    preventing a 4xx endpoint from passing reachability but failing profiling.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        for path in ["/health", "/v1/models"]:
            try:
                resp = await client.get(f"{url}{path}")
                if resp.status_code < 400:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                continue
    return False


async def _measure_single_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> float:
    """Send one POST request and return elapsed time in seconds.

    Raises on HTTP or connection errors so the caller can decide how to handle.
    """
    start = time.perf_counter()
    response = await client.post(url, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    # Accept any 2xx response; raise on server errors to surface issues.
    if response.status_code >= 400:
        sanitized = sanitize_error_string(response.text[:200]) if response.text else ""
        msg = f"HTTP {response.status_code} from {url}: {sanitized}"
        raise httpx.HTTPStatusError(msg, request=response.request, response=response)
    return elapsed


async def _warmup(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    iterations: int = WARMUP_ITERATIONS,
) -> None:
    """Send warm-up requests (not measured) to prime model caches and JIT."""
    for _ in range(iterations):
        try:
            await _measure_single_request(client, url, payload, timeout)
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException):
            pass  # Best-effort warm-up; failures are acceptable.


async def _profile_sequential(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    iterations: int = PROFILE_ITERATIONS,
) -> LatencyProfile:
    """Sequentially send requests and collect latency samples."""
    latencies: list[float] = []
    for _ in range(iterations):
        elapsed = await _measure_single_request(client, url, payload, timeout)
        latencies.append(elapsed)
    return LatencyProfile(latencies=tuple(latencies))


async def _profile_concurrent(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    concurrency: int,
    requests_per_worker: int = 5,
) -> LatencyProfile:
    """Send concurrent requests across `concurrency` workers.

    Each worker sends `requests_per_worker` sequential requests. All workers
    start at the same time via asyncio.gather.
    """

    async def _worker() -> list[float]:
        """Send sequential requests for one worker and return latencies.

        Individual request failures are caught and logged so that a single
        transient error does not discard all successful latencies from this
        worker.
        """
        worker_latencies: list[float] = []
        worker_failures = 0
        for i in range(requests_per_worker):
            try:
                elapsed = await _measure_single_request(client, url, payload, timeout)
                worker_latencies.append(elapsed)
            except (
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.TimeoutException,
                OSError,
            ) as exc:
                worker_failures += 1
                sanitized = sanitize_error_message(exc)
                warnings.warn(
                    f"Worker request {i + 1}/{requests_per_worker} failed "
                    f"(total failures: {worker_failures}): {sanitized}",
                    stacklevel=2,
                )
        return worker_latencies

    results = await asyncio.gather(
        *[_worker() for _ in range(concurrency)], return_exceptions=True
    )
    all_latencies: list[float] = []
    for worker_result in results:
        if isinstance(worker_result, BaseException):
            # Unexpected errors (not caught per-request); skip entire worker.
            continue
        all_latencies.extend(worker_result)
    return LatencyProfile(latencies=tuple(all_latencies))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Shared httpx.AsyncClient for all tests in a class."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(CHAT_TIMEOUT),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Test Classes: One per endpoint
# ---------------------------------------------------------------------------


class BaseLLMEndpointSloTest:
    """Parameterized base class for per-endpoint SLO profiling tests.

    Subclasses MUST define the following class-level attributes:

        ENDPOINT_URL:  str  - Full URL including path (e.g. ".../v1/chat/completions")
        ENDPOINT_NAME: str  - Human-readable model name for reports
        BASE_URL:      str  - Base URL for reachability checks (no path)
        SLO_P95_MS:    float - P95 latency target in milliseconds
        PAYLOAD:       dict  - Request payload template (without "model" key)
        TIMEOUT:       float - Per-request timeout in seconds
        MODEL:         str  - vLLM model name served by this endpoint

    This class is NOT collected by pytest because it lacks the ``Test`` prefix.
    """

    ENDPOINT_URL: str
    ENDPOINT_NAME: str
    BASE_URL: str
    SLO_P95_MS: float
    PAYLOAD: dict[str, Any]
    TIMEOUT: float
    MODEL: str

    @classmethod
    def _payload_with_model(cls) -> dict[str, Any]:
        """Return a copy of PAYLOAD with the class MODEL injected."""
        return {"model": cls.MODEL, **cls.PAYLOAD}

    # -- tests ----------------------------------------------------------------

    async def test_endpoint_reachable(self) -> None:
        """Verify the endpoint is reachable before profiling."""
        reachable = await _check_endpoint_reachable(self.BASE_URL)
        if not reachable:
            pytest.skip(f"{self.ENDPOINT_NAME} not reachable at {self.BASE_URL}")
        assert reachable, f"Endpoint {self.BASE_URL} was unexpectedly unreachable"

    async def test_cold_start_penalty(self, http_client: httpx.AsyncClient) -> None:
        """Measure cold start latency vs warm steady-state."""
        reachable = await _check_endpoint_reachable(self.BASE_URL)
        if not reachable:
            pytest.skip(f"{self.ENDPOINT_NAME} not reachable at {self.BASE_URL}")

        payload = self._payload_with_model()

        # Cold: first request on a fresh client (latency measured inside helper)
        async with httpx.AsyncClient(timeout=self.TIMEOUT) as cold_client:
            try:
                cold_latency = await _measure_single_request(
                    cold_client, self.ENDPOINT_URL, payload, self.TIMEOUT
                )
            except (
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.TimeoutException,
            ) as exc:
                pytest.skip(f"Cold request failed: {sanitize_error_message(exc)}")

        # Warm: after warm-up
        await _warmup(http_client, self.ENDPOINT_URL, payload, self.TIMEOUT)
        warm_profile = await _profile_sequential(
            http_client,
            self.ENDPOINT_URL,
            payload,
            self.TIMEOUT,
            iterations=10,
        )

        cold_latency_ms = cold_latency * 1000.0
        ratio = (
            cold_latency_ms / warm_profile.mean_ms if warm_profile.mean_ms > 0 else 1.0
        )

        print(f"\n{self.ENDPOINT_NAME} Cold Start Analysis:")
        print(f"  Cold:       {cold_latency_ms:.1f} ms")
        print(f"  Warm mean:  {warm_profile.mean_ms:.1f} ms")
        print(f"  Ratio:      {ratio:.1f}x")

        # Soft assertion: warn if cold start exceeds 5x the SLO P95 target.
        cold_threshold_ms = self.SLO_P95_MS * 5.0
        if cold_latency_ms > cold_threshold_ms:
            warnings.warn(
                f"{self.ENDPOINT_NAME} cold start latency {cold_latency_ms:.1f}ms "
                f"exceeds 5x SLO P95 target ({cold_threshold_ms:.0f}ms). "
                f"Cold/warm ratio: {ratio:.1f}x",
                stacklevel=1,
            )

    async def test_baseline_latency(self, http_client: httpx.AsyncClient) -> None:
        """Measure P50/P95/P99 latency at 1 concurrent request.

        Asserts P95 < SLO_P95_MS (transport + minimal inference).
        """
        reachable = await _check_endpoint_reachable(self.BASE_URL)
        if not reachable:
            pytest.skip(f"{self.ENDPOINT_NAME} not reachable at {self.BASE_URL}")

        payload = self._payload_with_model()
        await _warmup(http_client, self.ENDPOINT_URL, payload, self.TIMEOUT)
        profile = await _profile_sequential(
            http_client, self.ENDPOINT_URL, payload, self.TIMEOUT
        )

        print(f"\n{self.ENDPOINT_NAME} Baseline Latency (1 concurrent):")
        _print_profile("Sequential", profile)

        assert profile.p95_ms < self.SLO_P95_MS, (
            f"{self.ENDPOINT_NAME} P95 {profile.p95_ms:.1f}ms exceeds SLO target {self.SLO_P95_MS}ms"
        )

    async def test_concurrency_degradation(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Measure latency at 1, 2, 5, 10 concurrent requests."""
        reachable = await _check_endpoint_reachable(self.BASE_URL)
        if not reachable:
            pytest.skip(f"{self.ENDPOINT_NAME} not reachable at {self.BASE_URL}")

        payload = self._payload_with_model()
        await _warmup(http_client, self.ENDPOINT_URL, payload, self.TIMEOUT)

        results: dict[int, LatencyProfile] = {}
        for level in CONCURRENCY_LEVELS:
            profile = await _profile_concurrent(
                http_client,
                self.ENDPOINT_URL,
                payload,
                self.TIMEOUT,
                concurrency=level,
                requests_per_worker=5,
            )
            if not profile.latencies:
                pytest.skip(
                    f"{self.ENDPOINT_NAME}: all requests failed at concurrency={level}"
                )
            results[level] = profile

        _print_concurrency_table(self.ENDPOINT_NAME, results)

        # At 1 concurrent, P95 must meet SLO
        if 1 in results:
            assert results[1].p95_ms < self.SLO_P95_MS, (
                f"{self.ENDPOINT_NAME} P95 at concurrency=1: {results[1].p95_ms:.1f}ms "
                f"exceeds SLO target {self.SLO_P95_MS}ms"
            )


class TestCoder14BSlo(BaseLLMEndpointSloTest):
    """SLO profiling for Qwen2.5-Coder-14B (RTX 5090, code generation)."""

    ENDPOINT_URL = f"{CODER_14B_URL}/v1/chat/completions"
    ENDPOINT_NAME = "Qwen2.5-Coder-14B"
    BASE_URL = CODER_14B_URL
    SLO_P95_MS = 200.0
    PAYLOAD = MINIMAL_CHAT_PAYLOAD
    TIMEOUT = CHAT_TIMEOUT
    MODEL = os.getenv("LLM_CODER_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct")


class TestEmbeddingSlo(BaseLLMEndpointSloTest):
    """SLO profiling for GTE-Qwen2-1.5B (RTX 4090, embeddings)."""

    ENDPOINT_URL = f"{EMBEDDING_URL}/v1/embeddings"
    ENDPOINT_NAME = "GTE-Qwen2-1.5B"
    BASE_URL = EMBEDDING_URL
    SLO_P95_MS = 100.0
    PAYLOAD = MINIMAL_EMBEDDING_PAYLOAD
    TIMEOUT = EMBEDDING_TIMEOUT
    MODEL = os.getenv("LLM_EMBEDDING_MODEL", "Alibaba-NLP/gte-Qwen2-1.5B-instruct")


class TestQwen72BSlo(BaseLLMEndpointSloTest):
    """SLO profiling for Qwen2.5-72B (Mac Studio M2 Ultra, documentation/analysis)."""

    ENDPOINT_URL = f"{QWEN_72B_URL}/v1/chat/completions"
    ENDPOINT_NAME = "Qwen2.5-72B"
    BASE_URL = QWEN_72B_URL
    SLO_P95_MS = 800.0
    PAYLOAD = MINIMAL_CHAT_PAYLOAD
    TIMEOUT = CHAT_TIMEOUT
    MODEL = os.getenv("LLM_QWEN_72B_MODEL", "Qwen/Qwen2.5-72B-Instruct")


class TestVisionSlo(BaseLLMEndpointSloTest):
    """SLO profiling for Qwen2-VL (Mac Studio M2 Ultra, vision/multimodal)."""

    ENDPOINT_URL = f"{VISION_URL}/v1/chat/completions"
    ENDPOINT_NAME = "Qwen2-VL"
    BASE_URL = VISION_URL
    SLO_P95_MS = 600.0
    PAYLOAD = MINIMAL_CHAT_PAYLOAD
    TIMEOUT = CHAT_TIMEOUT
    MODEL = os.getenv("LLM_VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")


class TestQwen14BSlo(BaseLLMEndpointSloTest):
    """SLO profiling for Qwen2.5-14B (Mac Mini M2 Pro, routing/general purpose)."""

    ENDPOINT_URL = f"{QWEN_14B_URL}/v1/chat/completions"
    ENDPOINT_NAME = "Qwen2.5-14B"
    BASE_URL = QWEN_14B_URL
    SLO_P95_MS = 400.0
    PAYLOAD = MINIMAL_CHAT_PAYLOAD
    TIMEOUT = CHAT_TIMEOUT
    MODEL = os.getenv("LLM_QWEN_14B_MODEL", "Qwen/Qwen2.5-14B-Instruct")


# ---------------------------------------------------------------------------
# Cross-Endpoint Summary Test
# ---------------------------------------------------------------------------


class TestCrossEndpointSummary:
    """Run a quick baseline across all reachable endpoints and print a summary table."""

    async def test_all_endpoints_summary(self, http_client: httpx.AsyncClient) -> None:
        """Collect baseline P50/P95/P99 from all reachable endpoints.

        This test does NOT assert SLO compliance. It is purely informational
        and prints a summary table useful for baselining.
        """
        endpoints: list[tuple[str, str, str, dict[str, Any], float]] = [
            (
                "Qwen2.5-Coder-14B",
                CODER_14B_URL,
                f"{CODER_14B_URL}/v1/chat/completions",
                {"model": TestCoder14BSlo.MODEL, **MINIMAL_CHAT_PAYLOAD},
                CHAT_TIMEOUT,
            ),
            (
                "GTE-Qwen2-1.5B",
                EMBEDDING_URL,
                f"{EMBEDDING_URL}/v1/embeddings",
                {"model": TestEmbeddingSlo.MODEL, **MINIMAL_EMBEDDING_PAYLOAD},
                EMBEDDING_TIMEOUT,
            ),
            (
                "Qwen2.5-72B",
                QWEN_72B_URL,
                f"{QWEN_72B_URL}/v1/chat/completions",
                {"model": TestQwen72BSlo.MODEL, **MINIMAL_CHAT_PAYLOAD},
                CHAT_TIMEOUT,
            ),
            (
                "Qwen2-VL",
                VISION_URL,
                f"{VISION_URL}/v1/chat/completions",
                {"model": TestVisionSlo.MODEL, **MINIMAL_CHAT_PAYLOAD},
                CHAT_TIMEOUT,
            ),
            (
                "Qwen2.5-14B",
                QWEN_14B_URL,
                f"{QWEN_14B_URL}/v1/chat/completions",
                {"model": TestQwen14BSlo.MODEL, **MINIMAL_CHAT_PAYLOAD},
                CHAT_TIMEOUT,
            ),
        ]

        print(f"\n{'=' * 80}")
        print("  LLM Endpoint SLO Baseline Summary")
        print(f"{'=' * 80}")
        print(
            f"  {'Endpoint':<22} {'Status':<12} {'P50 ms':>10} {'P95 ms':>10} {'P99 ms':>10} {'Mean ms':>10}"
        )
        print(f"  {'-' * 22} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

        for name, base_url, full_url, payload, timeout in endpoints:
            reachable = await _check_endpoint_reachable(base_url)
            if not reachable:
                print(
                    f"  {name:<22} {'UNREACHABLE':<12} {'--':>10} {'--':>10} {'--':>10} {'--':>10}"
                )
                continue

            try:
                await _warmup(http_client, full_url, payload, timeout, iterations=2)
                profile = await _profile_sequential(
                    http_client, full_url, payload, timeout, iterations=10
                )
                print(
                    f"  {name:<22} {'OK':<12} "
                    f"{profile.p50_ms:>10.1f} {profile.p95_ms:>10.1f} "
                    f"{profile.p99_ms:>10.1f} {profile.mean_ms:>10.1f}"
                )
            except (
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.TimeoutException,
                OSError,
            ) as exc:
                print(
                    f"  {name:<22} {'ERROR':<12} {'--':>10} {'--':>10} {'--':>10} {sanitize_error_string(str(exc))[:30]}"
                )

        print(f"{'=' * 80}")
