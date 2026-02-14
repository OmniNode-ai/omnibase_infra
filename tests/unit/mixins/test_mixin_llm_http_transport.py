# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for MixinLlmHttpTransport.

This test suite validates:
- HTTP status code to typed exception mapping (401, 403, 404, 429, 400, 422, 500-504)
- 429 does NOT increment circuit breaker failure count
- Retry-After header parsing (present, absent, unparseable, capped)
- Timeout capping: min(request, contract.max)
- Circuit breaker state transitions (open after threshold, half-open after reset)
- Retry count respected (0 retries = single attempt)
- Exponential backoff timing
- Connection refused -> InfraConnectionError
- Timeout -> InfraTimeoutError
- Non-JSON content-type -> InfraProtocolError
- JSON parse failure -> InfraProtocolError
- Successful response parsing

Test Pattern:
    Uses httpx.MockTransport to simulate HTTP responses. The mixin is tested
    through a thin test harness class (LlmTransportHarness) that extends
    MixinLlmHttpTransport.

Related Ticket: OMN-2114 Phase 14
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraProtocolError,
    InfraRateLimitedError,
    InfraRequestRejectedError,
    InfraTimeoutError,
    InfraUnavailableError,
    ProtocolConfigurationError,
)
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport

# ── Test Harness ─────────────────────────────────────────────────────────


class LlmTransportHarness(MixinLlmHttpTransport):
    """Thin test harness wrapping MixinLlmHttpTransport for unit testing.

    Initializes the mixin with sensible test defaults and exposes circuit
    breaker internals for assertions.
    """

    def __init__(
        self,
        target_name: str = "test-llm",
        max_timeout_seconds: float = 120.0,
        max_retry_after_seconds: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
        cb_threshold: int = 5,
    ) -> None:
        self._init_llm_http_transport(
            target_name=target_name,
            max_timeout_seconds=max_timeout_seconds,
            max_retry_after_seconds=max_retry_after_seconds,
            http_client=http_client,
        )
        # Re-initialize CB with custom threshold if needed
        if cb_threshold != 5:
            from omnibase_infra.enums import EnumInfraTransportType

            self._init_circuit_breaker(
                threshold=cb_threshold,
                reset_timeout=60.0,
                service_name=target_name,
                transport_type=EnumInfraTransportType.HTTP,
            )
            self._circuit_breaker_initialized = True


def _make_mock_client(
    handler: Any,
) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with a mock transport handler.

    Args:
        handler: A callable(request) -> httpx.Response.

    Returns:
        An httpx.AsyncClient using MockTransport.
    """
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


def _json_response(
    data: dict[str, Any],
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build a mock httpx.Response with JSON body."""
    import json

    all_headers = {"content-type": "application/json"}
    if headers:
        all_headers.update(headers)
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(data).encode(),
        headers=all_headers,
    )


def _text_response(
    text: str,
    status_code: int = 200,
    content_type: str = "text/plain",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build a mock httpx.Response with text body."""
    all_headers = {"content-type": content_type}
    if headers:
        all_headers.update(headers)
    return httpx.Response(
        status_code=status_code,
        content=text.encode(),
        headers=all_headers,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────

URL = "http://test-llm:8000/v1/chat/completions"
PAYLOAD: dict[str, Any] = {"messages": [{"role": "user", "content": "hello"}]}


@pytest.fixture
def correlation_id() -> UUID:
    """Provide a stable correlation ID for tests."""
    return uuid4()


# ── HTTP Status -> Exception Mapping ─────────────────────────────────────


class TestHttpStatusToExceptionMapping:
    """Validate that non-2xx HTTP status codes map to correct typed exceptions."""

    @pytest.mark.parametrize("status_code", [401, 403])
    async def test_401_403_raises_infra_authentication_error(
        self, status_code: int, correlation_id: UUID
    ) -> None:
        """401 and 403 responses must raise InfraAuthenticationError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "unauthorized"}, status_code=status_code)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraAuthenticationError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

    async def test_404_raises_protocol_configuration_error(
        self, correlation_id: UUID
    ) -> None:
        """404 response must raise ProtocolConfigurationError (assumed misconfiguration)."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "not found"}, status_code=404)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(ProtocolConfigurationError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

    @pytest.mark.parametrize("status_code", [400, 422])
    async def test_400_422_raises_infra_request_rejected_error(
        self, status_code: int, correlation_id: UUID
    ) -> None:
        """400 and 422 responses must raise InfraRequestRejectedError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "bad request"}, status_code=status_code)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraRequestRejectedError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    async def test_5xx_raises_infra_unavailable_error(
        self, status_code: int, correlation_id: UUID
    ) -> None:
        """500, 502, 503, 504 responses must raise InfraUnavailableError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "server error"}, status_code=status_code)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraUnavailableError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

    async def test_429_raises_infra_rate_limited_error(
        self, correlation_id: UUID
    ) -> None:
        """429 response must raise InfraRateLimitedError when retries exhausted."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "rate limited"}, status_code=429)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraRateLimitedError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

    async def test_unexpected_status_code_raises_unavailable_error(
        self, correlation_id: UUID
    ) -> None:
        """Unmapped HTTP status codes (e.g. 418) must fall back to InfraUnavailableError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "I'm a teapot"}, status_code=418)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraUnavailableError) as exc_info:
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

        assert "418" in str(exc_info.value)


# ── 429 Circuit Breaker Exclusion ────────────────────────────────────────


class TestRateLimitCircuitBreakerExclusion:
    """429 must NOT increment circuit breaker failure count."""

    async def test_429_does_not_increment_circuit_breaker_failure_count(
        self, correlation_id: UUID
    ) -> None:
        """429 responses should never count toward circuit breaker threshold.

        This is critical because rate limits are a normal flow-control mechanism,
        not a service health signal. Counting them would cause premature circuit
        opening under high load.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "rate limited"}, status_code=429)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client, cb_threshold=2)

        # Send multiple 429 responses - CB should NOT open
        for _ in range(5):
            with pytest.raises(InfraRateLimitedError):
                await harness._execute_llm_http_call(
                    url=URL,
                    payload=PAYLOAD,
                    correlation_id=correlation_id,
                    max_retries=0,
                )

        # Circuit breaker should still be closed (failures not incremented)
        assert harness._circuit_breaker_failures == 0
        assert harness._circuit_breaker_open is False

    async def test_429_classify_error_sets_record_circuit_failure_false(
        self, correlation_id: UUID
    ) -> None:
        """_classify_error for InfraRateLimitedError must set record_circuit_failure=False."""
        harness = LlmTransportHarness()
        error = InfraRateLimitedError("rate limited")
        classification = harness._classify_error(error, "test_op")

        assert classification.record_circuit_failure is False
        assert classification.should_retry is True


# ── Retry-After Header Parsing ───────────────────────────────────────────


class TestRetryAfterParsing:
    """Validate Retry-After header parsing for 429 responses."""

    async def test_429_with_retry_after_header_sets_retry_after_seconds(
        self, correlation_id: UUID
    ) -> None:
        """429 with Retry-After header must populate retry_after_seconds on error."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=429,
                content=b'{"error": "rate limited"}',
                headers={
                    "content-type": "application/json",
                    "retry-after": "5",
                },
            )

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraRateLimitedError) as exc_info:
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

        assert exc_info.value.retry_after_seconds == 5.0

    async def test_429_without_retry_after_header_uses_default_retry_after(
        self, correlation_id: UUID
    ) -> None:
        """429 without Retry-After header must set retry_after_seconds to default (1.0)."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=429,
                content=b'{"error": "rate limited"}',
                headers={"content-type": "application/json"},
            )

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraRateLimitedError) as exc_info:
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

        # Default fallback is 1.0 when header is absent
        assert exc_info.value.retry_after_seconds == 1.0

    async def test_retry_after_capped_to_max(self, correlation_id: UUID) -> None:
        """Retry-After values above max_retry_after_seconds must be clamped."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=429,
                content=b'{"error": "rate limited"}',
                headers={
                    "content-type": "application/json",
                    "retry-after": "999",
                },
            )

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client, max_retry_after_seconds=10.0)

        with pytest.raises(InfraRateLimitedError) as exc_info:
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

        assert exc_info.value.retry_after_seconds == 10.0

    async def test_retry_after_unparseable_falls_back_to_default(
        self, correlation_id: UUID
    ) -> None:
        """Unparseable Retry-After header must fall back to 1.0 default."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=429,
                content=b'{"error": "rate limited"}',
                headers={
                    "content-type": "application/json",
                    "retry-after": "Thu, 01 Dec 2025 16:00:00 GMT",
                },
            )

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraRateLimitedError) as exc_info:
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

        # HTTP-date format is not supported -> falls back to 1.0
        assert exc_info.value.retry_after_seconds == 1.0

    def test_parse_retry_after_unit(self) -> None:
        """Direct unit test for _parse_retry_after with various inputs."""
        harness = LlmTransportHarness(max_retry_after_seconds=30.0)

        # With valid header
        response = httpx.Response(
            status_code=429,
            headers={"retry-after": "15"},
        )
        assert harness._parse_retry_after(response) == 15.0

        # Without header
        response = httpx.Response(status_code=429, headers={})
        assert harness._parse_retry_after(response) == 1.0

        # With zero value
        response = httpx.Response(status_code=429, headers={"retry-after": "0"})
        assert harness._parse_retry_after(response) == 0.0

        # With float value
        response = httpx.Response(status_code=429, headers={"retry-after": "2.5"})
        assert harness._parse_retry_after(response) == 2.5

        # With negative value (clamped to 0.0)
        response = httpx.Response(status_code=429, headers={"retry-after": "-5"})
        assert harness._parse_retry_after(response) == 0.0


# ── Timeout Capping ──────────────────────────────────────────────────────


class TestTimeoutCapping:
    """Validate that per-call timeout is clamped to [0.1, max_timeout_seconds]."""

    async def test_timeout_capped_to_max(self, correlation_id: UUID) -> None:
        """timeout_seconds > max_timeout_seconds must be clamped down."""
        call_records: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            call_records.append(request.extensions.get("timeout", {}).get("pool", 0))
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client, max_timeout_seconds=10.0)

        # Request with timeout_seconds=60 but max is 10
        result = await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
            timeout_seconds=60.0,
        )
        assert result == {"result": "ok"}
        assert len(call_records) == 1
        assert call_records[0] == 10.0

    async def test_timeout_below_minimum_clamped_to_0_1(
        self, correlation_id: UUID
    ) -> None:
        """timeout_seconds < 0.1 must be clamped up to 0.1."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        result = await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
            timeout_seconds=0.001,
        )
        assert result == {"result": "ok"}

    async def test_effective_timeout_is_min_of_request_and_max(
        self, correlation_id: UUID
    ) -> None:
        """The effective timeout must be min(request, contract.max), clamped to >= 0.1.

        Exercises _execute_llm_http_call with different timeout_seconds values
        and captures the effective timeout passed through to httpx via the mock
        transport's request.extensions["timeout"].
        """
        captured_timeouts: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            timeout_ext = request.extensions.get("timeout")
            if timeout_ext is not None:
                # httpx passes timeout as a dict with pool/connect/read/write keys
                # or as a Timeout object; extract the pool value as representative
                if isinstance(timeout_ext, dict):
                    captured_timeouts.append(timeout_ext.get("pool", 0.0))
                else:
                    captured_timeouts.append(float(timeout_ext))
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client, max_timeout_seconds=10.0)

        # Case 1: request (5.0) < max (10.0) -> uses request (5.0)
        await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
            timeout_seconds=5.0,
        )

        # Case 2: request (30.0) > max (10.0) -> clamps to max (10.0)
        await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
            timeout_seconds=30.0,
        )

        # Case 3: request (0.05) very small -> clamps to floor (0.1)
        await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
            timeout_seconds=0.05,
        )

        assert len(captured_timeouts) == 3
        assert captured_timeouts[0] == 5.0
        assert captured_timeouts[1] == 10.0
        assert captured_timeouts[2] == 0.1


# ── Connection Errors ────────────────────────────────────────────────────


class TestConnectionErrors:
    """Validate connection-level error handling."""

    async def test_connection_refused_raises_infra_connection_error(
        self, correlation_id: UUID
    ) -> None:
        """Connection refused must raise InfraConnectionError after retries exhausted."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraConnectionError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

    async def test_timeout_raises_infra_timeout_error(
        self, correlation_id: UUID
    ) -> None:
        """HTTP timeout must raise InfraTimeoutError after retries exhausted."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Request timed out")

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraTimeoutError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )


# ── Retry Behavior ──────────────────────────────────────────────────────


class TestRetryBehavior:
    """Validate retry logic: count, backoff, and exhaustion."""

    async def test_zero_retries_means_single_attempt(
        self, correlation_id: UUID
    ) -> None:
        """max_retries=0 must make exactly one attempt and then raise."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"error": "server error"}, status_code=500)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraUnavailableError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

        assert call_count == 1

    async def test_retry_count_respected(self, correlation_id: UUID) -> None:
        """Total attempts must be 1 + max_retries."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"error": "server error"}, status_code=500)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraUnavailableError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=2,
            )

        assert call_count == 3  # 1 initial + 2 retries

    async def test_retry_succeeds_on_later_attempt(self, correlation_id: UUID) -> None:
        """If a retry attempt succeeds, the result is returned."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return _json_response({"error": "error"}, status_code=500)
            return _json_response({"result": "success"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        result = await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
            max_retries=3,
        )

        assert result == {"result": "success"}
        assert call_count == 3

    async def test_non_retriable_errors_do_not_retry(
        self, correlation_id: UUID
    ) -> None:
        """401 (non-retriable) must not trigger retry attempts."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"error": "unauthorized"}, status_code=401)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraAuthenticationError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=3,
            )

        # 401 is non-retriable, so only 1 attempt
        assert call_count == 1

    async def test_exponential_backoff_timing(self, correlation_id: UUID) -> None:
        """Retry delays must follow exponential backoff pattern."""
        sleep_calls: list[float] = []

        async def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            # Don't actually sleep in tests

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"error": "error"}, status_code=500)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with patch(
            "omnibase_infra.mixins.mixin_llm_http_transport.asyncio.sleep",
            side_effect=mock_sleep,
        ):
            with pytest.raises(InfraUnavailableError):
                await harness._execute_llm_http_call(
                    url=URL,
                    payload=PAYLOAD,
                    correlation_id=correlation_id,
                    max_retries=3,
                )

        # ModelRetryState default: delay_seconds=1.0, backoff_multiplier=2.0
        # Attempt 0 -> fail, next_attempt -> delay=1.0*2.0=2.0, sleep(2.0)
        # Attempt 1 -> fail, next_attempt -> delay=2.0*2.0=4.0, sleep(4.0)
        # Attempt 2 -> fail, next_attempt -> delay=4.0*2.0=8.0, sleep(8.0)
        # Attempt 3 -> fail, next_attempt -> attempt=4 >= max_attempts=4, raise
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == 2.0
        assert sleep_calls[1] == 4.0
        assert sleep_calls[2] == 8.0

    async def test_429_retry_uses_retry_after_delay(self, correlation_id: UUID) -> None:
        """429 retry should use Retry-After as the delay, not exponential backoff."""
        sleep_calls: list[float] = []
        call_count = 0

        async def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(
                    status_code=429,
                    content=b'{"error": "rate limited"}',
                    headers={
                        "content-type": "application/json",
                        "retry-after": "3",
                    },
                )
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with patch(
            "omnibase_infra.mixins.mixin_llm_http_transport.asyncio.sleep",
            side_effect=mock_sleep,
        ):
            result = await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=3,
            )

        assert result == {"result": "ok"}
        # The 429 delays should use the Retry-After value (3.0)
        assert any(delay == 3.0 for delay in sleep_calls)


# ── Circuit Breaker State Transitions ────────────────────────────────────


class TestCircuitBreakerStateTransitions:
    """Validate circuit breaker opens after threshold and transitions to half-open."""

    async def test_circuit_opens_after_threshold_failures(
        self, correlation_id: UUID
    ) -> None:
        """Circuit breaker must open after reaching failure threshold."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "error"}, status_code=500)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client, cb_threshold=3)

        # Each call with max_retries=0 generates 1 attempt -> 1 CB failure
        for _ in range(3):
            with pytest.raises(InfraUnavailableError):
                await harness._execute_llm_http_call(
                    url=URL,
                    payload=PAYLOAD,
                    correlation_id=correlation_id,
                    max_retries=0,
                )

        # After 3 failures (threshold=3), circuit should be open
        assert harness._circuit_breaker_open is True

    async def test_circuit_open_rejects_requests(self, correlation_id: UUID) -> None:
        """When circuit is open, requests must be rejected immediately."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "error"}, status_code=500)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client, cb_threshold=2)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(InfraUnavailableError):
                await harness._execute_llm_http_call(
                    url=URL,
                    payload=PAYLOAD,
                    correlation_id=correlation_id,
                    max_retries=0,
                )

        assert harness._circuit_breaker_open is True

        # Next call should be rejected by circuit breaker (InfraUnavailableError)
        call_count = 0
        original_handler = handler

        def counting_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return original_handler(request)

        # Replace transport
        harness._http_client = _make_mock_client(counting_handler)

        with pytest.raises(InfraUnavailableError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

        # The HTTP handler should NOT have been called (CB rejected before HTTP)
        assert call_count == 0

    async def test_circuit_half_open_after_reset_timeout(
        self, correlation_id: UUID
    ) -> None:
        """Circuit must transition to half-open after reset timeout elapses."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"error": "error"}, status_code=500)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client, cb_threshold=2)

        # Open the circuit
        for _ in range(2):
            with pytest.raises(InfraUnavailableError):
                await harness._execute_llm_http_call(
                    url=URL,
                    payload=PAYLOAD,
                    correlation_id=correlation_id,
                    max_retries=0,
                )

        assert harness._circuit_breaker_open is True

        # Simulate time passing beyond reset_timeout (60s)
        harness._circuit_breaker_open_until = time.time() - 1

        # Replace handler with success response
        def success_handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"result": "recovered"})

        harness._http_client = _make_mock_client(success_handler)

        # This call should succeed (circuit transitions to half-open, then closed)
        result = await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
            max_retries=0,
        )

        assert result == {"result": "recovered"}
        # Circuit should be closed after successful half-open request
        assert harness._circuit_breaker_open is False

    async def test_successful_call_resets_circuit_breaker(
        self, correlation_id: UUID
    ) -> None:
        """A successful HTTP call must reset the circuit breaker failure count."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client, cb_threshold=5)

        # Manually set some failures
        harness._circuit_breaker_failures = 3

        result = await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
        )

        assert result == {"result": "ok"}
        assert harness._circuit_breaker_failures == 0


# ── Error Classification ─────────────────────────────────────────────────


class TestErrorClassification:
    """Validate _classify_error for different exception types."""

    def test_classify_httpx_connect_error(self) -> None:
        """httpx.ConnectError should be classified as CONNECTION, retriable, CB failure."""
        harness = LlmTransportHarness()
        classification = harness._classify_error(httpx.ConnectError("refused"), "test")
        assert classification.should_retry is True
        assert classification.record_circuit_failure is True

    def test_classify_httpx_timeout_error(self) -> None:
        """httpx.TimeoutException should be classified as TIMEOUT, retriable, CB failure."""
        harness = LlmTransportHarness()
        classification = harness._classify_error(httpx.ReadTimeout("timeout"), "test")
        assert classification.should_retry is True
        assert classification.record_circuit_failure is True

    def test_classify_auth_error_no_retry_no_cb(self) -> None:
        """InfraAuthenticationError: no retry, no CB failure."""
        harness = LlmTransportHarness()
        classification = harness._classify_error(
            InfraAuthenticationError("auth failed"), "test"
        )
        assert classification.should_retry is False
        assert classification.record_circuit_failure is False

    def test_classify_rate_limited_error_retry_no_cb(self) -> None:
        """InfraRateLimitedError: retry yes, CB failure no."""
        harness = LlmTransportHarness()
        classification = harness._classify_error(
            InfraRateLimitedError("rate limited"), "test"
        )
        assert classification.should_retry is True
        assert classification.record_circuit_failure is False

    def test_classify_request_rejected_error_no_retry_no_cb(self) -> None:
        """InfraRequestRejectedError: no retry, no CB failure."""
        harness = LlmTransportHarness()
        classification = harness._classify_error(
            InfraRequestRejectedError("rejected"), "test"
        )
        assert classification.should_retry is False
        assert classification.record_circuit_failure is False

    def test_classify_protocol_config_error_no_retry_no_cb(self) -> None:
        """ProtocolConfigurationError: no retry, no CB failure."""
        harness = LlmTransportHarness()
        classification = harness._classify_error(
            ProtocolConfigurationError("not found"), "test"
        )
        assert classification.should_retry is False
        assert classification.record_circuit_failure is False

    def test_classify_unavailable_error_retry_cb_failure(self) -> None:
        """InfraUnavailableError: retry yes, CB failure yes."""
        harness = LlmTransportHarness()
        classification = harness._classify_error(
            InfraUnavailableError("unavailable"), "test"
        )
        assert classification.should_retry is True
        assert classification.record_circuit_failure is True

    def test_classify_unknown_error_retry_cb_failure(self) -> None:
        """Unknown exceptions: retry yes, CB failure yes (default)."""
        harness = LlmTransportHarness()
        classification = harness._classify_error(RuntimeError("unexpected"), "test")
        assert classification.should_retry is True
        assert classification.record_circuit_failure is True


# ── Protocol Error (non-JSON responses) ──────────────────────────────────


class TestProtocolErrors:
    """Validate handling of non-JSON 2xx responses and JSON parse failures."""

    async def test_non_json_content_type_raises_infra_protocol_error(
        self, correlation_id: UUID
    ) -> None:
        """2xx with non-JSON content-type must raise InfraProtocolError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _text_response(
                "<html>Not JSON</html>", status_code=200, content_type="text/html"
            )

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraProtocolError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

    async def test_invalid_json_body_raises_infra_protocol_error(
        self, correlation_id: UUID
    ) -> None:
        """2xx with JSON content-type but invalid JSON body must raise InfraProtocolError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                content=b"not valid json {{{",
                headers={"content-type": "application/json"},
            )

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraProtocolError):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

    async def test_empty_content_type_with_valid_json_succeeds(
        self, correlation_id: UUID
    ) -> None:
        """2xx with empty/missing content-type but valid JSON body must succeed."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                content=b'{"result": "ok"}',
                headers={},  # No content-type
            )

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        result = await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
        )
        assert result == {"result": "ok"}


# ── Successful Response ──────────────────────────────────────────────────


class TestSuccessfulResponse:
    """Validate happy-path response handling."""

    async def test_successful_json_response_returns_data(
        self, correlation_id: UUID
    ) -> None:
        """Valid 200 JSON response must return parsed data."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(
                {
                    "id": "chatcmpl-123",
                    "choices": [{"message": {"content": "Hello!"}}],
                }
            )

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        result = await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
        )

        assert result["id"] == "chatcmpl-123"
        choices = result["choices"]
        assert isinstance(choices, list)
        first_choice = choices[0]
        assert isinstance(first_choice, dict)
        message = first_choice["message"]
        assert isinstance(message, dict)
        assert message["content"] == "Hello!"

    async def test_case_insensitive_json_content_type(
        self, correlation_id: UUID
    ) -> None:
        """Content-type matching must be case-insensitive."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                content=b'{"result": "ok"}',
                headers={"content-type": "Application/JSON; charset=utf-8"},
            )

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        result = await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
        )
        assert result == {"result": "ok"}


# ── HTTP Client Management ───────────────────────────────────────────────


class TestHttpClientManagement:
    """Validate lazy HTTP client creation and lifecycle."""

    async def test_injected_client_is_used_directly(self, correlation_id: UUID) -> None:
        """When an external client is injected, it must be used without creating a new one."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        assert harness._http_client is client
        assert harness._owns_http_client is False

    async def test_lazy_client_creation(self, correlation_id: UUID) -> None:
        """Without injected client, a client must be created on first use."""
        harness = LlmTransportHarness()

        assert harness._http_client is None
        assert harness._owns_http_client is True

        client = await harness._get_http_client()
        assert client is not None
        assert harness._http_client is client

        # Cleanup
        await harness._close_http_client()

    async def test_close_only_closes_owned_client(self, correlation_id: UUID) -> None:
        """_close_http_client must not close an injected (non-owned) client."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        await harness._close_http_client()
        # Injected client should still be there (not closed by us)
        assert harness._http_client is client


# ── Transport Type and Target Name ───────────────────────────────────────


class TestTransportMetadata:
    """Validate transport type and target name accessor methods."""

    def test_get_transport_type_returns_http(self) -> None:
        """_get_transport_type must return HTTP."""
        from omnibase_infra.enums import EnumInfraTransportType

        harness = LlmTransportHarness()
        assert harness._get_transport_type() == EnumInfraTransportType.HTTP

    def test_get_target_name_returns_configured_name(self) -> None:
        """_get_target_name must return the configured target name."""
        harness = LlmTransportHarness(target_name="my-custom-llm")
        assert harness._get_target_name() == "my-custom-llm"
