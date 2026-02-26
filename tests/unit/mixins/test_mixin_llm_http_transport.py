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
- CIDR allowlist validation (OMN-2250)
- HMAC request signing (OMN-2250)
- Fail-closed behavior when secret is missing (OMN-2250)

Test Pattern:
    Uses httpx.MockTransport to simulate HTTP responses. The mixin is tested
    through a thin test harness class (LlmTransportHarness) that extends
    MixinLlmHttpTransport.

Related Tickets: OMN-2114 Phase 14, OMN-2250
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from ipaddress import IPv4Network
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
from omnibase_infra.mixins.mixin_llm_http_transport import (
    MixinLlmHttpTransport,
    _parse_cidr_allowlist,
)

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

#: URL uses an IP within the CIDR allowlist (192.168.86.0/24) so that
#: the endpoint trust boundary check passes for all standard tests.
URL = "http://192.168.86.201:8000/v1/chat/completions"
PAYLOAD: dict[str, Any] = {"messages": [{"role": "user", "content": "hello"}]}

#: Shared secret used in tests. Set via the autouse fixture below.
TEST_SHARED_SECRET = "test-hmac-secret-for-unit-tests"


@pytest.fixture(autouse=True)
def _set_llm_shared_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set LOCAL_LLM_SHARED_SECRET for all tests in this module.

    This is autouse so that the HMAC fail-closed check passes for standard
    transport tests. Tests that specifically validate missing-secret behavior
    override this by unsetting the variable.
    """
    monkeypatch.setenv("LOCAL_LLM_SHARED_SECRET", TEST_SHARED_SECRET)


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

    async def test_unexpected_non_httpx_exception_raises_infra_connection_error(
        self, correlation_id: UUID
    ) -> None:
        """An unexpected non-httpx exception (e.g. RuntimeError) from the transport
        must be caught by the generic Exception handler and raised as
        InfraConnectionError after retries are exhausted.
        """
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("unexpected transport failure")

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraConnectionError, match="RuntimeError") as exc_info:
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

        # Should have attempted exactly once (max_retries=0)
        assert call_count == 1
        # Original exception preserved in chain
        assert isinstance(exc_info.value.__cause__, RuntimeError)


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

    @pytest.mark.parametrize(
        ("status_code", "expected_error"),
        [
            (401, InfraAuthenticationError),
            (404, ProtocolConfigurationError),
            (400, InfraRequestRejectedError),
        ],
        ids=["401-auth", "404-config", "400-rejected"],
    )
    async def test_non_retriable_errors_do_not_retry(
        self,
        status_code: int,
        expected_error: type[Exception],
        correlation_id: UUID,
    ) -> None:
        """Non-retriable status codes (401, 404, 400) must not trigger retry attempts."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"error": "error"}, status_code=status_code)

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(expected_error):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=3,
            )

        # Non-retriable errors must produce exactly 1 attempt
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
        # The first two sleeps correspond to the two 429 responses and must
        # each use the Retry-After value (3.0), not exponential backoff.
        assert len(sleep_calls) >= 2
        assert sleep_calls[0] == 3.0
        assert sleep_calls[1] == 3.0


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

        def counting_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise AssertionError("HTTP handler should not be called")

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

    async def test_non_json_content_type_not_retried_despite_max_retries(
        self, correlation_id: UUID
    ) -> None:
        """InfraProtocolError from non-JSON content-type must NOT be retried.

        Although _classify_error marks InfraProtocolError as should_retry=True,
        _execute_llm_http_call explicitly re-raises it in the typed-exception
        handler block (alongside InfraAuthenticationError, etc.), bypassing the
        retry loop. This is by design: protocol errors on 2xx indicate a
        fundamental misconfiguration that retrying won't fix.

        This test verifies the end-to-end behavior: even with max_retries > 0,
        a non-JSON 2xx response produces exactly one attempt.
        """
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _text_response(
                "<html>Service Unavailable</html>",
                status_code=200,
                content_type="text/html",
            )

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraProtocolError, match="text/html"):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=3,
            )

        # InfraProtocolError is re-raised immediately, no retries
        assert call_count == 1

    @pytest.mark.parametrize(
        "content_type",
        [
            "application/json",
            "application/json; charset=utf-8",
            "Application/JSON; charset=utf-8",
        ],
        ids=["plain-json", "json-with-charset", "mixed-case-with-charset"],
    )
    async def test_json_content_type_variants_succeed(
        self, content_type: str, correlation_id: UUID
    ) -> None:
        """JSON content-type variants (with/without charset, mixed case) must all succeed."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=200,
                content=b'{"result": "ok"}',
                headers={"content-type": content_type},
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


# ── CIDR Allowlist Validation (OMN-2250) ──────────────────────────────


class TestCidrAllowlistValidation:
    """Validate CIDR allowlist enforcement on LLM endpoint URLs."""

    async def test_ip_within_allowlist_passes(self, correlation_id: UUID) -> None:
        """An IP within 192.168.86.0/24 must pass the allowlist check."""
        harness = LlmTransportHarness()
        # Should not raise
        await harness._validate_endpoint_allowlist(
            "http://192.168.86.201:8000/v1/completions", correlation_id
        )

    async def test_ip_outside_allowlist_rejected(self, correlation_id: UUID) -> None:
        """An IP outside 192.168.86.0/24 must raise InfraAuthenticationError."""
        harness = LlmTransportHarness()
        with pytest.raises(
            InfraAuthenticationError, match="outside the local LLM allowlist"
        ):
            await harness._validate_endpoint_allowlist(
                "http://10.0.0.1:8000/v1/completions", correlation_id
            )

    async def test_public_ip_rejected(self, correlation_id: UUID) -> None:
        """A public IP must raise InfraAuthenticationError."""
        harness = LlmTransportHarness()
        with pytest.raises(
            InfraAuthenticationError, match="outside the local LLM allowlist"
        ):
            await harness._validate_endpoint_allowlist(
                "http://8.8.8.8:8000/v1/completions", correlation_id
            )

    async def test_localhost_rejected(self, correlation_id: UUID) -> None:
        """127.0.0.1 (localhost) must raise InfraAuthenticationError."""
        harness = LlmTransportHarness()
        with pytest.raises(
            InfraAuthenticationError, match="outside the local LLM allowlist"
        ):
            await harness._validate_endpoint_allowlist(
                "http://127.0.0.1:8000/v1/completions", correlation_id
            )

    async def test_different_subnet_rejected(self, correlation_id: UUID) -> None:
        """192.168.87.1 (adjacent subnet) must raise InfraAuthenticationError."""
        harness = LlmTransportHarness()
        with pytest.raises(
            InfraAuthenticationError, match="outside the local LLM allowlist"
        ):
            await harness._validate_endpoint_allowlist(
                "http://192.168.87.1:8000/v1/completions", correlation_id
            )

    async def test_all_ips_in_subnet_accepted(self, correlation_id: UUID) -> None:
        """All IPs from .0 to .255 in the 192.168.86.0/24 range must pass."""
        harness = LlmTransportHarness()
        for octet in (0, 1, 100, 200, 201, 254, 255):
            await harness._validate_endpoint_allowlist(
                f"http://192.168.86.{octet}:8000/v1/completions", correlation_id
            )

    async def test_hostname_resolving_to_allowed_ip_passes(
        self, correlation_id: UUID
    ) -> None:
        """A hostname that resolves to an IP within the allowlist must pass."""
        harness = LlmTransportHarness()

        async def mock_getaddrinfo(
            *args: object, **kwargs: object
        ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
            return [(2, 1, 6, "", ("192.168.86.201", 0))]

        with patch.object(
            asyncio.get_running_loop(), "getaddrinfo", side_effect=mock_getaddrinfo
        ):
            await harness._validate_endpoint_allowlist(
                "http://my-local-llm:8000/v1/completions", correlation_id
            )

    async def test_hostname_resolving_to_disallowed_ip_rejected(
        self, correlation_id: UUID
    ) -> None:
        """A hostname resolving to an IP outside the allowlist must be rejected."""
        harness = LlmTransportHarness()

        async def mock_getaddrinfo(
            *args: object, **kwargs: object
        ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
            return [(2, 1, 6, "", ("10.0.0.5", 0))]

        with patch.object(
            asyncio.get_running_loop(), "getaddrinfo", side_effect=mock_getaddrinfo
        ):
            with pytest.raises(
                InfraAuthenticationError, match="outside the local LLM allowlist"
            ):
                await harness._validate_endpoint_allowlist(
                    "http://external-llm:8000/v1/completions", correlation_id
                )

    async def test_unresolvable_hostname_rejected(self, correlation_id: UUID) -> None:
        """A hostname that cannot be resolved must raise InfraAuthenticationError."""
        import socket as _socket

        harness = LlmTransportHarness()

        async def mock_getaddrinfo(
            *args: object, **kwargs: object
        ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
            raise _socket.gaierror("Name resolution failed")

        with patch.object(
            asyncio.get_running_loop(), "getaddrinfo", side_effect=mock_getaddrinfo
        ):
            with pytest.raises(
                InfraAuthenticationError, match="Cannot resolve hostname"
            ):
                await harness._validate_endpoint_allowlist(
                    "http://nonexistent-host:8000/v1/completions", correlation_id
                )

    async def test_empty_url_hostname_rejected(self, correlation_id: UUID) -> None:
        """A URL with no extractable hostname must raise InfraAuthenticationError."""
        harness = LlmTransportHarness()
        with pytest.raises(InfraAuthenticationError, match="Cannot extract hostname"):
            await harness._validate_endpoint_allowlist(
                "not-a-valid-url", correlation_id
            )

    async def test_allowlist_checked_before_http_call(
        self, correlation_id: UUID
    ) -> None:
        """Allowlist validation must run before any HTTP call is made.

        When the allowlist check fails, the HTTP handler must never be invoked.
        """
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(InfraAuthenticationError):
            await harness._execute_llm_http_call(
                url="http://10.0.0.1:8000/v1/completions",
                payload=PAYLOAD,
                correlation_id=correlation_id,
                max_retries=0,
            )

        # HTTP handler must NOT have been called
        assert call_count == 0

    async def test_cidr_allowlist_custom_range(self, correlation_id: UUID) -> None:
        """Patching LOCAL_LLM_CIDRS to a custom range must accept IPs in that range."""
        harness = LlmTransportHarness()

        with patch.object(
            MixinLlmHttpTransport,
            "LOCAL_LLM_CIDRS",
            (IPv4Network("10.0.0.0/8"),),
        ):
            # 10.0.0.1 is within 10.0.0.0/8 -- should pass
            await harness._validate_endpoint_allowlist(
                "http://10.0.0.1:8000/v1/completions", correlation_id
            )

    async def test_cidr_allowlist_multiple_ranges(self, correlation_id: UUID) -> None:
        """Patching LOCAL_LLM_CIDRS to multiple ranges must accept IPs in either range."""
        harness = LlmTransportHarness()

        with patch.object(
            MixinLlmHttpTransport,
            "LOCAL_LLM_CIDRS",
            (IPv4Network("10.0.0.0/8"), IPv4Network("172.16.0.0/12")),
        ):
            # 10.0.0.1 is within 10.0.0.0/8
            await harness._validate_endpoint_allowlist(
                "http://10.0.0.1:8000/v1/completions", correlation_id
            )
            # 172.16.5.10 is within 172.16.0.0/12
            await harness._validate_endpoint_allowlist(
                "http://172.16.5.10:8000/v1/completions", correlation_id
            )

    async def test_cidr_allowlist_empty_rejects_all(self, correlation_id: UUID) -> None:
        """Patching LOCAL_LLM_CIDRS to an empty tuple must reject all IPs."""
        harness = LlmTransportHarness()

        with patch.object(
            MixinLlmHttpTransport,
            "LOCAL_LLM_CIDRS",
            (),
        ):
            with pytest.raises(
                InfraAuthenticationError, match="outside the local LLM allowlist"
            ):
                await harness._validate_endpoint_allowlist(
                    "http://192.168.86.201:8000/v1/completions", correlation_id
                )


# ── CIDR Allowlist Parsing (OMN-2250) ─────────────────────────────────


@pytest.mark.unit
class TestParseCidrAllowlist:
    """Validate _parse_cidr_allowlist() with various env var inputs."""

    def test_parse_cidr_default(self) -> None:
        """Without setting env var, returns the default (192.168.86.0/24)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_ENDPOINT_CIDR_ALLOWLIST", None)
            result = _parse_cidr_allowlist()
        assert result == (IPv4Network("192.168.86.0/24"),)

    def test_parse_cidr_custom_single(self) -> None:
        """A single custom CIDR parses correctly."""
        with patch.dict(os.environ, {"LLM_ENDPOINT_CIDR_ALLOWLIST": "10.0.0.0/8"}):
            result = _parse_cidr_allowlist()
        assert result == (IPv4Network("10.0.0.0/8"),)

    def test_parse_cidr_multiple_comma_separated(self) -> None:
        """Multiple comma-separated CIDRs parse correctly."""
        with patch.dict(
            os.environ,
            {"LLM_ENDPOINT_CIDR_ALLOWLIST": "10.0.0.0/8, 172.16.0.0/12"},
        ):
            result = _parse_cidr_allowlist()
        assert result == (IPv4Network("10.0.0.0/8"), IPv4Network("172.16.0.0/12"))

    def test_parse_cidr_malformed_skipped(self) -> None:
        """Malformed entries are skipped; valid ones are kept."""
        with patch.dict(
            os.environ,
            {"LLM_ENDPOINT_CIDR_ALLOWLIST": "10.0.0.0/8, not-a-cidr, 172.16.0.0/12"},
        ):
            result = _parse_cidr_allowlist()
        assert result == (IPv4Network("10.0.0.0/8"), IPv4Network("172.16.0.0/12"))

    def test_parse_cidr_all_malformed_falls_back(self) -> None:
        """When all entries are malformed, falls back to default."""
        with patch.dict(
            os.environ,
            {"LLM_ENDPOINT_CIDR_ALLOWLIST": "garbage, also-garbage"},
        ):
            result = _parse_cidr_allowlist()
        assert result == (IPv4Network("192.168.86.0/24"),)

    def test_parse_cidr_host_bits_accepted(self) -> None:
        """strict=False auto-masks host bits: 192.168.86.100/24 -> 192.168.86.0/24."""
        with patch.dict(
            os.environ,
            {"LLM_ENDPOINT_CIDR_ALLOWLIST": "192.168.86.100/24"},
        ):
            result = _parse_cidr_allowlist()
        assert result == (IPv4Network("192.168.86.0/24"),)

    def test_parse_cidr_empty_string_falls_back(self) -> None:
        """An empty string env var falls back to default."""
        with patch.dict(os.environ, {"LLM_ENDPOINT_CIDR_ALLOWLIST": ""}):
            result = _parse_cidr_allowlist()
        assert result == (IPv4Network("192.168.86.0/24"),)

    def test_parse_cidr_default_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When env var is unset, a warning is logged about the default [OMN-2811]."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_ENDPOINT_CIDR_ALLOWLIST", None)
            with caplog.at_level("WARNING"):
                result = _parse_cidr_allowlist()
        assert result == (IPv4Network("192.168.86.0/24"),)
        assert any(
            "LLM_ENDPOINT_CIDR_ALLOWLIST not set" in record.message
            for record in caplog.records
        )

    def test_parse_cidr_explicit_value_no_default_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When env var is explicitly set, no 'not set' warning is logged [OMN-2811]."""
        with patch.dict(os.environ, {"LLM_ENDPOINT_CIDR_ALLOWLIST": "10.0.0.0/8"}):
            with caplog.at_level("WARNING"):
                _parse_cidr_allowlist()
        assert not any(
            "LLM_ENDPOINT_CIDR_ALLOWLIST not set" in record.message
            for record in caplog.records
        )


# ── HMAC Signing (OMN-2250) ───────────────────────────────────────────


class TestHmacSigning:
    """Validate HMAC-SHA256 request signing for LLM endpoints."""

    def test_hmac_signature_computed_correctly(self, correlation_id: UUID) -> None:
        """HMAC-SHA256 signature must match manual computation."""
        harness = LlmTransportHarness()
        payload: dict[str, Any] = {"messages": [{"role": "user", "content": "hello"}]}

        signature = harness._compute_hmac_signature(payload, correlation_id)

        # Manually compute expected signature
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        expected = hmac.new(
            TEST_SHARED_SECRET.encode("utf-8"),
            canonical,
            hashlib.sha256,
        ).hexdigest()

        assert signature == expected

    def test_hmac_signature_deterministic(self, correlation_id: UUID) -> None:
        """Same payload must produce identical signatures across calls."""
        harness = LlmTransportHarness()
        payload: dict[str, Any] = {"model": "test", "prompt": "hello"}

        sig1 = harness._compute_hmac_signature(payload, correlation_id)
        sig2 = harness._compute_hmac_signature(payload, correlation_id)

        assert sig1 == sig2

    def test_hmac_signature_changes_with_payload(self, correlation_id: UUID) -> None:
        """Different payloads must produce different signatures."""
        harness = LlmTransportHarness()

        sig1 = harness._compute_hmac_signature({"a": 1}, correlation_id)
        sig2 = harness._compute_hmac_signature({"a": 2}, correlation_id)

        assert sig1 != sig2

    def test_hmac_signature_changes_with_secret(
        self, correlation_id: UUID, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Different secrets must produce different signatures for the same payload."""
        harness = LlmTransportHarness()
        payload: dict[str, Any] = {"test": "data"}

        monkeypatch.setenv("LOCAL_LLM_SHARED_SECRET", "secret-a")
        sig_a = harness._compute_hmac_signature(payload, correlation_id)

        monkeypatch.setenv("LOCAL_LLM_SHARED_SECRET", "secret-b")
        sig_b = harness._compute_hmac_signature(payload, correlation_id)

        assert sig_a != sig_b

    def test_hmac_uses_canonical_json_sorted_keys(self, correlation_id: UUID) -> None:
        """Signature must be based on sorted-key canonical JSON.

        Two dicts with the same content but different insertion order
        must produce the same signature.
        """
        harness = LlmTransportHarness()

        payload_a: dict[str, Any] = {"z": 1, "a": 2}
        payload_b: dict[str, Any] = {"a": 2, "z": 1}

        sig_a = harness._compute_hmac_signature(payload_a, correlation_id)
        sig_b = harness._compute_hmac_signature(payload_b, correlation_id)

        assert sig_a == sig_b

    async def test_hmac_header_sent_in_request(self, correlation_id: UUID) -> None:
        """The x-omn-node-signature header must be present in outbound HTTP requests."""
        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            for key, value in request.headers.items():
                captured_headers[key.lower()] = value
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
        )

        assert "x-omn-node-signature" in captured_headers
        # Verify the header value matches computed signature
        expected_sig = harness._compute_hmac_signature(PAYLOAD, correlation_id)
        assert captured_headers["x-omn-node-signature"] == expected_sig

    def test_hmac_signature_is_hex_string(self, correlation_id: UUID) -> None:
        """HMAC signature must be a valid hex string of length 64 (SHA-256)."""
        harness = LlmTransportHarness()
        signature = harness._compute_hmac_signature(PAYLOAD, correlation_id)

        assert len(signature) == 64
        assert all(c in "0123456789abcdef" for c in signature)


# ── Fail-Closed Behavior (OMN-2250) ──────────────────────────────────


class TestFailClosedBehavior:
    """Validate fail-closed behavior: missing secret or invalid config rejects requests."""

    async def test_missing_secret_rejects_request(
        self, correlation_id: UUID, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When LOCAL_LLM_SHARED_SECRET is not set, requests must be rejected."""
        monkeypatch.delenv("LOCAL_LLM_SHARED_SECRET", raising=False)

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(ProtocolConfigurationError, match="LOCAL_LLM_SHARED_SECRET"):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
            )

        # HTTP handler must NOT have been called
        assert call_count == 0

    async def test_empty_secret_rejects_request(
        self, correlation_id: UUID, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When LOCAL_LLM_SHARED_SECRET is set to empty string, requests must be rejected."""
        monkeypatch.setenv("LOCAL_LLM_SHARED_SECRET", "")

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        with pytest.raises(ProtocolConfigurationError, match="LOCAL_LLM_SHARED_SECRET"):
            await harness._execute_llm_http_call(
                url=URL,
                payload=PAYLOAD,
                correlation_id=correlation_id,
            )

        # HTTP handler must NOT have been called
        assert call_count == 0

    def test_compute_hmac_fails_closed_on_missing_secret(
        self, correlation_id: UUID, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_compute_hmac_signature must raise ProtocolConfigurationError when secret is missing."""
        monkeypatch.delenv("LOCAL_LLM_SHARED_SECRET", raising=False)
        harness = LlmTransportHarness()

        with pytest.raises(
            ProtocolConfigurationError, match="HMAC signing requires a shared secret"
        ):
            harness._compute_hmac_signature(PAYLOAD, correlation_id)

    async def test_allowlist_failure_prevents_hmac_computation(
        self, correlation_id: UUID
    ) -> None:
        """If allowlist check fails, HMAC computation must not be attempted.

        This validates the ordering: allowlist check runs first, and if it
        fails, the HMAC signing path is never reached.
        """
        harness = LlmTransportHarness()

        hmac_called = False
        original_compute = harness._compute_hmac_signature

        def tracking_compute(payload: dict[str, Any], cid: UUID) -> str:
            nonlocal hmac_called
            hmac_called = True
            return original_compute(payload, cid)

        harness._compute_hmac_signature = tracking_compute  # type: ignore[assignment]

        with pytest.raises(InfraAuthenticationError):
            await harness._execute_llm_http_call(
                url="http://10.0.0.1:8000/v1/completions",
                payload=PAYLOAD,
                correlation_id=correlation_id,
            )

        assert hmac_called is False

    async def test_both_checks_pass_allows_request(self, correlation_id: UUID) -> None:
        """When both CIDR allowlist and HMAC signing pass, the HTTP call proceeds."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"result": "ok"})

        client = _make_mock_client(handler)
        harness = LlmTransportHarness(http_client=client)

        result = await harness._execute_llm_http_call(
            url=URL,
            payload=PAYLOAD,
            correlation_id=correlation_id,
        )

        assert result == {"result": "ok"}
