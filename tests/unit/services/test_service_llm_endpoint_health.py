# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ServiceLlmEndpointHealth.

This test suite validates:
- Service instantiation and configuration
- Health probing with /health and /v1/models fallback
- In-memory status map tracking
- Per-endpoint circuit breaker behaviour
- Event emission to Kafka via ProtocolEventBusLike
- Background probe loop start/stop lifecycle
- Error handling during probes and event emission

Test Organization:
    - TestModelLlmEndpointHealthConfig: Configuration validation
    - TestModelLlmEndpointStatus: Status model behaviour
    - TestServiceLlmEndpointHealthInit: Constructor and setup
    - TestServiceLlmEndpointHealthProbe: Core probe logic
    - TestServiceLlmEndpointHealthCircuitBreaker: CB integration
    - TestServiceLlmEndpointHealthEventEmission: Kafka events
    - TestServiceLlmEndpointHealthLifecycle: Start/stop

Related Tickets:
    - OMN-2255: LLM endpoint health checker service
    - OMN-2249: SLO profiling baselines
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from omnibase_infra.services.service_llm_endpoint_health import (
    TOPIC_LLM_ENDPOINT_HEALTH,
    EndpointCircuitBreaker,
    ModelLlmEndpointHealthConfig,
    ModelLlmEndpointHealthEvent,
    ModelLlmEndpointStatus,
    ServiceLlmEndpointHealth,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_endpoints() -> dict[str, str]:
    """Return a sample endpoint mapping for tests."""
    return {
        "coder-14b": "http://192.168.86.201:8000",
        "qwen-72b": "http://192.168.86.200:8100",
    }


@pytest.fixture
def config(sample_endpoints: dict[str, str]) -> ModelLlmEndpointHealthConfig:
    """Return a default config with two endpoints."""
    return ModelLlmEndpointHealthConfig(
        endpoints=sample_endpoints,
        probe_interval_seconds=5.0,
        probe_timeout_seconds=2.0,
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout=30.0,
    )


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Return a mock ProtocolEventBusLike."""
    bus = AsyncMock()
    bus.publish_envelope = AsyncMock()
    return bus


@pytest.fixture
def service(
    config: ModelLlmEndpointHealthConfig,
    mock_event_bus: AsyncMock,
) -> ServiceLlmEndpointHealth:
    """Return a ServiceLlmEndpointHealth wired with mocks."""
    return ServiceLlmEndpointHealth(config=config, event_bus=mock_event_bus)


@pytest.fixture
def service_no_bus(
    config: ModelLlmEndpointHealthConfig,
) -> ServiceLlmEndpointHealth:
    """Return a ServiceLlmEndpointHealth without an event bus."""
    return ServiceLlmEndpointHealth(config=config, event_bus=None)


# =============================================================================
# TestModelLlmEndpointHealthConfig
# =============================================================================


class TestModelLlmEndpointHealthConfig:
    """Validate configuration model constraints."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "invalid_url",
        [
            "ftp://example.com",
            "not-a-url",
            "",
            "ws://example.com:8000",
            "file:///tmp/model",
            "tcp://192.168.1.1:8000",
        ],
        ids=[
            "ftp-scheme",
            "bare-string",
            "empty-string",
            "websocket-scheme",
            "file-scheme",
            "tcp-scheme",
        ],
    )
    def test_invalid_endpoint_url_rejected(self, invalid_url: str) -> None:
        """Endpoint URLs with non-HTTP(S) schemes must raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelLlmEndpointHealthConfig(
                endpoints={"bad-ep": invalid_url},
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "empty_netloc_url",
        [
            "http://",
            "https://",
            "http:///health",
            "https:///v1/models",
        ],
        ids=[
            "http-empty-netloc",
            "https-empty-netloc",
            "http-empty-netloc-with-path",
            "https-empty-netloc-with-path",
        ],
    )
    def test_empty_netloc_url_rejected(self, empty_netloc_url: str) -> None:
        """Endpoint URLs with empty netloc (no hostname) must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmEndpointHealthConfig(
                endpoints={"bad-ep": empty_netloc_url},
            )
        error_msg = exc_info.value.errors()[0]["msg"]
        assert "hostname" in error_msg.lower()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "valid_url",
        [
            "http://example.com",
            "https://example.com",
            "http://192.168.86.201:8000",
            "https://example.com:8000/health",
            "http://localhost:9999",
        ],
        ids=[
            "http-plain",
            "https-plain",
            "http-ip-port",
            "https-host-port-path",
            "http-localhost",
        ],
    )
    def test_valid_endpoint_url_accepted(self, valid_url: str) -> None:
        """Endpoint URLs with HTTP or HTTPS schemes must be accepted."""
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"good-ep": valid_url},
        )
        assert cfg.endpoints["good-ep"] == valid_url

    def test_defaults(self) -> None:
        """Default config should have empty endpoints and sensible defaults."""
        cfg = ModelLlmEndpointHealthConfig()
        assert cfg.endpoints == {}
        assert cfg.probe_interval_seconds == 30.0
        assert cfg.probe_timeout_seconds == 5.0
        assert cfg.circuit_breaker_threshold == 3
        assert cfg.circuit_breaker_reset_timeout == 60.0

    def test_frozen(self, config: ModelLlmEndpointHealthConfig) -> None:
        """Config should be immutable."""
        with pytest.raises(ValidationError):
            config.probe_interval_seconds = 999  # type: ignore[misc]

    @pytest.mark.unit
    def test_invalid_url_does_not_leak_query_params(self) -> None:
        """Validation error messages must NOT expose query-string tokens.

        URLs may carry credentials as query parameters (e.g.
        ``ftp://host:8000/v1?token=supersecret``).  The sanitized error
        message must strip the query string so that secrets are never
        included in user-visible error output.

        Note: Pydantic's ``str(ValidationError)`` also renders the raw
        ``input_value``, which is outside our control.  We verify the
        error *message* (``errors()[0]['msg']``) which is the text that
        handlers and loggers typically extract.
        """
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmEndpointHealthConfig(
                endpoints={"leaky": "ftp://host:8000/v1?token=supersecret&key=abc123"},
            )
        error_msg = exc_info.value.errors()[0]["msg"]
        assert "supersecret" not in error_msg
        assert "abc123" not in error_msg
        assert "token=" not in error_msg
        assert "key=" not in error_msg
        # The sanitized URL (scheme + host + path) should still appear
        assert "ftp://host:8000/v1" in error_msg

    @pytest.mark.unit
    def test_invalid_url_does_not_leak_fragment(self) -> None:
        """Fragments should also be stripped from validation error messages."""
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmEndpointHealthConfig(
                endpoints={"frag": "ftp://host:8000/v1#secret-anchor"},
            )
        error_msg = exc_info.value.errors()[0]["msg"]
        assert "secret-anchor" not in error_msg
        assert "ftp://host:8000/v1" in error_msg

    def test_probe_interval_minimum(self) -> None:
        """Probe interval must be >= 1."""
        with pytest.raises(ValidationError):
            ModelLlmEndpointHealthConfig(probe_interval_seconds=0.5)

    def test_probe_timeout_range(self) -> None:
        """Probe timeout must be within [0.5, 30.0]."""
        with pytest.raises(ValidationError):
            ModelLlmEndpointHealthConfig(probe_timeout_seconds=0.1)
        with pytest.raises(ValidationError):
            ModelLlmEndpointHealthConfig(probe_timeout_seconds=60.0)


# =============================================================================
# TestModelLlmEndpointStatus
# =============================================================================


class TestModelLlmEndpointStatus:
    """Validate status model."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "invalid_url",
        [
            "ftp://example.com",
            "not-a-url",
            "",
            "ws://example.com:8000",
        ],
        ids=["ftp-scheme", "bare-string", "empty-string", "websocket-scheme"],
    )
    def test_invalid_status_url_rejected(self, invalid_url: str) -> None:
        """Status model URLs with non-HTTP(S) schemes must raise ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            ModelLlmEndpointStatus(
                url=invalid_url,
                name="test",
                available=True,
                last_check=now,
                latency_ms=10.0,
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "valid_url",
        [
            "http://example.com",
            "https://example.com:8000/health",
            "http://192.168.86.201:8000",
        ],
        ids=["http-plain", "https-host-port-path", "http-ip-port"],
    )
    def test_valid_status_url_accepted(self, valid_url: str) -> None:
        """Status model URLs with HTTP or HTTPS schemes must be accepted."""
        now = datetime.now(UTC)
        status = ModelLlmEndpointStatus(
            url=valid_url,
            name="test",
            available=True,
            last_check=now,
            latency_ms=10.0,
        )
        assert status.url == valid_url

    @pytest.mark.unit
    def test_invalid_status_url_does_not_leak_query_params(self) -> None:
        """Status model validation error messages must NOT expose query-string secrets.

        See ``TestModelLlmEndpointHealthConfig.test_invalid_url_does_not_leak_query_params``
        for a detailed explanation of why we check ``errors()[0]['msg']``
        rather than ``str(ValidationError)``.
        """
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmEndpointStatus(
                url="ftp://host:8000/v1?token=supersecret&key=abc123",
                name="test",
                available=True,
                last_check=now,
                latency_ms=10.0,
            )
        error_msg = exc_info.value.errors()[0]["msg"]
        assert "supersecret" not in error_msg
        assert "abc123" not in error_msg
        assert "token=" not in error_msg
        assert "key=" not in error_msg
        assert "ftp://host:8000/v1" in error_msg

    @pytest.mark.unit
    def test_invalid_status_url_does_not_leak_fragment(self) -> None:
        """Status model validation error messages must NOT expose URL fragments."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelLlmEndpointStatus(
                url="ftp://host:8000/v1#secret-anchor",
                name="test",
                available=True,
                last_check=now,
                latency_ms=10.0,
            )
        error_msg = exc_info.value.errors()[0]["msg"]
        assert "secret-anchor" not in error_msg
        assert "ftp://host:8000/v1" in error_msg

    def test_healthy_status(self) -> None:
        """Healthy status should have available=True."""
        now = datetime.now(UTC)
        status = ModelLlmEndpointStatus(
            url="http://localhost:8000",
            name="test",
            available=True,
            last_check=now,
            latency_ms=42.5,
        )
        assert status.available is True
        assert status.latency_ms == 42.5
        assert status.error == ""
        assert status.circuit_state == "closed"

    def test_unhealthy_status(self) -> None:
        """Unhealthy status should carry error info."""
        now = datetime.now(UTC)
        status = ModelLlmEndpointStatus(
            url="http://localhost:8000",
            name="test",
            available=False,
            last_check=now,
            latency_ms=-1.0,
            error="Connection refused",
            circuit_state="open",
        )
        assert status.available is False
        assert status.latency_ms == -1.0
        assert status.error == "Connection refused"
        assert status.circuit_state == "open"

    def test_frozen(self) -> None:
        """Status model should be immutable."""
        now = datetime.now(UTC)
        status = ModelLlmEndpointStatus(
            url="http://localhost:8000",
            name="test",
            available=True,
            last_check=now,
            latency_ms=10.0,
        )
        with pytest.raises(ValidationError):
            status.available = False  # type: ignore[misc]


# =============================================================================
# TestServiceLlmEndpointHealthInit
# =============================================================================


class TestServiceLlmEndpointHealthInit:
    """Validate constructor and initial state."""

    def test_creates_circuit_breakers_per_endpoint(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """Each configured endpoint should get its own circuit breaker."""
        # Whitebox: access internal CB map to verify per-endpoint isolation.
        # No public API enumerates configured circuit breakers by name.
        assert "coder-14b" in service._circuit_breakers
        assert "qwen-72b" in service._circuit_breakers
        assert len(service._circuit_breakers) == 2

    def test_initial_status_map_empty(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """Status map should be empty before first probe."""
        assert service.get_status() == {}

    def test_not_running_initially(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """Service should not be running after construction."""
        assert service.is_running is False

    def test_no_event_bus(
        self,
        service_no_bus: ServiceLlmEndpointHealth,
    ) -> None:
        """Service should work without an event bus."""
        # Whitebox: verify the internal event bus reference is None.
        # There is no public accessor; the observable effect (no events
        # emitted) is tested in TestServiceLlmEndpointHealthEventEmission.
        assert service_no_bus._event_bus is None

    def test_empty_endpoints(self) -> None:
        """Service with no endpoints should still construct."""
        cfg = ModelLlmEndpointHealthConfig(endpoints={})
        svc = ServiceLlmEndpointHealth(config=cfg)
        assert svc.get_status() == {}
        # Whitebox: verify no circuit breakers created for empty config.
        assert len(svc._circuit_breakers) == 0


# =============================================================================
# TestServiceLlmEndpointHealthProbe
# =============================================================================


class TestServiceLlmEndpointHealthProbe:
    """Validate probe logic with mocked HTTP calls."""

    @pytest.mark.asyncio
    async def test_probe_health_endpoint_success(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """When /health returns 200, endpoint should be marked available."""
        mock_response = httpx.Response(200, request=httpx.Request("GET", "http://test"))

        with patch.object(
            httpx.AsyncClient, "get", return_value=mock_response
        ) as mock_get:
            status_map = await service.probe_all()

        assert len(status_map) == 2
        for name, status in status_map.items():
            assert status.available is True
            assert status.latency_ms > 0
            assert status.error == ""

    @pytest.mark.asyncio
    async def test_probe_fallback_to_v1_models(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """When /health fails, should fall back to /v1/models."""
        health_fail = httpx.Response(
            500, request=httpx.Request("GET", "http://test/health")
        )
        models_ok = httpx.Response(
            200, request=httpx.Request("GET", "http://test/v1/models")
        )

        call_count = 0

        async def mock_get(url: str, **kwargs: object) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if "/health" in url:
                return health_fail
            return models_ok

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            status_map = await service.probe_all()

        # Each endpoint should have been probed via /health then /v1/models
        for status in status_map.values():
            assert status.available is True

    @pytest.mark.asyncio
    async def test_probe_both_fail(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """When both /health and /v1/models fail, endpoint is unavailable."""
        fail_response = httpx.Response(503, request=httpx.Request("GET", "http://test"))

        with patch.object(httpx.AsyncClient, "get", return_value=fail_response):
            status_map = await service.probe_all()

        for status in status_map.values():
            assert status.available is False
            assert "503" in status.error

    @pytest.mark.asyncio
    async def test_probe_connection_error(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """Connection errors should result in unavailable status."""

        async def raise_connection_error(url: str, **kwargs: object) -> None:
            raise httpx.ConnectError("Connection refused")

        with patch.object(httpx.AsyncClient, "get", side_effect=raise_connection_error):
            status_map = await service.probe_all()

        for status in status_map.values():
            assert status.available is False
            assert "ConnectError" in status.error or "Connection" in status.error

    @pytest.mark.asyncio
    async def test_probe_exception_message_sanitized(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """Catch-all exception handler must sanitize messages to prevent credential leakage.

        When an unexpected exception contains sensitive data (e.g. a connection
        string with embedded credentials), the error stored in the status model
        must be redacted while preserving the exception type name for debugging.
        """

        async def raise_sensitive_error(url: str, **kwargs: object) -> None:
            raise RuntimeError(
                "Cannot connect to postgresql://admin:s3cret@db:5432/prod"
            )

        with patch.object(httpx.AsyncClient, "get", side_effect=raise_sensitive_error):
            status_map = await service.probe_all()

        for status in status_map.values():
            assert status.available is False
            # Exception type name must be preserved for debugging
            assert "RuntimeError" in status.error
            # Sensitive data must be redacted
            assert "s3cret" not in status.error
            assert "admin" not in status.error
            assert "postgresql://" not in status.error
            assert "[REDACTED" in status.error

    @pytest.mark.asyncio
    async def test_get_endpoint_status(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """get_endpoint_status should return per-endpoint status."""
        mock_response = httpx.Response(200, request=httpx.Request("GET", "http://test"))

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            await service.probe_all()

        status = service.get_endpoint_status("coder-14b")
        assert status is not None
        assert status.name == "coder-14b"
        assert status.available is True

        # Non-existent endpoint returns None
        assert service.get_endpoint_status("nonexistent") is None

    @pytest.mark.asyncio
    async def test_status_map_returns_copy(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """get_status should return a copy, not internal state."""
        mock_response = httpx.Response(200, request=httpx.Request("GET", "http://test"))

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            await service.probe_all()

        map1 = service.get_status()
        map2 = service.get_status()
        assert map1 is not map2
        assert map1 == map2


# =============================================================================
# TestServiceLlmEndpointHealthCircuitBreaker
# =============================================================================


class TestServiceLlmEndpointHealthCircuitBreaker:
    """Validate per-endpoint circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """After 3 consecutive failures, circuit should open."""

        async def raise_error(url: str, **kwargs: object) -> None:
            raise httpx.ConnectError("Connection refused")

        with patch.object(httpx.AsyncClient, "get", side_effect=raise_error):
            # Probe 3 times to reach threshold
            for _ in range(3):
                await service.probe_all()

        # After 3 failures with threshold=3, circuit should be open.
        # Whitebox: access internal CB map to call the public is_open
        # property.  The service has no public method to query a single
        # endpoint's circuit breaker state; the status map only reports
        # the last probe result, not the live CB state.
        cb = service._circuit_breakers["coder-14b"]
        assert cb.is_open is True

    @pytest.mark.asyncio
    async def test_open_circuit_returns_immediately(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """When circuit is open, probe should return immediately without HTTP."""

        async def raise_error(url: str, **kwargs: object) -> None:
            raise httpx.ConnectError("Connection refused")

        # Trip the circuit
        with patch.object(httpx.AsyncClient, "get", side_effect=raise_error):
            for _ in range(3):
                await service.probe_all()

        # Now probe again -- should not make HTTP calls
        with patch.object(httpx.AsyncClient, "get") as mock_get:
            status_map = await service.probe_all()
            # The mock should NOT have been called for the tripped endpoints
            # (though it might be called for endpoints not yet tripped)

        for status in status_map.values():
            assert status.available is False
            assert "circuit" in status.error.lower() or "Circuit" in status.error

    @pytest.mark.asyncio
    async def test_circuit_breakers_are_independent(
        self,
    ) -> None:
        """Each endpoint should have its own independent circuit breaker."""
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={
                "ep-a": "http://a:8000",
                "ep-b": "http://b:8000",
            },
            circuit_breaker_threshold=2,
        )
        svc = ServiceLlmEndpointHealth(config=cfg)

        call_urls: list[str] = []

        async def selective_fail(url: str, **kwargs: object) -> httpx.Response:
            call_urls.append(url)
            if "://a:" in url:
                raise httpx.ConnectError("refused")
            return httpx.Response(200, request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", side_effect=selective_fail):
            # Trip ep-a (2 failures = threshold)
            for _ in range(2):
                await svc.probe_all()

        # Whitebox: verify CB isolation -- ep-a open, ep-b still closed.
        # See comment in test_circuit_opens_after_threshold_failures for
        # rationale on accessing _circuit_breakers directly.
        assert svc._circuit_breakers["ep-a"].is_open is True
        assert svc._circuit_breakers["ep-b"].is_open is False

    @pytest.mark.asyncio
    async def test_endpoint_circuit_breaker_class(self) -> None:
        """EndpointCircuitBreaker should initialize CB state."""
        cb = EndpointCircuitBreaker(
            endpoint_name="test",
            threshold=5,
            reset_timeout=60.0,
        )
        state = cb.get_state()
        assert state["initialized"] is True
        assert state["state"] == "closed"
        assert state["threshold"] == 5

    @pytest.mark.asyncio
    async def test_circuit_recovers_after_reset_timeout(self) -> None:
        """After reset_timeout elapses, a successful probe should close the circuit.

        This validates the full CLOSED -> OPEN -> HALF_OPEN -> CLOSED recovery
        transition.  The test:
          1. Trips the circuit with consecutive failures.
          2. Patches ``time.time`` inside the mixin to simulate time advancing
             past the ``reset_timeout``, causing the CB to transition from
             OPEN -> HALF_OPEN on the next ``check()`` call.
          3. Issues a successful probe (half-open allows one trial).
          4. Asserts the circuit is closed again and the endpoint is healthy.
        """
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"recovery-ep": "http://localhost:7777"},
            circuit_breaker_threshold=2,
            circuit_breaker_reset_timeout=30.0,
        )
        svc = ServiceLlmEndpointHealth(config=cfg)

        # -- Phase 1: Trip the circuit with consecutive failures ---------------
        async def raise_error(url: str, **kwargs: object) -> None:
            raise httpx.ConnectError("Connection refused")

        with patch.object(httpx.AsyncClient, "get", side_effect=raise_error):
            for _ in range(2):
                await svc.probe_all()

        # Whitebox: access internal CB map to assert the live circuit state.
        # See comment in test_circuit_opens_after_threshold_failures.
        cb = svc._circuit_breakers["recovery-ep"]
        assert cb.is_open is True, "Circuit should be open after hitting threshold"

        # Verify the status map reflects the open circuit
        status_before = svc.get_endpoint_status("recovery-ep")
        assert status_before is not None
        assert status_before.available is False

        # -- Phase 2: Simulate time advancing past reset_timeout ---------------
        # The mixin's _check_circuit_breaker compares time.time() against the
        # internal open_until timestamp.  Rather than mutating the private
        # attribute directly, we patch time.time in the mixin module to
        # return a value far enough in the future to trigger the
        # OPEN -> HALF_OPEN transition.
        far_future = cb._circuit_breaker_open_until + 1.0

        # -- Phase 3: Successful probe in half-open state ----------------------
        mock_response = httpx.Response(
            200, request=httpx.Request("GET", "http://localhost:7777/health")
        )
        with (
            patch(
                "omnibase_infra.mixins.mixin_async_circuit_breaker.time.time",
                return_value=far_future,
            ),
            patch.object(httpx.AsyncClient, "get", return_value=mock_response),
        ):
            status_map = await svc.probe_all()

        # -- Phase 4: Verify full recovery -------------------------------------
        assert cb.is_open is False, "Circuit should be closed after successful probe"

        recovered_status = status_map["recovery-ep"]
        assert recovered_status.available is True
        assert recovered_status.circuit_state == "closed"
        assert recovered_status.error == ""
        assert recovered_status.latency_ms > 0


# =============================================================================
# TestServiceLlmEndpointHealthEventEmission
# =============================================================================


class TestServiceLlmEndpointHealthEventEmission:
    """Validate Kafka event emission."""

    @pytest.mark.asyncio
    async def test_emits_event_after_probe(
        self,
        service: ServiceLlmEndpointHealth,
        mock_event_bus: AsyncMock,
    ) -> None:
        """After probing, should emit a health event."""
        mock_response = httpx.Response(200, request=httpx.Request("GET", "http://test"))

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            await service.probe_all()

        mock_event_bus.publish_envelope.assert_called_once()
        call_args = mock_event_bus.publish_envelope.call_args
        assert call_args.kwargs["topic"] == TOPIC_LLM_ENDPOINT_HEALTH

    @pytest.mark.asyncio
    async def test_no_emission_without_bus(
        self,
        service_no_bus: ServiceLlmEndpointHealth,
    ) -> None:
        """Without event bus, probing should succeed without emission."""
        mock_response = httpx.Response(200, request=httpx.Request("GET", "http://test"))

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            status_map = await service_no_bus.probe_all()

        assert len(status_map) == 2
        for status in status_map.values():
            assert status.available is True

    @pytest.mark.asyncio
    async def test_emission_failure_does_not_crash_probe(
        self,
        service: ServiceLlmEndpointHealth,
        mock_event_bus: AsyncMock,
    ) -> None:
        """If event emission fails, the probe should still succeed."""
        mock_event_bus.publish_envelope.side_effect = RuntimeError("Kafka down")
        mock_response = httpx.Response(200, request=httpx.Request("GET", "http://test"))

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            # Should not raise
            status_map = await service.probe_all()

        assert len(status_map) == 2
        for status in status_map.values():
            assert status.available is True

    @pytest.mark.asyncio
    async def test_no_emission_for_empty_endpoints(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """With no endpoints, no event should be emitted."""
        cfg = ModelLlmEndpointHealthConfig(endpoints={})
        svc = ServiceLlmEndpointHealth(config=cfg, event_bus=mock_event_bus)
        await svc.probe_all()
        mock_event_bus.publish_envelope.assert_not_called()


# =============================================================================
# TestServiceLlmEndpointHealthLifecycle
# =============================================================================


class TestServiceLlmEndpointHealthLifecycle:
    """Validate start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """Service should start and stop cleanly."""
        # Patch probe_all to avoid real HTTP
        with patch.object(service, "probe_all", new_callable=AsyncMock):
            await service.start()
            assert service.is_running is True
            # Whitebox: verify the background task handle exists.
            # The public is_running property confirms the service is active,
            # but does not prove an asyncio.Task was actually created.
            assert service._probe_task is not None

            await service.stop()
            assert service.is_running is False
            # Whitebox: verify task handle is cleaned up after stop.
            assert service._probe_task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """Calling start twice should be safe."""
        with patch.object(service, "probe_all", new_callable=AsyncMock):
            await service.start()
            # Whitebox: capture the task handle to verify idempotency --
            # a second start() must reuse the same asyncio.Task object.
            task1 = service._probe_task
            await service.start()  # idempotent
            assert service._probe_task is task1
            await service.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """Calling stop on a stopped service should be safe."""
        await service.stop()  # No-op on never-started service
        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_stop_closes_http_client_without_start(
        self,
        service: ServiceLlmEndpointHealth,
    ) -> None:
        """stop() should close the HTTP client even if start() was never called.

        This covers the one-shot usage pattern where probe_all() lazily
        creates an HTTP client but the caller never invokes start()/stop().
        """
        mock_response = httpx.Response(200, request=httpx.Request("GET", "http://test"))

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            await service.probe_all()

        # Whitebox: verify the lazily-created HTTP client exists and is
        # cleaned up by stop().  There is no public accessor for the HTTP
        # client; the test validates an important resource-leak invariant
        # that cannot be observed through the public API alone.
        assert service._http_client is not None

        # stop() should close it even though start() was never called
        await service.stop()
        assert service._http_client is None

    @pytest.mark.asyncio
    async def test_async_context_manager(
        self,
        config: ModelLlmEndpointHealthConfig,
    ) -> None:
        """Service should support async context manager for one-shot usage."""
        mock_response = httpx.Response(200, request=httpx.Request("GET", "http://test"))

        async with ServiceLlmEndpointHealth(config=config) as svc:
            with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
                status_map = await svc.probe_all()
            assert len(status_map) == 2

        # Whitebox: verify the context manager closes the internal HTTP
        # client on exit (resource-leak prevention, no public accessor).
        assert svc._http_client is None

    @pytest.mark.asyncio
    async def test_probe_loop_continues_on_error(
        self,
    ) -> None:
        """The background loop should survive errors in probe_all."""
        cfg = ModelLlmEndpointHealthConfig(
            endpoints={"test": "http://localhost:9999"},
            probe_interval_seconds=1.0,
        )
        svc = ServiceLlmEndpointHealth(config=cfg)

        probe_count = 0

        async def counting_probe() -> dict[str, ModelLlmEndpointStatus]:
            nonlocal probe_count
            probe_count += 1
            if probe_count == 1:
                raise RuntimeError("Simulated error")
            return {}

        # Save a reference to real asyncio.sleep before patching
        real_sleep = asyncio.sleep

        async def fast_sleep(seconds: float) -> None:
            await real_sleep(0.01)

        with (
            patch.object(svc, "probe_all", side_effect=counting_probe),
            patch(
                "omnibase_infra.services.service_llm_endpoint_health.asyncio.sleep",
                side_effect=fast_sleep,
            ),
        ):
            await svc.start()
            # Let a few probe cycles run
            await real_sleep(0.15)
            await svc.stop()

        # Should have continued after the first error
        assert probe_count >= 2


# =============================================================================
# TestModelLlmEndpointHealthEvent
# =============================================================================


class TestModelLlmEndpointHealthEvent:
    """Validate the health event payload model."""

    def test_event_construction(self) -> None:
        """Event should serialize endpoint statuses."""
        from uuid import uuid4

        now = datetime.now(UTC)
        statuses = (
            ModelLlmEndpointStatus(
                url="http://localhost:8000",
                name="test-a",
                available=True,
                last_check=now,
                latency_ms=10.0,
            ),
            ModelLlmEndpointStatus(
                url="http://localhost:8100",
                name="test-b",
                available=False,
                last_check=now,
                latency_ms=-1.0,
                error="Down",
            ),
        )
        event = ModelLlmEndpointHealthEvent(
            timestamp=now,
            endpoints=statuses,
            correlation_id=uuid4(),
        )
        assert len(event.endpoints) == 2
        assert event.endpoints[0].available is True
        assert event.endpoints[1].available is False

    def test_event_frozen(self) -> None:
        """Event model should be immutable."""
        from uuid import uuid4

        now = datetime.now(UTC)
        event = ModelLlmEndpointHealthEvent(
            timestamp=now,
            endpoints=(),
            correlation_id=uuid4(),
        )
        with pytest.raises(ValidationError):
            event.timestamp = now  # type: ignore[misc]


# =============================================================================
# TestTopicConstant
# =============================================================================


class TestTopicConstant:
    """Validate the topic constant."""

    def test_topic_follows_onex_convention(self) -> None:
        """Topic should follow onex.evt.{domain}.{name}.v1 pattern."""
        assert TOPIC_LLM_ENDPOINT_HEALTH.startswith("onex.evt.")
        assert TOPIC_LLM_ENDPOINT_HEALTH.endswith(".v1")
        assert "llm-endpoint-health" in TOPIC_LLM_ENDPOINT_HEALTH
