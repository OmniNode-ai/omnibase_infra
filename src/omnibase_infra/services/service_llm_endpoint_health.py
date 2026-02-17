# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM Endpoint Health Checker Service.

Probes configured local LLM endpoints at a configurable interval and maintains
an in-memory status map with availability, latency, and last-check timestamps.
Each probe cycle optionally emits a health event to Kafka for downstream
consumers (dashboards, alerting, orchestrators).

The service applies the ``MixinAsyncCircuitBreaker`` pattern per endpoint so
that a persistently-down endpoint is quickly circuit-broken rather than
consuming probe resources on every tick.

Architecture:
    - One circuit breaker **per endpoint** (independent failure tracking)
    - Probes hit ``GET /health`` first; if that returns non-2xx, falls back
      to ``GET /v1/models`` (vLLM-style discovery)
    - Results are stored in a dict keyed by endpoint name
    - An optional ``ProtocolEventBusLike`` dependency enables Kafka emission

Topic:
    ``onex.evt.omnibase-infra.llm-endpoint-health.v1``

Related:
    - OMN-2249: SLO profiling baselines that inform health thresholds
    - OMN-2250: CIDR allowlist and HMAC signing for LLM HTTP transport
    - MixinAsyncCircuitBreaker: Circuit breaker pattern

.. versionadded:: 0.9.0
    Part of OMN-2255 LLM endpoint health checker.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import httpx

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins.mixin_async_circuit_breaker import MixinAsyncCircuitBreaker
from omnibase_infra.models.health.model_llm_endpoint_health_config import (
    ModelLlmEndpointHealthConfig,
)
from omnibase_infra.models.health.model_llm_endpoint_health_event import (
    ModelLlmEndpointHealthEvent,
)
from omnibase_infra.models.health.model_llm_endpoint_status import (
    ModelLlmEndpointStatus,
)
from omnibase_infra.utils.correlation import generate_correlation_id
from omnibase_infra.utils.util_error_sanitization import (
    sanitize_error_message,
    sanitize_url,
)

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias and constants
# ---------------------------------------------------------------------------
CircuitState = Literal["closed", "open", "half_open"]
"""Valid circuit breaker states for endpoint status."""

_VALID_CIRCUIT_STATES: frozenset[str] = frozenset({"closed", "open", "half_open"})

TOPIC_LLM_ENDPOINT_HEALTH: str = "onex.evt.omnibase-infra.llm-endpoint-health.v1"
"""Canonical topic for LLM endpoint health events."""


def _parse_circuit_state(
    cb_state: dict[str, JsonType],
    default: CircuitState,
) -> CircuitState:
    """Extract and validate the circuit breaker state from introspection dict.

    Args:
        cb_state: Dict returned by ``EndpointCircuitBreaker.get_state()``.
        default: Fallback value if the state key is missing or invalid.

    Returns:
        A validated ``CircuitState`` literal value.
    """
    raw = str(cb_state.get("state", default))
    if raw in _VALID_CIRCUIT_STATES:
        return raw  # type: ignore[return-value]
    return default


# ---------------------------------------------------------------------------
# Per-endpoint circuit breaker wrapper
# ---------------------------------------------------------------------------
class EndpointCircuitBreaker(MixinAsyncCircuitBreaker):
    """Thin wrapper that gives each endpoint its own circuit breaker state.

    ``MixinAsyncCircuitBreaker`` stores state on ``self``, so we need one
    instance per endpoint to isolate failure counts.

    This class exposes public wrapper methods around the private
    ``MixinAsyncCircuitBreaker`` API so that external consumers do not
    need to reach into private attributes.
    """

    def __init__(
        self,
        endpoint_name: str,
        threshold: int,
        reset_timeout: float,
    ) -> None:
        """Create a circuit breaker for a single LLM endpoint.

        Args:
            endpoint_name: Logical name used in the service name tag
                (e.g. ``"coder-14b"`` becomes ``llm-endpoint.coder-14b``).
            threshold: Consecutive failures before the circuit opens.
            reset_timeout: Seconds before an open circuit transitions to
                half-open.
        """
        self._init_circuit_breaker(
            threshold=threshold,
            reset_timeout=reset_timeout,
            service_name=f"llm-endpoint.{endpoint_name}",
            transport_type=EnumInfraTransportType.HTTP,
            half_open_successes=1,
        )

    # -- Public facade over MixinAsyncCircuitBreaker internals ---------------

    @property
    def lock(self) -> asyncio.Lock:
        """Return the circuit breaker lock for coroutine-safe access."""
        return self._circuit_breaker_lock

    async def check(self, operation: str, correlation_id: UUID) -> None:
        """Check whether the circuit breaker allows an operation.

        Must be called while holding :pyattr:`lock`.

        Args:
            operation: Operation name for error context.
            correlation_id: Correlation ID for distributed tracing.

        Raises:
            InfraUnavailableError: If the circuit is open.
        """
        await self._check_circuit_breaker(
            operation=operation,
            correlation_id=correlation_id,
        )

    async def record_failure(self, operation: str, correlation_id: UUID) -> None:
        """Record a failure and potentially open the circuit.

        Must be called while holding :pyattr:`lock`.

        Args:
            operation: Operation name for logging context.
            correlation_id: Correlation ID for distributed tracing.
        """
        await self._record_circuit_failure(
            operation=operation,
            correlation_id=correlation_id,
        )

    async def record_success(self) -> None:
        """Record a success and potentially close the circuit.

        Must be called while holding :pyattr:`lock`.
        """
        await self._reset_circuit_breaker()

    def get_state(self) -> dict[str, JsonType]:
        """Return the current circuit breaker state for introspection.

        Does **not** require holding the lock (read-only snapshot).

        Returns:
            Dict with keys ``initialized``, ``state``, ``failures``,
            ``threshold``, etc.
        """
        return self._get_circuit_breaker_state()

    @property
    def is_open(self) -> bool:
        """Return ``True`` if the circuit breaker is currently open."""
        return self._circuit_breaker_open


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class ServiceLlmEndpointHealth:
    """Probes local LLM endpoints and tracks availability.

    Usage::

        config = ModelLlmEndpointHealthConfig(
            endpoints={
                "coder-14b": "http://192.168.86.201:8000",
                "qwen-72b":  "http://192.168.86.200:8100",
            },
            probe_interval_seconds=30.0,
        )
        svc = ServiceLlmEndpointHealth(config=config, event_bus=bus)
        await svc.start()       # launches background probe loop
        ...
        statuses = svc.get_status()  # read current status map
        await svc.stop()        # cancels background loop

    The service can also be used without ``start``/``stop`` by calling
    ``probe_all`` directly for one-shot health checks.

    For one-shot usage the service supports the async context manager
    protocol, which ensures the HTTP client is closed on exit::

        async with ServiceLlmEndpointHealth(config=config) as svc:
            status_map = await svc.probe_all()
    """

    def __init__(
        self,
        config: ModelLlmEndpointHealthConfig,
        event_bus: ProtocolEventBusLike | None = None,
    ) -> None:
        """Initialize the health checker.

        Args:
            config: Endpoint configuration and probe settings.
            event_bus: Optional event bus for emitting health events.
                If ``None``, events are not emitted (probe-only mode).
        """
        self._config = config
        self._event_bus = event_bus

        # In-memory status map: name -> latest status
        self._status_map: dict[str, ModelLlmEndpointStatus] = {}

        # Per-endpoint circuit breakers
        self._circuit_breakers: dict[str, EndpointCircuitBreaker] = {}
        for name in config.endpoints:
            self._circuit_breakers[name] = EndpointCircuitBreaker(
                endpoint_name=name,
                threshold=config.circuit_breaker_threshold,
                reset_timeout=config.circuit_breaker_reset_timeout,
            )

        # Shared HTTP client (created lazily, closed on stop)
        self._http_client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

        # Background task handle
        self._probe_task: asyncio.Task[None] | None = None
        self._running = False

    # -- Async context manager ----------------------------------------------

    async def __aenter__(self) -> ServiceLlmEndpointHealth:
        """Enter the async context manager.

        Returns ``self`` without starting the background loop.  Use
        ``start()`` explicitly if you want the probe loop.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the async context manager and release resources."""
        await self.stop()

    # -- Public API ---------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the background probe loop is active."""
        return self._running

    def get_status(self) -> dict[str, ModelLlmEndpointStatus]:
        """Return the current in-memory status map (name -> status).

        Returns:
            Shallow copy of the status map so callers cannot mutate
            internal state.
        """
        return dict(self._status_map)

    def get_endpoint_status(self, name: str) -> ModelLlmEndpointStatus | None:
        """Return the status for a single endpoint by logical name.

        Args:
            name: Logical endpoint name (e.g. ``"coder-14b"``).

        Returns:
            The latest status, or ``None`` if not yet probed.
        """
        return self._status_map.get(name)

    async def start(self) -> None:
        """Start the background probe loop.

        Idempotent -- calling ``start`` on a running service is a no-op.
        """
        if self._running:
            logger.debug("ServiceLlmEndpointHealth already running, skipping start")
            return

        self._running = True
        self._probe_task = asyncio.create_task(
            self._probe_loop(), name="llm-endpoint-health-probe"
        )
        logger.info(
            "ServiceLlmEndpointHealth started",
            extra={
                "endpoint_count": len(self._config.endpoints),
                "probe_interval_seconds": self._config.probe_interval_seconds,
            },
        )

    async def stop(self) -> None:
        """Stop the background probe loop and release resources.

        Safe to call even if ``start()`` was never called.  This ensures
        that the lazily-created HTTP client is closed in one-shot usage
        scenarios (i.e. calling ``probe_all`` directly without
        ``start``/``stop``).

        Idempotent -- calling ``stop`` multiple times is safe.
        """
        if self._running:
            self._running = False
            if self._probe_task is not None:
                self._probe_task.cancel()
                try:
                    await self._probe_task
                except asyncio.CancelledError:
                    pass
                self._probe_task = None

        # Always close the HTTP client if it was created (covers one-shot
        # usage where probe_all() lazily created the client without start()).
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("ServiceLlmEndpointHealth stopped")

    async def probe_all(self) -> dict[str, ModelLlmEndpointStatus]:
        """Run a single probe cycle across all configured endpoints.

        This is the core probe method. It can be called directly for
        one-shot health checks or is invoked repeatedly by the background
        loop.

        Returns:
            Updated status map after probing all endpoints.
        """
        correlation_id = generate_correlation_id()

        # Probe all endpoints concurrently to avoid worst-case
        # N * 2 * timeout sequential latency.
        probe_coros = [
            self._probe_endpoint(name, url, correlation_id)
            for name, url in self._config.endpoints.items()
        ]
        results: list[ModelLlmEndpointStatus] = list(await asyncio.gather(*probe_coros))

        for status in results:
            self._status_map[status.name] = status

        # Emit health event if event bus is available
        if self._event_bus is not None and results:
            await self._emit_health_event(
                results=tuple(results),
                correlation_id=correlation_id,
            )

        return dict(self._status_map)

    # -- Internal -----------------------------------------------------------

    async def _probe_loop(self) -> None:
        """Background loop that probes endpoints at the configured interval.

        Runs until ``_running`` is set to ``False`` by ``stop()``.  Handles
        ``CancelledError`` in two cases:

        - **Normal shutdown**: ``stop()`` sets ``_running = False`` then cancels
          the task.  The ``CancelledError`` is re-raised to exit cleanly.
        - **Spurious cancellation**: ``_running`` is still ``True``, so the
          error is logged and the loop continues on the next iteration.

        Unexpected exceptions are logged but do not terminate the loop.
        """
        while self._running:
            try:
                await self.probe_all()
            except asyncio.CancelledError:
                # In Python 3.12+ CancelledError is a BaseException and
                # escapes ``except Exception``.  Handle it explicitly so
                # that a normal stop() (which sets _running=False before
                # cancelling the task) exits cleanly, while a spurious
                # cancellation merely logs and retries.
                if not self._running:
                    raise
                logger.warning(
                    "Probe loop received spurious CancelledError, continuing"
                )
                continue
            except Exception:
                logger.exception("Unexpected error in probe loop")
            try:
                await asyncio.sleep(self._config.probe_interval_seconds)
            except asyncio.CancelledError:
                break

    async def _probe_endpoint(
        self,
        name: str,
        url: str,
        correlation_id: UUID,
    ) -> ModelLlmEndpointStatus:
        """Probe a single endpoint with circuit breaker protection.

        Tries ``GET /health`` first, then falls back to ``GET /v1/models``.

        Args:
            name: Logical endpoint name.
            url: Base URL (e.g. ``http://192.168.86.201:8000``).
            correlation_id: Correlation ID for tracing.

        Returns:
            A ``ModelLlmEndpointStatus`` snapshot.
        """
        cb = self._circuit_breakers[name]

        # Check circuit breaker
        try:
            async with cb.lock:
                await cb.check(
                    operation="probe_health",
                    correlation_id=correlation_id,
                )
        except InfraUnavailableError:
            cb_state = cb.get_state()
            return ModelLlmEndpointStatus(
                url=url,
                name=name,
                available=False,
                last_check=datetime.now(UTC),
                latency_ms=-1.0,
                error="Circuit breaker open",
                circuit_state=_parse_circuit_state(cb_state, "open"),
            )

        # Probe the endpoint
        start_ns = time.perf_counter_ns()
        try:
            available, error = await self._http_probe(url)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

            if available:
                # Record success with circuit breaker
                async with cb.lock:
                    await cb.record_success()
            else:
                # Record failure with circuit breaker
                async with cb.lock:
                    await cb.record_failure(
                        operation="probe_health",
                        correlation_id=correlation_id,
                    )

            cb_state = cb.get_state()
            now = datetime.now(UTC)
            return ModelLlmEndpointStatus(
                url=url,
                name=name,
                available=available,
                last_check=now,
                latency_ms=round(elapsed_ms, 2) if available else -1.0,
                error=error,
                circuit_state=_parse_circuit_state(cb_state, "closed"),
            )

        except Exception as exc:
            # Record failure with circuit breaker
            async with cb.lock:
                await cb.record_failure(
                    operation="probe_health",
                    correlation_id=correlation_id,
                )

            cb_state = cb.get_state()
            now = datetime.now(UTC)
            error_msg = sanitize_error_message(exc)
            logger.warning(
                "Probe failed for %s (%s): %s",
                name,
                sanitize_url(url),
                error_msg,
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelLlmEndpointStatus(
                url=url,
                name=name,
                available=False,
                last_check=now,
                latency_ms=-1.0,
                error=error_msg,
                circuit_state=_parse_circuit_state(cb_state, "closed"),
            )

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Return the shared HTTP client, creating it lazily if needed.

        This allows ``probe_all`` to work both in background-loop mode
        (where ``start``/``stop`` manage the lifecycle) and in one-shot
        mode (where ``probe_all`` is called directly).

        An ``asyncio.Lock`` protects the check-then-create logic so that
        concurrent coroutines from ``asyncio.gather`` in ``probe_all``
        cannot race and create duplicate ``httpx.AsyncClient`` instances.

        Returns:
            A shared ``httpx.AsyncClient`` configured with the probe
            timeout from the service config.
        """
        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self._config.probe_timeout_seconds),
                )
            return self._http_client

    async def _http_probe(self, base_url: str) -> tuple[bool, str]:
        """Perform the HTTP probe against an endpoint.

        Tries ``GET /health`` first.  If that returns a non-2xx status,
        falls back to ``GET /v1/models`` (vLLM model listing).  If both
        probes fail, the error message includes details from both attempts.

        Args:
            base_url: The endpoint base URL (no trailing slash).

        Returns:
            ``(available, error)`` where *available* is ``True`` on
            success and *error* is an empty string, or ``False`` with
            a human-readable error description.
        """
        client = await self._get_http_client()
        primary_error: str = ""

        # Primary probe: /health
        try:
            resp = await client.get(f"{base_url.rstrip('/')}/health")
            if 200 <= resp.status_code < 300:
                return True, ""
            primary_error = f"Primary /health: HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            primary_error = f"Primary /health: {type(exc).__name__}"

        # Fallback probe: /v1/models (vLLM-style)
        try:
            resp = await client.get(f"{base_url.rstrip('/')}/v1/models")
            if 200 <= resp.status_code < 300:
                return True, ""
            fallback_error = f"Fallback /v1/models: HTTP {resp.status_code}"
            return False, f"{primary_error}; {fallback_error}"
        except httpx.HTTPError as exc:
            fallback_error = f"Fallback /v1/models: {type(exc).__name__}"
            return False, f"{primary_error}; {fallback_error}"

    async def _emit_health_event(
        self,
        results: tuple[ModelLlmEndpointStatus, ...],
        correlation_id: UUID,
    ) -> None:
        """Emit an LLM endpoint health event to the event bus.

        Wraps the probe results in a ``ModelEventEnvelope`` and publishes
        to ``TOPIC_LLM_ENDPOINT_HEALTH``.  This is fire-and-forget:
        publication failures are logged but do not propagate, so the
        in-memory status map is still updated even if Kafka is down.

        Args:
            results: Tuple of endpoint status snapshots from the current
                probe cycle.
            correlation_id: Correlation ID for distributed tracing.
        """
        if self._event_bus is None:
            return

        event = ModelLlmEndpointHealthEvent(
            timestamp=datetime.now(UTC),
            endpoints=results,
            correlation_id=correlation_id,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=event,
            correlation_id=correlation_id,
            event_type="llm-endpoint-health",
            source_tool="ServiceLlmEndpointHealth",
        )

        try:
            await self._event_bus.publish_envelope(
                envelope=envelope,
                topic=TOPIC_LLM_ENDPOINT_HEALTH,
            )
        except Exception:
            # Health event emission failure should not crash the probe loop.
            # Log and continue -- the in-memory status map is still updated.
            logger.exception(
                "Failed to emit LLM endpoint health event",
                extra={"correlation_id": str(correlation_id)},
            )


__all__: list[str] = [
    "EndpointCircuitBreaker",
    "ModelLlmEndpointHealthConfig",
    "ModelLlmEndpointHealthEvent",
    "ModelLlmEndpointStatus",
    "ServiceLlmEndpointHealth",
    "TOPIC_LLM_ENDPOINT_HEALTH",
]
