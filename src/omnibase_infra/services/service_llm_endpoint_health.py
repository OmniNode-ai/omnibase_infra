# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""LLM Endpoint Health Checker Service.

Probes configured local LLM endpoints at a configurable interval and maintains
an in-memory status map with availability, latency, and last-check timestamps.
Each probe cycle optionally emits a health event to Kafka for downstream
consumers (dashboards, alerting, orchestrators).

The service applies the ``MixinAsyncCircuitBreaker`` pattern per endpoint so
that a persistently-down endpoint is quickly circuit-broken rather than
consuming probe resources on every tick.

Architecture:
    - One circuit breaker **per probeable endpoint** (independent failure
      tracking)
    - Probes hit ``GET /health`` first; if that returns non-2xx, falls back
      to ``GET /v1/models`` (vLLM-style discovery)
    - Results are stored in a dict keyed by endpoint name
    - An optional ``ProtocolEventBusLike`` dependency enables Kafka emission

Authentication state is first-class (OMN-16900).  A credential problem is not
an outage, so it is never retried at outage cadence:

    - An endpoint whose declared auth secret is absent or unresolvable is
      classified ``SKIPPED_NO_AUTH`` **once, at construction**, and is never
      probed at all.
    - Sustained 401/403 becomes ``AUTH_FAILED`` after
      ``config.auth_failure_threshold`` consecutive occurrences and moves the
      endpoint onto an exponential backoff-to-idle schedule.
    - Transient failures (5xx, timeouts, connection errors) are untouched:
      full probe cadence, circuit breaker as before.

Topic:
    ``onex.evt.omnibase-infra.llm-endpoint-health.v1``

Related:
    - OMN-2249: SLO profiling baselines that inform health thresholds
    - OMN-2250: CIDR allowlist and HMAC signing for LLM HTTP transport
    - OMN-16900: auth-state classification and terminal-auth backoff
    - MixinAsyncCircuitBreaker: Circuit breaker pattern

.. versionadded:: 0.9.0
    Part of OMN-2255 LLM endpoint health checker.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import httpx

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.mixins.mixin_async_circuit_breaker import MixinAsyncCircuitBreaker
from omnibase_infra.models.health.enum_llm_endpoint_probe_state import (
    EnumLlmEndpointProbeState,
)
from omnibase_infra.models.health.model_llm_endpoint_health_config import (
    ModelLlmEndpointHealthConfig,
)
from omnibase_infra.models.health.model_llm_endpoint_health_event import (
    ModelLlmEndpointHealthEvent,
)
from omnibase_infra.models.health.model_llm_endpoint_status import (
    ModelLlmEndpointStatus,
)
from omnibase_infra.protocols import ProtocolTopicRegistry
from omnibase_infra.topics import topic_keys
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
_CHAT_COMPLETIONS_PATH_SUFFIX = "/chat/completions"
_AUTH_STATUS_CODES: frozenset[int] = frozenset({401, 403})
"""HTTP statuses that mean 'your credential is wrong', not 'the service is down'."""

_SKIPPED_NO_AUTH_ERROR = (
    "Auth secret absent or unresolvable for this endpoint; classified "
    "SKIPPED_NO_AUTH and not probed (OMN-16900)"
)


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
        # Why: Runtime validation guarantees the returned value matches the contract.
        return raw  # type: ignore[return-value]
    return default


def _probe_paths(endpoint_url: str) -> tuple[str, str]:
    """Return health and model-discovery probe URLs for an endpoint.

    Some contract-owned cloud endpoints are configured as complete chat
    completion URLs because the inference caller POSTs them verbatim. Health
    probing still needs to hit sibling discovery paths, not append paths below
    ``/chat/completions``.
    """
    parsed = urlsplit(endpoint_url)
    path = parsed.path.rstrip("/")
    if path.endswith(_CHAT_COMPLETIONS_PATH_SUFFIX):
        path = path[: -len(_CHAT_COMPLETIONS_PATH_SUFFIX)]
        health_path = f"{path}/health"
        models_path = f"{path}/models"
    else:
        base_path = path.rstrip("/")
        health_path = f"{base_path}/health"
        models_path = f"{base_path}/v1/models"
    return (
        urlunsplit((parsed.scheme, parsed.netloc, health_path, "", "")),
        urlunsplit((parsed.scheme, parsed.netloc, models_path, "", "")),
    )


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
                (e.g. ``"qwen3-coder-30b"`` becomes ``llm-endpoint.qwen3-coder-30b``).
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

        Returns a point-in-time snapshot.  Reads multiple mutable fields
        without holding the lock, so the returned dict may not reflect a
        single consistent state under concurrent access.

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

    Endpoint configuration must be sourced from the routing contract YAML via
    ``ModelLlmEndpointHealthConfig.from_model_registry()``.  Do **not** build
    the ``endpoints`` map by calling ``os.getenv`` directly — that pattern
    hard-codes stale model aliases and produces silent failures on empty env vars.

    Usage::

        import os
        from pathlib import Path
        from omnibase_infra.models.health.model_llm_endpoint_health_config import (
            ModelLlmEndpointHealthConfig,
        )

        registry = Path("docker/catalog/model_registry.yaml")
        config = ModelLlmEndpointHealthConfig.from_model_registry(
            registry_path=registry,
            env_resolver=os.getenv,
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

    ``get_status()`` may therefore contain entries that were never probed:
    ``from_model_registry`` routes endpoints with an unresolvable auth secret
    into ``config.unauthenticated_endpoints``, and those appear in the status
    map as ``SKIPPED_NO_AUTH`` from construction onward (OMN-16900).  Read
    ``probe_state`` rather than ``available`` when the distinction matters.
    """

    def __init__(
        self,
        config: ModelLlmEndpointHealthConfig,
        event_bus: ProtocolEventBusLike | None = None,
        topic_registry: ProtocolTopicRegistry | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the health checker.

        Args:
            config: Endpoint configuration and probe settings.
            event_bus: Optional event bus for emitting health events.
                If ``None``, events are not emitted (probe-only mode).
            topic_registry: Optional topic registry for resolving topic strings.
                If ``None``, uses ``ServiceTopicRegistry.from_defaults()``.
            monotonic: Monotonic clock used for auth-failure backoff windows.
                Injected so backoff behaviour is testable without sleeping.
        """
        if topic_registry is None:
            from omnibase_infra.topics.service_topic_registry import (
                ServiceTopicRegistry,
            )

            topic_registry = ServiceTopicRegistry.from_defaults()
        self._health_topic = topic_registry.resolve(topic_keys.LLM_ENDPOINT_HEALTH)
        self._config = config
        self._event_bus = event_bus
        self._monotonic = monotonic

        # In-memory status map: name -> latest status
        self._status_map: dict[str, ModelLlmEndpointStatus] = {}

        # Per-endpoint circuit breakers.  Only probeable endpoints get one —
        # an endpoint that is never probed has no failure sequence to track.
        self._circuit_breakers: dict[str, EndpointCircuitBreaker] = {}
        for name in config.endpoints:
            self._circuit_breakers[name] = EndpointCircuitBreaker(
                endpoint_name=name,
                threshold=config.circuit_breaker_threshold,
                reset_timeout=config.circuit_breaker_reset_timeout,
            )

        # OMN-16900: terminal-auth-failure tracking.  ``_auth_failures`` counts
        # consecutive 401/403 results; ``_next_probe_at`` holds the monotonic
        # deadline before which an auth-failed endpoint must not be re-probed.
        self._auth_failures: dict[str, int] = {}
        self._next_probe_at: dict[str, float] = {}

        # OMN-16900: endpoints whose auth secret is absent are classified once,
        # here, and never probed.  The classification is emitted on the first
        # probe cycle and then never again.
        self._pending_skipped_emit: list[ModelLlmEndpointStatus] = []
        classified_at = datetime.now(UTC)
        for name, url in config.unauthenticated_endpoints.items():
            status = ModelLlmEndpointStatus(
                url=sanitize_url(url),
                name=name,
                available=False,
                last_check=classified_at,
                latency_ms=-1.0,
                error=_SKIPPED_NO_AUTH_ERROR,
                circuit_state="closed",
                probe_state=EnumLlmEndpointProbeState.SKIPPED_NO_AUTH,
            )
            self._status_map[name] = status
            self._pending_skipped_emit.append(status)
            logger.warning(
                "LLM endpoint %s (%s) has no resolvable auth secret; classified "
                "SKIPPED_NO_AUTH and will not be probed",
                name,
                sanitize_url(url),
            )

        # Shared HTTP client (created lazily, closed on stop)
        self._http_client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

        # Background task handle
        self._probe_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def circuit_breaker_names(self) -> tuple[str, ...]:
        """Return the endpoint names that have a circuit breaker.

        Only probeable endpoints appear here; endpoints classified
        ``SKIPPED_NO_AUTH`` are never probed and so have no breaker.
        """
        return tuple(self._circuit_breakers)

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
            name: Logical endpoint name (e.g. ``"qwen3-coder-30b"``).

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

        Two classes of endpoint are deliberately **not** probed (OMN-16900):
        those classified ``SKIPPED_NO_AUTH`` at construction, and those in a
        terminal ``AUTH_FAILED`` backoff window whose deadline has not yet
        elapsed.  Both keep their existing status in the map rather than
        producing a fresh one, so a wrong credential costs zero requests.

        Returns:
            Updated status map after probing all due endpoints.
        """
        correlation_id = generate_correlation_id()
        cycle_start = datetime.now(UTC)
        now = self._monotonic()

        # Probe all due endpoints concurrently to avoid worst-case
        # N * 2 * timeout sequential latency.
        probe_coros = [
            self._probe_endpoint(name, url, correlation_id)
            for name, url in self._config.endpoints.items()
            if now >= self._next_probe_at.get(name, 0.0)
        ]
        results: list[ModelLlmEndpointStatus] = list(await asyncio.gather(*probe_coros))

        for status in results:
            self._status_map[status.name] = status

        # Newly-classified SKIPPED_NO_AUTH endpoints ride along on the first
        # cycle only.  Re-emitting them every 30s would just move the wasted
        # 401 traffic onto the event bus.
        emitted = [*results, *self._pending_skipped_emit]
        self._pending_skipped_emit.clear()

        # Emit health event if event bus is available
        if self._event_bus is not None and emitted:
            await self._emit_health_event(
                results=tuple(emitted),
                correlation_id=correlation_id,
                cycle_start=cycle_start,
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

        Auth failures (401/403) are routed away from the circuit breaker and
        into the dedicated auth-backoff path: the breaker exists to protect a
        *flaky* dependency, and a rejected credential is not flaky — it is
        deterministic and will not recover on its own (OMN-16900).

        Args:
            name: Logical endpoint name.
            url: Base URL (e.g. ``http://localhost:8000``).
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
                url=sanitize_url(url),
                name=name,
                available=False,
                last_check=datetime.now(UTC),
                latency_ms=-1.0,
                error="Circuit breaker open",
                circuit_state=_parse_circuit_state(cb_state, "open"),
                probe_state=EnumLlmEndpointProbeState.CIRCUIT_OPEN,
            )

        # Probe the endpoint
        start_ns = time.perf_counter_ns()
        try:
            probe_state, error = await self._http_probe(url)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

            if probe_state is EnumLlmEndpointProbeState.HEALTHY:
                self._clear_auth_failure(name)
                async with cb.lock:
                    await cb.record_success()
            elif probe_state is EnumLlmEndpointProbeState.AUTH_FAILED:
                # Terminal condition: back off instead of feeding the breaker.
                self._record_auth_failure(name, url, error)
            else:
                self._clear_auth_failure(name)
                async with cb.lock:
                    await cb.record_failure(
                        operation="probe_health",
                        correlation_id=correlation_id,
                    )

            available = probe_state is EnumLlmEndpointProbeState.HEALTHY
            cb_state = cb.get_state()
            now = datetime.now(UTC)
            return ModelLlmEndpointStatus(
                url=sanitize_url(url),
                name=name,
                available=available,
                last_check=now,
                latency_ms=round(elapsed_ms, 2) if available else -1.0,
                error=error,
                circuit_state=_parse_circuit_state(cb_state, "closed"),
                probe_state=probe_state,
            )

        except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
            # Record failure with circuit breaker
            self._clear_auth_failure(name)
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
                url=sanitize_url(url),
                name=name,
                available=False,
                last_check=now,
                latency_ms=-1.0,
                error=error_msg,
                circuit_state=_parse_circuit_state(cb_state, "closed"),
                probe_state=EnumLlmEndpointProbeState.UNAVAILABLE,
            )

    def _record_auth_failure(self, name: str, url: str, error: str) -> None:
        """Count a 401/403 and, past the threshold, arm the backoff window.

        The window doubles on every further auth failure, capped at
        ``config.auth_failure_backoff_max_seconds`` — backoff *to idle*, not to
        never, so a restored credential is picked up without a restart.

        Args:
            name: Logical endpoint name.
            url: Base URL, sanitized before it reaches any log line.
            error: Sanitized probe error description for the log record.
        """
        failures = self._auth_failures.get(name, 0) + 1
        self._auth_failures[name] = failures
        if failures < self._config.auth_failure_threshold:
            return

        exponent = failures - self._config.auth_failure_threshold + 1
        delay = min(
            self._config.probe_interval_seconds * (2**exponent),
            self._config.auth_failure_backoff_max_seconds,
        )
        self._next_probe_at[name] = self._monotonic() + delay
        logger.warning(
            "LLM endpoint %s (%s) auth-rejected %d consecutive probes (%s); "
            "classified AUTH_FAILED, next probe deferred %.0fs",
            name,
            sanitize_url(url),
            failures,
            error,
            delay,
        )

    def _clear_auth_failure(self, name: str) -> None:
        """Drop any auth-failure streak and backoff window for an endpoint.

        Called on a healthy probe and on transient failures alike: a 5xx is
        not evidence that the credential is good, but it is also not evidence
        that it is bad, so the auth streak must not accumulate across a
        genuine outage.

        Args:
            name: Logical endpoint name.
        """
        self._auth_failures.pop(name, None)
        self._next_probe_at.pop(name, None)

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

    async def _http_probe(self, base_url: str) -> tuple[EnumLlmEndpointProbeState, str]:
        """Perform the HTTP probe against an endpoint and classify the result.

        Tries ``GET /health`` first.  If that returns a non-2xx status,
        falls back to ``GET /v1/models`` (vLLM model listing).  If both
        probes fail, the error message includes details from both attempts.

        Classification (OMN-16900): when **both** probes come back 401/403 the
        endpoint is reachable but our credential is rejected, which is an
        ``AUTH_FAILED`` condition rather than an outage.  Previously this case
        returned ``available=True`` on the theory that an auth wall proves
        reachability — which reported fully unusable endpoints as healthy while
        re-probing them every 30 seconds forever.

        Args:
            base_url: The endpoint base URL (no trailing slash).

        Returns:
            ``(probe_state, error)`` where *error* is an empty string when the
            state is ``HEALTHY`` and a human-readable description otherwise.
        """
        client = await self._get_http_client()
        primary_error: str = ""
        primary_was_auth = False

        health_url, models_url = _probe_paths(base_url)

        # Primary probe: /health
        try:
            resp = await client.get(health_url)
            if 200 <= resp.status_code < 300:
                return EnumLlmEndpointProbeState.HEALTHY, ""
            primary_was_auth = resp.status_code in _AUTH_STATUS_CODES
            primary_error = f"Primary /health: HTTP {resp.status_code}"
        except Exception as exc:  # noqa: BLE001 — boundary: returns degraded response
            primary_error = f"Primary /health: {type(exc).__name__}"

        # Fallback probe: model discovery.
        try:
            resp = await client.get(models_url)
            if 200 <= resp.status_code < 300:
                return EnumLlmEndpointProbeState.HEALTHY, ""
            fallback_error = f"Fallback model discovery: HTTP {resp.status_code}"
            if resp.status_code in _AUTH_STATUS_CODES:
                # Reachable, but the credential is rejected. Terminal, not
                # transient — a 404 /health on an auth-gated cloud route is
                # normal, so the fallback verdict is the authoritative one.
                return (
                    EnumLlmEndpointProbeState.AUTH_FAILED,
                    f"{primary_error}; {fallback_error}",
                )
            if primary_was_auth:
                return (
                    EnumLlmEndpointProbeState.AUTH_FAILED,
                    f"{primary_error}; {fallback_error}",
                )
            return (
                EnumLlmEndpointProbeState.UNAVAILABLE,
                f"{primary_error}; {fallback_error}",
            )
        except httpx.HTTPError as exc:
            fallback_error = f"Fallback model discovery: {type(exc).__name__}"
            return (
                EnumLlmEndpointProbeState.UNAVAILABLE,
                f"{primary_error}; {fallback_error}",
            )

    async def _emit_health_event(
        self,
        results: tuple[ModelLlmEndpointStatus, ...],
        correlation_id: UUID,
        cycle_start: datetime,
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
            cycle_start: Timestamp captured at the beginning of the probe
                cycle, used as the event timestamp so it stays close to
                the individual endpoint probe timestamps.
        """
        if self._event_bus is None:
            return

        event = ModelLlmEndpointHealthEvent(
            timestamp=cycle_start,
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
                topic=self._health_topic,
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
]
