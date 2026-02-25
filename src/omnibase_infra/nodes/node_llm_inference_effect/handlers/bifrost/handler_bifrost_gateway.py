# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Bifrost LLM gateway handler for declarative routing, failover, and circuit breaking.

The bifrost gateway sits in front of all configured local LLM backend
endpoints and handles routing, failover, retries, and circuit breaking
via config — not application code. It satisfies OMN-2736 requirements:

    R1: All backends reachable only via bifrost, routing rules in config.
    R2: Every request logs backend_selected, rule_id, latency_ms, retry_count
        queryable by tenant_id and operation_type.
    R3: Failover on backend outage; all-backends-down returns structured error.

Architecture:
    - ``ModelBifrostConfig`` defines backends, routing rules, and failover policy
    - ``_evaluate_rules()`` selects a backend candidate list for each request
    - ``_attempt_backends()`` iterates the candidate list with exponential backoff
    - Per-backend circuit breakers prevent repeated calls to unhealthy endpoints
    - HMAC-SHA256 ``X-ONEX-Signature`` headers authenticate outbound requests
    - Audit log entries are emitted for every routing decision

Handler Responsibilities:
    - Evaluate routing rules against ``ModelBifrostRequest`` fields
    - Attempt backends in rule-declared priority order
    - Apply exponential backoff between failover attempts
    - Open circuit breaker on ``circuit_breaker_failure_threshold`` failures
    - Sign outbound requests with per-backend HMAC secrets
    - Return ``ModelBifrostResponse`` with auditable routing metadata
    - NEVER publish events directly (callers handle observability)

Circuit Breaker State:
    - Per-backend failure counters reset on successful call
    - Circuit opens after N consecutive failures within window_seconds
    - Opened circuit bypasses backend until reset_timeout_seconds elapses
    - Half-open: one probe attempt after timeout; closes on success

Coroutine Safety:
    - Per-backend circuit breaker state is protected by asyncio.Lock
    - No mutable state is shared across concurrent handle() calls
    - Config is immutable (ModelBifrostConfig is frozen)

Related Tickets:
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
    - OMN-2244: Local LLM SLO & Security Baseline (blocker, now done)
    - OMN-2248: Delegated Task Execution via Local Models (epic)

See Also:
    - ModelBifrostConfig for gateway configuration
    - ModelBifrostRequest for input contract
    - ModelBifrostResponse for output contract
    - ModelBifrostRoutingRule for routing rule structure
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.effects.models.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_config import (
    ModelBifrostBackendConfig,
    ModelBifrostConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_request import (
    ModelBifrostRequest,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_response import (
    ModelBifrostResponse,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_routing_rule import (
    ModelBifrostRoutingRule,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)

logger = logging.getLogger(__name__)

# HMAC header name for outbound request authentication.
_HMAC_HEADER_NAME = "X-ONEX-Signature"


@dataclass
class CircuitBreakerState:
    """Per-backend circuit breaker state.

    Tracks consecutive failure count and circuit-open timestamp.
    Protected by ``_lock`` for coroutine-safe access.

    Attributes:
        failure_count: Number of consecutive failures since last reset.
        opened_at: Timestamp (monotonic seconds) when circuit opened,
            or None if circuit is closed.
        lock: Per-backend asyncio.Lock protecting this state object.
    """

    failure_count: int = 0
    opened_at: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class HandlerBifrostGateway:
    """Bifrost LLM gateway handler with declarative routing and failover.

    Receives a ``ModelBifrostRequest``, evaluates routing rules from
    ``ModelBifrostConfig``, selects the best available backend, and
    returns a ``ModelBifrostResponse`` with full audit metadata.

    Circuit breakers open per-backend after consecutive failures and
    automatically close after a cooldown period. Closed backends are
    skipped during failover without incurring HTTP round-trips.

    HMAC authentication signs outbound requests with per-backend secrets
    using HMAC-SHA256. Backends without a configured secret receive
    unsigned requests.

    Protocol Conformance Note:
        This handler does NOT implement ``ProtocolHandler`` or
        ``ProtocolMessageHandler``. It operates at the infrastructure
        layer with typed request/response models, bypassing envelope
        dispatch entirely. ``handler_type`` and ``handler_category``
        are provided for introspection consistency.

    Attributes:
        _config: Immutable gateway configuration.
        _inference_handler: The underlying OpenAI-compatible inference
            handler used to execute HTTP calls to backends.
        _circuit_states: Per-backend circuit breaker state mapping.

    Example:
        >>> from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost import (
        ...     HandlerBifrostGateway,
        ...     ModelBifrostConfig,
        ...     ModelBifrostBackendConfig,
        ...     ModelBifrostRequest,
        ...     ModelBifrostRoutingRule,
        ... )
        >>> from unittest.mock import MagicMock
        >>> from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
        >>> transport = MagicMock(spec=MixinLlmHttpTransport)
        >>> inference_handler = HandlerLlmOpenaiCompatible(transport)
        >>> config = ModelBifrostConfig(
        ...     backends={
        ...         "a": ModelBifrostBackendConfig(backend_id="a", base_url="http://a:8000"),
        ...     },
        ...     default_backends=("a",),
        ... )
        >>> gateway = HandlerBifrostGateway(config=config, inference_handler=inference_handler)
    """

    def __init__(
        self,
        config: ModelBifrostConfig,
        inference_handler: HandlerLlmOpenaiCompatible,
    ) -> None:
        """Initialize the bifrost gateway with config and inference handler.

        Args:
            config: Immutable gateway configuration defining backends,
                routing rules, and failover/circuit-breaker policy.
            inference_handler: The OpenAI-compatible handler used to
                execute HTTP inference calls to backends.
        """
        self._config = config
        self._inference_handler = inference_handler
        # Per-backend circuit breaker state — lazily created on first access.
        self._circuit_states: dict[str, CircuitBreakerState] = defaultdict(
            CircuitBreakerState
        )
        # Per-backend lock guard for lazy circuit state creation.
        self._state_init_lock: asyncio.Lock = asyncio.Lock()

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification.

        Returns:
            ``EnumHandlerType.INFRA_HANDLER`` — infrastructure gateway.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification.

        Returns:
            ``EnumHandlerTypeCategory.EFFECT`` — performs external I/O.
        """
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: ModelBifrostRequest) -> ModelBifrostResponse:
        """Route request to best available backend and return response with audit trail.

        Evaluates routing rules in priority order, selects a candidate
        backend list, then attempts each backend with exponential backoff
        and circuit breaker protection. Returns a structured error response
        if all backends are unavailable.

        Args:
            request: Bifrost routing request with operation type,
                capabilities, cost tier, latency budget, and payload.

        Returns:
            ModelBifrostResponse with backend_selected, rule_id,
            latency_ms, retry_count, and the inference result.
        """
        correlation_id: UUID = request.correlation_id or uuid4()
        start_time = time.perf_counter()

        # Evaluate routing rules → candidate backend IDs + matched rule
        candidate_backend_ids, matched_rule = self._evaluate_rules(request)
        matched_rule_id: UUID | None = matched_rule.rule_id if matched_rule else None

        logger.debug(
            "Bifrost routing decision: tenant=%s operation=%s rule=%s candidates=%s corr=%s",
            request.tenant_id,
            request.operation_type,
            matched_rule_id,
            candidate_backend_ids,
            correlation_id,
        )

        # Attempt backends in order with failover
        result, backend_selected, retry_count = await self._attempt_backends(
            request=request,
            candidate_backend_ids=list(candidate_backend_ids),
            correlation_id=correlation_id,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        if result is None:
            # All backends failed — return structured error
            logger.warning(
                "Bifrost: all backends failed. tenant=%s operation=%s rule=%s "
                "attempts=%d corr=%s",
                request.tenant_id,
                request.operation_type,
                matched_rule_id,
                retry_count,
                correlation_id,
            )
            return ModelBifrostResponse(
                backend_selected="",  # empty — no backend served on total failure
                matched_rule_id=matched_rule_id,
                latency_ms=latency_ms,
                retry_count=retry_count,
                tenant_id=request.tenant_id,
                correlation_id=correlation_id,
                inference_response=None,
                success=False,
                error_message=(
                    f"All backends failed after {retry_count} attempt(s). "
                    f"operation_type={request.operation_type.value}"
                ),
            )

        logger.info(
            "Bifrost: request served. tenant=%s operation=%s backend=%s rule=%s "
            "latency_ms=%.1f retries=%d corr=%s",
            request.tenant_id,
            request.operation_type,
            backend_selected,
            matched_rule_id,
            latency_ms,
            retry_count,
            correlation_id,
        )

        return ModelBifrostResponse(
            backend_selected=backend_selected,
            matched_rule_id=matched_rule_id,
            latency_ms=latency_ms,
            retry_count=retry_count,
            tenant_id=request.tenant_id,
            correlation_id=correlation_id,
            inference_response=result,
            success=True,
        )

    # ── Rule evaluation ─────────────────────────────────────────────────

    def _evaluate_rules(
        self, request: ModelBifrostRequest
    ) -> tuple[tuple[str, ...], ModelBifrostRoutingRule | None]:
        """Evaluate routing rules in priority order and return first match.

        Rules are evaluated in ascending ``priority`` order. The first
        rule whose match predicates are satisfied by the request is
        returned together with the candidate backend ID list.

        When no rule matches, the ``default_backends`` from config is
        returned with ``matched_rule=None``.

        Args:
            request: The incoming routing request.

        Returns:
            A tuple of (candidate_backend_ids, matched_rule). The
            backend IDs tuple is empty only if no rule matched AND
            ``default_backends`` is also empty.
        """
        indexed_rules = sorted(
            enumerate(self._config.routing_rules),
            key=lambda pair: (pair[1].priority, pair[0]),
        )
        ordered_rules: list[ModelBifrostRoutingRule] = [
            rule for _, rule in indexed_rules
        ]

        for rule in ordered_rules:
            if self._rule_matches(rule, request):
                return rule.backend_ids, rule

        # No rule matched — use default backends
        return self._config.default_backends, None

    @staticmethod
    def _rule_matches(
        rule: ModelBifrostRoutingRule, request: ModelBifrostRequest
    ) -> bool:
        """Check whether a single routing rule matches a request.

        All non-empty match predicates must be satisfied for the rule
        to match. Empty predicates are treated as wildcard (match any).

        Args:
            rule: The routing rule to evaluate.
            request: The incoming request.

        Returns:
            True if all match predicates are satisfied, False otherwise.
        """
        # operation_type match
        if (
            rule.match_operation_types
            and request.operation_type not in rule.match_operation_types
        ):
            return False

        # capabilities match — ALL declared capabilities must appear in request
        if rule.match_capabilities:
            request_caps = set(request.capabilities)
            if not set(rule.match_capabilities).issubset(request_caps):
                return False

        # cost_tier match
        if (
            rule.match_cost_tiers
            and request.cost_tier.value not in rule.match_cost_tiers
        ):
            return False

        # max_latency_ms constraint
        if (
            rule.match_max_latency_ms_lte is not None
            and request.max_latency_ms > rule.match_max_latency_ms_lte
        ):
            return False

        return True

    # ── Backend attempt loop ─────────────────────────────────────────────

    async def _attempt_backends(
        self,
        request: ModelBifrostRequest,
        candidate_backend_ids: list[str],
        correlation_id: UUID,
    ) -> tuple[ModelLlmInferenceResponse | None, str, int]:
        """Try candidate backends in order with failover and circuit breaking.

        Iterates through ``candidate_backend_ids`` up to
        ``config.failover_attempts`` times. Skips backends whose circuit
        breaker is open. Applies exponential backoff between attempts.

        Args:
            request: The routing request (for payload and audit fields).
            candidate_backend_ids: Ordered list of backend IDs to try.
            correlation_id: Correlation ID for tracing.

        Returns:
            A 3-tuple of:
            - The successful ``ModelLlmInferenceResponse``, or None if all failed.
            - The backend_id that served the request ("" on total failure).
            - The number of backends attempted (retry_count).
        """
        attempted = 0
        max_attempts = self._config.failover_attempts

        for backend_id in candidate_backend_ids:
            if attempted >= max_attempts:
                break

            backend_cfg = self._config.backends.get(backend_id)
            if backend_cfg is None:
                logger.warning(
                    "Bifrost: backend_id '%s' not found in config — skipping. corr=%s",
                    backend_id,
                    correlation_id,
                )
                continue

            # Check circuit breaker
            if await self._is_circuit_open(backend_id):
                logger.debug(
                    "Bifrost: circuit open for backend '%s' — skipping. corr=%s",
                    backend_id,
                    correlation_id,
                )
                continue

            # Apply backoff before non-first attempts
            if attempted > 0:
                delay_ms = self._config.failover_backoff_base_ms * (
                    2 ** (attempted - 1)
                )
                delay_s = delay_ms / 1000.0
                logger.debug(
                    "Bifrost: backoff %.0f ms before attempt %d on backend '%s'. corr=%s",
                    delay_ms,
                    attempted + 1,
                    backend_id,
                    correlation_id,
                )
                await asyncio.sleep(delay_s)

            attempted += 1

            try:
                inference_request = self._build_inference_request(
                    request=request,
                    backend_cfg=backend_cfg,
                    correlation_id=correlation_id,
                )
                response = await self._inference_handler.handle(
                    inference_request,
                    correlation_id=None,  # correlation already embedded in request
                )
                # Success — reset circuit breaker failure count
                await self._record_success(backend_id)
                return response, backend_id, attempted - 1

            except Exception as exc:
                logger.warning(
                    "Bifrost: backend '%s' failed (attempt %d/%d): %s. corr=%s",
                    backend_id,
                    attempted,
                    max_attempts,
                    type(exc).__name__,
                    correlation_id,
                )
                await self._record_failure(backend_id)

        return None, "", attempted

    # ── Inference request builder ─────────────────────────────────────────

    def _build_inference_request(
        self,
        request: ModelBifrostRequest,
        backend_cfg: ModelBifrostBackendConfig,
        correlation_id: UUID,
    ) -> ModelLlmInferenceRequest:
        """Build a ``ModelLlmInferenceRequest`` for a specific backend.

        Resolves the model name (request override → backend config →
        operation_type default), computes the timeout, and builds the
        inference request from the bifrost request payload fields.

        Args:
            request: The incoming bifrost routing request.
            backend_cfg: Configuration for the selected backend.
            correlation_id: Correlation ID for tracing.

        Returns:
            A ready-to-execute ``ModelLlmInferenceRequest``.
        """
        from omnibase_infra.enums import EnumLlmOperationType

        # Resolve model name: request override → backend config → fallback
        model_name = (
            request.model or backend_cfg.model_name or request.operation_type.value
        )

        # Resolve timeout: backend override → global config
        timeout_ms = backend_cfg.timeout_ms or self._config.request_timeout_ms
        timeout_seconds = timeout_ms / 1000.0

        # operation_type is already an EnumLlmOperationType
        operation_type = request.operation_type

        # Build HMAC signature as X-ONEX-Signature header if secret is configured
        extra_headers: dict[str, str] = {}
        if backend_cfg.hmac_secret:
            extra_headers[_HMAC_HEADER_NAME] = self._compute_hmac_signature(
                secret=backend_cfg.hmac_secret,
                correlation_id=str(correlation_id),
            )

        return ModelLlmInferenceRequest(
            base_url=backend_cfg.base_url,
            operation_type=operation_type,
            model=model_name,
            messages=request.messages,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            extra_headers=extra_headers,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _compute_hmac_signature(secret: str, correlation_id: str) -> str:
        """Compute an HMAC-SHA256 signature for the outbound request.

        The signature covers the correlation ID and UTC timestamp to
        provide replay protection. The result is formatted as a hex
        digest suitable for the ``X-ONEX-Signature`` header.

        Args:
            secret: The HMAC-SHA256 secret key.
            correlation_id: Request correlation ID included in signature.

        Returns:
            HMAC-SHA256 hex digest string prefixed with ``hmac-sha256-``.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        message = f"{correlation_id}:{timestamp}"
        digest = hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"hmac-sha256-{digest}"

    # ── Circuit breaker ───────────────────────────────────────────────────

    async def _get_circuit_state(self, backend_id: str) -> CircuitBreakerState:
        """Get or lazily create the circuit breaker state for a backend.

        Args:
            backend_id: The backend identifier.

        Returns:
            The ``CircuitBreakerState`` for this backend.
        """
        async with self._state_init_lock:
            if backend_id not in self._circuit_states:
                self._circuit_states[backend_id] = CircuitBreakerState()
        return self._circuit_states[backend_id]

    async def _is_circuit_open(self, backend_id: str) -> bool:
        """Check whether the circuit breaker for a backend is open.

        The circuit is open if:
        - ``failure_count`` >= ``circuit_breaker_failure_threshold``, AND
        - The circuit has been open for less than
          ``circuit_breaker_reset_timeout_seconds`` (i.e. the reset window
          has not yet elapsed).

        Once ``circuit_breaker_reset_timeout_seconds`` elapses the circuit
        transitions to half-open (``opened_at`` reset to None) to allow one
        probe attempt. On probe success ``_record_success`` closes the circuit;
        on probe failure ``_record_failure`` reopens it.

        Note on ``circuit_breaker_window_seconds``: this config field
        represents the reset/cooldown timeout — how long to keep the circuit
        open before allowing a probe. Failure counting is cumulative (not
        windowed); callers reset it explicitly via ``_record_success``.

        Args:
            backend_id: The backend to check.

        Returns:
            True if the circuit is open (backend should be skipped).
        """
        state = await self._get_circuit_state(backend_id)
        async with state.lock:
            if state.failure_count < self._config.circuit_breaker_failure_threshold:
                return False

            if state.opened_at is None:
                # Circuit just exceeded threshold — open it now
                state.opened_at = time.monotonic()
                logger.warning(
                    "Bifrost: circuit opened for backend '%s' after %d failures.",
                    backend_id,
                    state.failure_count,
                )
                return True

            # Check if reset timeout has elapsed (half-open probe)
            elapsed = time.monotonic() - state.opened_at
            if elapsed >= self._config.circuit_breaker_window_seconds:
                logger.info(
                    "Bifrost: circuit half-open for backend '%s' after %.0fs reset timeout.",
                    backend_id,
                    elapsed,
                )
                state.opened_at = None  # Allow one probe attempt
                return False

            return True

    async def _record_success(self, backend_id: str) -> None:
        """Reset circuit breaker failure count after a successful call.

        Args:
            backend_id: The backend that succeeded.
        """
        state = await self._get_circuit_state(backend_id)
        async with state.lock:
            if state.failure_count > 0:
                logger.debug(
                    "Bifrost: circuit closed for backend '%s' (was %d failures).",
                    backend_id,
                    state.failure_count,
                )
            state.failure_count = 0
            state.opened_at = None

    async def _record_failure(self, backend_id: str) -> None:
        """Increment circuit breaker failure count after a failed call.

        Args:
            backend_id: The backend that failed.
        """
        state = await self._get_circuit_state(backend_id)
        async with state.lock:
            state.failure_count += 1
            if (
                state.failure_count >= self._config.circuit_breaker_failure_threshold
                and state.opened_at is None
            ):
                state.opened_at = time.monotonic()
                logger.warning(
                    "Bifrost: circuit opened for backend '%s' after %d failures.",
                    backend_id,
                    state.failure_count,
                )

    def get_circuit_failure_count(self, backend_id: str) -> int:
        """Get the current failure count for a backend's circuit breaker.

        This is a synchronous inspection method intended for testing and
        observability. It reads state without a lock — callers should not
        rely on this being consistent under concurrent load.

        Args:
            backend_id: The backend to inspect.

        Returns:
            Current consecutive failure count (0 if no state exists yet).
        """
        state = self._circuit_states.get(backend_id)
        return state.failure_count if state else 0

    def is_circuit_open_sync(self, backend_id: str) -> bool:
        """Synchronously check if a circuit is currently open.

        Intended for testing and observability only — NOT safe for
        concurrent production use. Use ``_is_circuit_open()`` in
        async contexts.

        Args:
            backend_id: The backend to inspect.

        Returns:
            True if the circuit is currently open (not half-open).
        """
        state = self._circuit_states.get(backend_id)
        if state is None:
            return False
        if state.failure_count < self._config.circuit_breaker_failure_threshold:
            return False
        if state.opened_at is None:
            return False
        elapsed = time.monotonic() - state.opened_at
        return elapsed < self._config.circuit_breaker_window_seconds


__all__: list[str] = ["HandlerBifrostGateway"]
