# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
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

Shadow Mode (OMN-5570):
    - When ``shadow_config.enabled`` is True, a learned routing policy runs
      in parallel with static rules via ``asyncio.create_task``
    - Shadow computation is async and does NOT affect the actual routing decision
    - Shadow decisions are collected via a callback (typically Kafka event emission)
    - Shadow latency is bounded by ``shadow_config.max_shadow_latency_ms``
    - Shadow mode adds < 5ms to request latency (fire-and-forget async)

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
    - Shadow tasks are fire-and-forget — exceptions are logged, never raised

Related Tickets:
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
    - OMN-2244: Local LLM SLO & Security Baseline (blocker, now done)
    - OMN-2248: Delegated Task Execution via Local Models (epic)
    - OMN-5570: Shadow Mode + Comparison Dashboard
    - OMN-5556: Learned Decision Optimization Platform (epic)

See Also:
    - ModelBifrostConfig for gateway configuration
    - ModelBifrostRequest for input contract
    - ModelBifrostResponse for output contract
    - ModelBifrostRoutingRule for routing rule structure
    - ModelBifrostShadowConfig for shadow mode configuration
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import random
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.llm.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config.bifrost_shadow import (
    ModelBifrostShadowConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config.model_shadow_decision_log import (
    ModelShadowDecisionLog,
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


class ProtocolShadowPolicy(Protocol):
    """Protocol for learned routing policies used in shadow mode.

    Implementations must provide an async ``recommend`` method that takes
    a bifrost request and the list of available backend IDs, and returns
    the recommended backend ID along with a confidence score and the
    full action probability distribution.

    The protocol is intentionally minimal — policy implementations may
    use ONNX, PyTorch, scikit-learn, or any other framework internally.
    """

    async def recommend(
        self,
        request: ModelBifrostRequest,
        available_backends: tuple[str, ...],
    ) -> tuple[str, float, dict[str, float]]:
        """Recommend a backend for the given request.

        Args:
            request: The incoming bifrost routing request.
            available_backends: Tuple of backend IDs currently configured
                in the gateway (not filtered by circuit breaker state).

        Returns:
            A 3-tuple of:
            - recommended_backend_id: The backend the policy recommends.
            - confidence: Confidence score in [0.0, 1.0].
            - action_distribution: Full probability distribution over
              backends (keys are backend IDs, values sum to ~1.0).
        """
        ...  # pragma: no cover


# Type alias for shadow decision log callbacks.
ShadowDecisionCallback = Callable[[ModelShadowDecisionLog], Awaitable[None]]


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


class HandlerBifrostGateway:  # ONEX_EXCLUDE: method_count - gateway is a single cohesive unit managing routing, failover, circuit-breaking, and shadow mode
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

    Shadow Mode (OMN-5570):
        When ``shadow_config.enabled`` is True and a ``shadow_policy`` is
        provided, the gateway runs the learned policy in parallel with
        static routing. Shadow computation is async (fire-and-forget via
        ``asyncio.create_task``) and adds < 5ms latency. Shadow decisions
        are emitted via the ``shadow_decision_callback`` for persistence
        and dashboard comparison.

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
        _shadow_config: Shadow mode configuration (None = disabled).
        _shadow_policy: Learned routing policy (None = no shadow).
        _shadow_decision_callback: Async callback for shadow decision logs.

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
        shadow_config: ModelBifrostShadowConfig | None = None,
        shadow_policy: ProtocolShadowPolicy | None = None,
        shadow_decision_callback: ShadowDecisionCallback | None = None,
    ) -> None:
        """Initialize the bifrost gateway with config and inference handler.

        Args:
            config: Immutable gateway configuration defining backends,
                routing rules, and failover/circuit-breaker policy.
            inference_handler: The OpenAI-compatible handler used to
                execute HTTP inference calls to backends.
            shadow_config: Shadow mode configuration. When None or
                ``enabled=False``, shadow mode is inactive.
            shadow_policy: The learned routing policy to run in shadow
                mode. Required when shadow_config.enabled is True.
            shadow_decision_callback: Async callback invoked with each
                shadow decision log entry. Typically publishes to Kafka.
                Required when shadow_config.enabled is True.
        """
        self._config = config
        self._inference_handler = inference_handler
        # Per-backend circuit breaker state — lazily created on first access.
        self._circuit_states: dict[str, CircuitBreakerState] = defaultdict(
            CircuitBreakerState
        )
        # Per-backend lock guard for lazy circuit state creation.
        self._state_init_lock: asyncio.Lock = asyncio.Lock()

        # Shadow mode (OMN-5570)
        self._shadow_config = shadow_config or ModelBifrostShadowConfig()
        self._shadow_policy = shadow_policy
        self._shadow_decision_callback = shadow_decision_callback

        # Validate shadow mode configuration
        if self._shadow_config.enabled:
            if self._shadow_policy is None:
                logger.warning(
                    "Bifrost: shadow mode enabled but no shadow_policy provided. "
                    "Shadow decisions will not be computed."
                )
            if self._shadow_decision_callback is None:
                logger.warning(
                    "Bifrost: shadow mode enabled but no shadow_decision_callback "
                    "provided. Shadow decisions will be computed but not persisted."
                )

    @property
    def shadow_config(self) -> ModelBifrostShadowConfig:
        """Return the current shadow mode configuration.

        Returns:
            The ``ModelBifrostShadowConfig`` for this gateway instance.
        """
        return self._shadow_config

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

        When shadow mode is enabled, the learned policy is evaluated in
        parallel via ``asyncio.create_task``. The shadow computation does
        NOT affect the actual routing decision and adds < 5ms latency.

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

        # Fire shadow policy evaluation (async, fire-and-forget).
        # We store the task reference to prevent GC collection and satisfy RUF006,
        # but intentionally do not await it — shadow must not delay the response.
        # Inline check: shadow enabled + policy present + sample rate pass
        _run_shadow = (
            self._shadow_config.enabled
            and self._shadow_policy is not None
            and (
                self._shadow_config.log_sample_rate >= 1.0
                or random.random() <= self._shadow_config.log_sample_rate
            )
        )
        if _run_shadow:
            _shadow_task = asyncio.create_task(
                self._evaluate_shadow_policy(
                    request=request,
                    static_backend_selected=backend_selected or "",
                    static_rule_id=matched_rule_id,
                    correlation_id=correlation_id,
                )
            )
            _shadow_task.add_done_callback(
                lambda t: t.exception() if not t.cancelled() else None
            )

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

    # ── Shadow mode ──────────────────────────────────────────────────────

    async def _evaluate_shadow_policy(
        self,
        request: ModelBifrostRequest,
        static_backend_selected: str,
        static_rule_id: UUID | None,
        correlation_id: UUID,
    ) -> None:
        """Evaluate the shadow policy and log the comparison result.

        Runs the learned policy with a timeout to ensure < 5ms added
        latency. On timeout or error, logs a warning and returns without
        affecting the static routing decision.

        This method is designed to be called via ``asyncio.create_task``
        (fire-and-forget). Exceptions are caught and logged, never raised.

        Args:
            request: The original bifrost routing request.
            static_backend_selected: Backend selected by static rules.
            static_rule_id: UUID of the matched static rule (or None).
            correlation_id: Request correlation ID for tracing.
        """
        shadow_start = time.perf_counter()
        try:
            assert self._shadow_policy is not None  # guaranteed by _should_run_shadow

            available_backends = tuple(self._config.backends.keys())
            timeout_s = self._shadow_config.max_shadow_latency_ms / 1000.0

            # Run shadow policy with timeout
            recommended, confidence, distribution = await asyncio.wait_for(
                self._shadow_policy.recommend(
                    request=request,
                    available_backends=available_backends,
                ),
                timeout=timeout_s,
            )

            shadow_latency_ms = (time.perf_counter() - shadow_start) * 1000
            agreed = recommended == static_backend_selected

            log_entry = ModelShadowDecisionLog(
                correlation_id=correlation_id,
                static_backend_selected=static_backend_selected,
                shadow_backend_recommended=recommended,
                agreed=agreed,
                static_rule_id=static_rule_id,
                request_operation_type=request.operation_type.value,
                request_cost_tier=request.cost_tier.value,
                request_max_latency_ms=request.max_latency_ms,
                shadow_confidence=confidence,
                shadow_latency_ms=shadow_latency_ms,
                policy_version=self._shadow_config.policy_version,
                shadow_action_distribution=distribution,
                tenant_id=request.tenant_id,
            )

            logger.debug(
                "Bifrost shadow: agreed=%s static=%s shadow=%s confidence=%.3f "
                "latency_ms=%.1f corr=%s",
                agreed,
                static_backend_selected,
                recommended,
                confidence,
                shadow_latency_ms,
                correlation_id,
            )

            # Emit the shadow decision log via callback
            if (
                self._shadow_decision_callback is not None
                and self._shadow_config.comparison_logging_enabled
            ):
                await self._shadow_decision_callback(log_entry)

        except TimeoutError:
            shadow_latency_ms = (time.perf_counter() - shadow_start) * 1000
            logger.warning(
                "Bifrost shadow: policy evaluation timed out after %.1f ms "
                "(limit=%.1f ms). corr=%s",
                shadow_latency_ms,
                self._shadow_config.max_shadow_latency_ms,
                correlation_id,
            )
        except Exception:  # noqa: BLE001 — shadow must never crash the gateway
            shadow_latency_ms = (time.perf_counter() - shadow_start) * 1000
            logger.warning(
                "Bifrost shadow: policy evaluation failed after %.1f ms. corr=%s",
                shadow_latency_ms,
                correlation_id,
                exc_info=True,
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

            except Exception as exc:  # noqa: BLE001 — boundary: logs warning and degrades
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
        # Resolve model name: request override → backend config → fallback
        model_name = (
            request.model or backend_cfg.model_name or request.operation_type.value
        )

        # Resolve timeout: backend override → global config
        timeout_ms = backend_cfg.timeout_ms or self._config.request_timeout_ms
        timeout_seconds = timeout_ms / 1000.0

        # Build HMAC signature as X-ONEX-Signature header if secret is configured
        extra_headers: dict[str, str] = {}
        if backend_cfg.hmac_secret:
            extra_headers[_HMAC_HEADER_NAME] = self._compute_hmac_signature(
                secret=backend_cfg.hmac_secret,
                correlation_id=str(correlation_id),
            )

        return ModelLlmInferenceRequest(
            base_url=backend_cfg.base_url,
            operation_type=request.operation_type,
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


__all__: list[str] = [
    "HandlerBifrostGateway",
    "ProtocolShadowPolicy",
    "ShadowDecisionCallback",
]
