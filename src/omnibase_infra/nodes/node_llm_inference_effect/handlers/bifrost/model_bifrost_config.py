# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Bifrost gateway configuration model.

Defines the complete configuration for the bifrost LLM gateway handler,
including backend endpoints, routing rules, failover policy, circuit
breaker settings, and HMAC authentication.

Related:
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_routing_rule import (
    ModelBifrostRoutingRule,
)


class ModelBifrostBackendConfig(BaseModel):
    """Configuration for a single LLM backend endpoint.

    Attributes:
        backend_id: Stable unique identifier for this backend.
            Referenced by ``ModelBifrostRoutingRule.backend_ids``.
        base_url: Base URL for the backend endpoint
            (e.g. ``"http://192.168.86.201:8000"``). The gateway
            appends ``/v1/chat/completions`` or ``/v1/completions``
            based on the operation type.
        hmac_secret: Optional HMAC-SHA256 secret key for request
            authentication. When set, the gateway adds an
            ``X-ONEX-Signature`` header to outbound requests.
            None disables HMAC authentication for this backend.
        model_name: Model identifier to send in the ``"model"`` field
            of outbound requests. If None, the model from the incoming
            ``ModelBifrostRequest`` is used as-is.
        timeout_ms: Per-request HTTP timeout for this backend in
            milliseconds. Overrides the global ``request_timeout_ms``
            when set.
        weight: Load balancing weight (reserved for future weighted
            round-robin). Currently unused; all backends are treated
            equally within a rule's priority ordering.

    Example:
        >>> backend = ModelBifrostBackendConfig(
        ...     backend_id="qwen-14b",
        ...     base_url="http://192.168.86.201:8000",
        ...     model_name="qwen2.5-coder-14b",
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    backend_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Stable unique identifier for audit logging and routing.",
    )
    base_url: str = Field(
        ...,
        min_length=1,
        description="Base URL for the backend endpoint.",
    )
    hmac_secret: str | None = Field(
        default=None,
        repr=False,
        description="HMAC-SHA256 secret for X-ONEX-Signature header (None = disabled).",
    )
    model_name: str | None = Field(
        default=None,
        description="Model identifier override sent in outbound requests.",
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=100,
        le=600_000,
        description="Per-backend timeout override in milliseconds.",
    )
    weight: int = Field(
        default=1,
        ge=1,
        description="Load balancing weight (reserved for future use).",
    )


class ModelBifrostConfig(BaseModel):
    """Complete configuration for the bifrost LLM gateway handler.

    The bifrost gateway reads this config at construction time and
    evaluates routing rules against each incoming ``ModelBifrostRequest``
    to select the target backend. Routing, failover, circuit breaking,
    and HMAC authentication are all defined here — not in application code.

    Attributes:
        backends: Mapping of backend_id to backend configuration.
            All backend_ids referenced in routing rules must have an
            entry here.
        routing_rules: Ordered list of routing rules evaluated in
            ascending ``priority`` order. The first matching rule is
            applied.
        default_backends: Fallback backend IDs when no routing rule
            matches. Attempted in order.
        failover_attempts: Maximum number of backends to try before
            returning a structured error. Applies per request.
        failover_backoff_base_ms: Base delay (ms) for exponential
            backoff between failover attempts.
            Delay for attempt N = ``backoff_base_ms * 2^(N-1)``.
        circuit_breaker_failure_threshold: Number of consecutive
            failures that open the circuit for a backend.
        circuit_breaker_window_seconds: Rolling time window (seconds)
            in which failures are counted for circuit breaker.
        request_timeout_ms: Default per-request HTTP timeout in
            milliseconds. Per-backend ``timeout_ms`` overrides this.
        health_check_interval_seconds: How often the gateway probes
            each backend's health endpoint in background (seconds).

    Example:
        >>> config = ModelBifrostConfig(
        ...     backends={
        ...         "qwen-14b": ModelBifrostBackendConfig(
        ...             backend_id="qwen-14b",
        ...             base_url="http://192.168.86.201:8000",
        ...         ),
        ...         "codestral": ModelBifrostBackendConfig(
        ...             backend_id="codestral",
        ...             base_url="http://192.168.86.202:8000",
        ...         ),
        ...     },
        ...     routing_rules=[
        ...         ModelBifrostRoutingRule(
        ...             rule_id="default-chat",
        ...             priority=100,
        ...             backend_ids=("qwen-14b", "codestral"),
        ...         ),
        ...     ],
        ...     default_backends=("codestral",),
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    backends: dict[str, ModelBifrostBackendConfig] = Field(
        ...,
        min_length=1,
        description="Mapping of backend_id to backend configuration.",
    )
    routing_rules: tuple[ModelBifrostRoutingRule, ...] = Field(
        default_factory=tuple,
        description="Routing rules evaluated in ascending priority order.",
    )
    default_backends: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Fallback backend IDs when no routing rule matches.",
    )
    failover_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum backends to try before returning structured error.",
    )
    failover_backoff_base_ms: int = Field(
        default=500,
        ge=0,
        le=10_000,
        description="Base exponential backoff delay in milliseconds between failover attempts.",
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Consecutive failures that open the circuit breaker for a backend.",
    )
    circuit_breaker_window_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Rolling time window for counting circuit breaker failures.",
    )
    request_timeout_ms: int = Field(
        default=10_000,
        ge=100,
        le=600_000,
        description="Default per-request HTTP timeout in milliseconds.",
    )
    health_check_interval_seconds: int = Field(
        default=10,
        ge=1,
        le=300,
        description="How often to probe backend health endpoints (seconds).",
    )


__all__: list[str] = ["ModelBifrostBackendConfig", "ModelBifrostConfig"]
