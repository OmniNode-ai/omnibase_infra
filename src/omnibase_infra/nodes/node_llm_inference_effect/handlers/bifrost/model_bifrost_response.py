# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Bifrost gateway response model.

Defines the output contract for the bifrost gateway handler, capturing
the selected backend, applied routing rule, per-call latency, retry
count, and the underlying LLM inference response.

Related:
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.effects.models.model_llm_inference_response import (
    ModelLlmInferenceResponse,
)


class ModelBifrostResponse(BaseModel):
    """Output contract for the bifrost LLM gateway handler.

    Every routing decision produces a ``ModelBifrostResponse``. The
    ``backend_selected``, ``rule_id``, ``latency_ms``, and
    ``retry_count`` fields form the auditable routing log required by
    OMN-2736 requirement R2.

    Attributes:
        backend_selected: ID of the backend that served the request.
            Corresponds to a ``backend_id`` in ``ModelBifrostConfig``.
        rule_id: ID of the routing rule that matched the request, or
            ``"default"`` when the fallback backend list was used.
            ``"none"`` indicates no backends were available.
        latency_ms: End-to-end gateway latency in milliseconds,
            measured from rule evaluation to response receipt.
        retry_count: Number of backends attempted before success.
            0 means the first backend succeeded; N means N backends
            were tried before a successful response.
        tenant_id: Caller identity copied from the request for
            audit log queryability.
        correlation_id: Correlation ID for distributed tracing.
        inference_response: The underlying LLM inference response
            from the selected backend. None when all backends failed
            and ``success=False``.
        success: Whether the request was served successfully.
        error_message: Structured error description when ``success``
            is False. Empty string on success.

    Example:
        >>> resp = ModelBifrostResponse(
        ...     backend_selected="qwen-14b",
        ...     rule_id="low-cost-chat",
        ...     latency_ms=342.5,
        ...     retry_count=0,
        ...     tenant_id="omniintelligence",
        ...     correlation_id="abc-123",
        ...     success=True,
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    backend_selected: str = Field(
        ...,
        description="Backend ID that served the request.",
    )
    rule_id: str = Field(
        ...,
        description="Routing rule ID applied, or 'default' / 'none'.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end gateway latency in milliseconds.",
    )
    retry_count: int = Field(
        ...,
        ge=0,
        description="Number of backends attempted before success (0 = first succeeded).",
    )
    tenant_id: str = Field(
        ...,
        description="Caller identity from the request for audit log queryability.",
    )
    correlation_id: str = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )
    inference_response: ModelLlmInferenceResponse | None = Field(
        default=None,
        description="Underlying LLM inference response, or None on total failure.",
    )
    success: bool = Field(
        ...,
        description="Whether the request was served successfully.",
    )
    error_message: str = Field(
        default="",
        description="Structured error description on failure; empty on success.",
    )


__all__: list[str] = ["ModelBifrostResponse"]
