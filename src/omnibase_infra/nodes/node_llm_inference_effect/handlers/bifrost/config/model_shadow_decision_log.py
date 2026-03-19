# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Shadow decision log model for bifrost shadow mode.

Captures both the static (actual) routing decision and the shadow
(learned policy) recommendation for the same request, along with
comparison metrics for promotion gate evaluation.

Promotion Gate Criteria (evaluated on dashboard):
    - Minimum sample threshold: >= 100 shadow decisions
    - Estimated reward delta is positive overall
    - No critical scenario bucket shows materially worse cost/latency tradeoffs
    - Endpoint concentration: no single endpoint > 80% of shadow recommendations
      unless justified by routing rules
    - Shadow recommendations don't exhibit unstable action selection
    - Top disagreement scenarios manually reviewed and accepted

Related Tickets:
    - OMN-5570: Shadow Mode + Comparison Dashboard
    - OMN-5556: Learned Decision Optimization Platform (epic)
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelShadowDecisionLog(BaseModel):
    """Log entry for a single shadow routing decision comparison.

    Captures both the static (actual) routing decision and the shadow
    (learned policy) recommendation for the same request, along with
    comparison metrics.

    Attributes:
        correlation_id: Request correlation ID for tracing.
        timestamp: When the shadow decision was computed.
        static_backend_selected: Backend selected by static routing rules.
        shadow_backend_recommended: Backend recommended by the learned policy.
        agreed: Whether static and shadow selected the same backend.
        static_rule_id: UUID of the matched static routing rule (None if default).
        request_operation_type: The operation type from the request.
        request_cost_tier: The cost tier from the request.
        request_max_latency_ms: The latency budget from the request.
        estimated_token_count: Estimated token count for the request (if available).
        shadow_confidence: Confidence score from the learned policy (0.0-1.0).
        shadow_latency_ms: Time taken for the shadow policy evaluation.
        policy_version: Version of the shadow policy that produced this recommendation.
        shadow_action_distribution: Full action probability distribution from policy.
        static_backend_estimated_cost: Estimated cost for the static selection.
        shadow_backend_estimated_cost: Estimated cost for the shadow recommendation.
        static_backend_estimated_latency_ms: Estimated latency for static selection.
        shadow_backend_estimated_latency_ms: Estimated latency for shadow recommendation.
        tenant_id: Tenant ID from the request.

    Example:
        >>> from uuid import uuid4
        >>> log = ModelShadowDecisionLog(
        ...     correlation_id=uuid4(),
        ...     static_backend_selected="qwen-14b",
        ...     shadow_backend_recommended="qwen-30b",
        ...     agreed=False,
        ...     request_operation_type="chat_completion",
        ...     request_cost_tier="mid",
        ...     request_max_latency_ms=5000,
        ...     shadow_confidence=0.85,
        ...     shadow_latency_ms=2.3,
        ...     policy_version="v1.0.0-alpha",
        ...     tenant_id=uuid4(),
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID for distributed tracing.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the shadow decision was computed.",
    )
    static_backend_selected: str = Field(
        ...,
        description="Backend selected by the static routing rules.",
    )
    shadow_backend_recommended: str = Field(
        ...,
        description="Backend recommended by the learned shadow policy.",
    )
    agreed: bool = Field(
        ...,
        description="Whether static and shadow selected the same backend.",
    )
    static_rule_id: UUID | None = Field(
        default=None,
        description="UUID of the matched static routing rule (None if default).",
    )
    request_operation_type: str = Field(
        ...,
        description="The operation_type from the bifrost request.",
    )
    request_cost_tier: str = Field(
        ...,
        description="The cost_tier from the bifrost request.",
    )
    request_max_latency_ms: int = Field(
        ...,
        ge=1,
        description="The max_latency_ms from the bifrost request.",
    )
    estimated_token_count: int | None = Field(
        default=None,
        ge=0,
        description="Estimated token count for the request (if available).",
    )
    shadow_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from the learned policy (0.0-1.0).",
    )
    shadow_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken for the shadow policy evaluation (ms).",
    )
    policy_version: str = Field(
        ...,
        max_length=128,
        description="Version of the shadow policy that produced this recommendation.",
    )
    shadow_action_distribution: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Full action probability distribution from the learned policy. "
            "Keys are backend IDs, values are probabilities summing to 1.0."
        ),
    )
    static_backend_estimated_cost: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated cost for the static backend selection.",
    )
    shadow_backend_estimated_cost: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated cost for the shadow backend recommendation.",
    )
    static_backend_estimated_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated latency (ms) for the static backend selection.",
    )
    shadow_backend_estimated_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated latency (ms) for the shadow backend recommendation.",
    )
    tenant_id: UUID = Field(
        ...,
        description="Tenant ID from the request for audit logging.",
    )


__all__: list[str] = ["ModelShadowDecisionLog"]
