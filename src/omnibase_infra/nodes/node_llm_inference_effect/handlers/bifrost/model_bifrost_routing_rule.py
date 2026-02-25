# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Bifrost routing rule model.

Defines a single declarative routing rule that maps a combination of
operation type, capabilities, cost tier, and max latency to a ranked
list of backend endpoint IDs. The bifrost gateway evaluates rules in
priority order and selects the first matching rule.

Related:
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelBifrostRoutingRule(BaseModel):
    """A single declarative routing rule for the bifrost gateway.

    Rules are evaluated in priority order (ascending ``priority`` value).
    The first rule whose ``match_*`` predicates are satisfied by an
    incoming request is applied. When no rule matches, the gateway
    falls through to the ``default_backends`` in ``ModelBifrostConfig``.

    Attributes:
        rule_id: Stable UUID identifier for this rule, recorded in
            every audit log entry. Must be unique within a config.
        priority: Rule evaluation order. Lower values are evaluated first.
            Ties are broken by list insertion order.
        match_operation_types: If non-empty, the request's
            ``operation_type`` must be one of these values. Empty means
            "match any operation type".
        match_capabilities: If non-empty, the request must declare ALL
            of these capabilities. Empty means "match any capabilities".
        match_cost_tiers: If non-empty, the request's ``cost_tier`` must
            be one of these values. Empty means "match any cost tier".
        match_max_latency_ms_lte: If non-None, the request's
            ``max_latency_ms`` must be less than or equal to this value.
            None means "no latency constraint".
        backend_ids: Ordered list of backend IDs to try when this rule
            matches. Backends are attempted in order; the first healthy
            one receives the request.

    Example:
        >>> from uuid import UUID
        >>> rule = ModelBifrostRoutingRule(
        ...     rule_id=UUID("12345678-1234-5678-1234-567812345678"),
        ...     priority=10,
        ...     match_cost_tiers=["low"],
        ...     backend_ids=["qwen-7b", "codestral-7b"],
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    rule_id: UUID = Field(
        ...,
        description="Stable unique identifier for audit logging.",
    )
    priority: int = Field(
        default=100,
        ge=0,
        description="Evaluation order — lower values are evaluated first.",
    )
    match_operation_types: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Restrict to these operation_type values (empty = any).",
    )
    match_capabilities: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Request must declare ALL of these capabilities (empty = any).",
    )
    match_cost_tiers: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Restrict to these cost_tier values (empty = any).",
    )
    match_max_latency_ms_lte: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Request max_latency_ms must be <= this value (None = no constraint)."
        ),
    )
    backend_ids: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Ordered backend IDs to try when this rule matches.",
    )


__all__: list[str] = ["ModelBifrostRoutingRule"]
