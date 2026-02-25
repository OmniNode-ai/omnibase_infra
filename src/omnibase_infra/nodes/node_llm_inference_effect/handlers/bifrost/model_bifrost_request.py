# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Bifrost gateway request model.

Defines the input contract for the bifrost gateway handler. Callers
describe their routing intent (operation type, required capabilities,
cost tier, latency budget) rather than naming a specific backend — the
gateway selects the backend from its routing rules.

Related:
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums.enum_cost_tier import EnumCostTier
from omnibase_infra.enums.enum_llm_operation_type import EnumLlmOperationType
from omnibase_infra.models.types import JsonDict


class ModelBifrostRequest(BaseModel):
    """Input contract for the bifrost LLM gateway handler.

    Callers declare *what* they need (operation type, capabilities,
    cost tier, latency budget, prompt content) rather than *where* to
    send it. The bifrost gateway resolves the target backend by
    evaluating routing rules against these fields.

    Attributes:
        operation_type: LLM operation to perform (e.g.
            ``EnumLlmOperationType.CHAT_COMPLETION``).
            Used for routing rule matching.
        capabilities: Set of capabilities the selected backend must
            support (e.g. ``["tool_calling", "json_mode"]``). All
            declared capabilities must be supported by the chosen
            backend's routing rule.
        max_latency_ms: Maximum acceptable end-to-end latency in
            milliseconds. The gateway uses this for SLA-aware routing
            rule selection (``match_max_latency_ms_lte``).
        cost_tier: Cost tier preference. The gateway routes to the
            cheapest tier that satisfies the request.
        tenant_id: Caller identity UUID for audit logging and per-tenant
            routing policy (future). Recorded in every audit log entry.
        model: Optional model identifier override. When set, the gateway
            passes this to the backend; otherwise the backend's configured
            ``model_name`` is used.
        messages: Chat messages for ``chat_completion`` operations.
        prompt: Text prompt for ``completion`` operations.
        system_prompt: Optional system prompt prepended to messages.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        correlation_id: Optional correlation ID for distributed tracing.
            Auto-generated if not provided.

    Example:
        >>> from uuid import UUID
        >>> from omnibase_infra.enums.enum_llm_operation_type import EnumLlmOperationType
        >>> req = ModelBifrostRequest(
        ...     operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        ...     capabilities=["tool_calling"],
        ...     max_latency_ms=5000,
        ...     cost_tier=EnumCostTier.LOW,
        ...     tenant_id=UUID("12345678-1234-5678-1234-567812345678"),
        ...     messages=[{"role": "user", "content": "Hello"}],
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    operation_type: EnumLlmOperationType = Field(
        ...,
        description="LLM operation type for routing rule matching.",
    )
    capabilities: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Capabilities the selected backend must support.",
    )
    max_latency_ms: int = Field(
        default=10_000,
        ge=1,
        le=600_000,
        description="Maximum acceptable end-to-end latency in milliseconds.",
    )
    cost_tier: EnumCostTier = Field(
        default=EnumCostTier.MID,
        description="Cost tier preference for backend selection.",
    )
    tenant_id: UUID = Field(
        ...,
        description="Caller identity (UUID) for audit logging.",
    )
    model: str | None = Field(
        default=None,
        description="Optional model identifier override for the backend request.",
    )
    messages: tuple[JsonDict, ...] = Field(
        default_factory=tuple,
        description="Chat messages for chat_completion operations.",
    )
    prompt: str | None = Field(
        default=None,
        description="Text prompt for completion operations.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt prepended to messages.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing (auto-generated if None).",
    )

    @field_validator("capabilities", mode="before")
    @classmethod
    def _coerce_capabilities(cls, v: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        """Coerce list inputs to tuple for strict=True compatibility.

        Args:
            v: The raw capabilities value.

        Returns:
            A tuple of capability strings.
        """
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("messages", mode="before")
    @classmethod
    def _coerce_messages(
        cls, v: list[JsonDict] | tuple[JsonDict, ...]
    ) -> tuple[JsonDict, ...]:
        """Coerce list inputs to tuple for strict=True compatibility.

        Args:
            v: The raw messages value.

        Returns:
            A tuple of message dicts.
        """
        if isinstance(v, list):
            return tuple(v)
        return v


__all__: list[str] = ["ModelBifrostRequest"]
