# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Concrete Pydantic model implementing ProtocolLLMHealthResponse.

Provides a frozen, serializable health response that satisfies the SPI
structural protocol for LLM provider health checks.

Related:
    - ProtocolLLMHealthResponse (omnibase_spi.protocols.types.protocol_llm_types)
    - OMN-2319: Implement SPI LLM protocol adapters
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLlmHealthResponse(BaseModel):
    """Concrete implementation of ProtocolLLMHealthResponse.

    Captures the result of an LLM provider health check including
    availability status, response latency, available models, and any
    error information.

    Attributes:
        is_healthy: Whether the provider is operational.
        provider_name: Identifier of the provider that was checked.
        response_time_ms: Round-trip latency of the health check in ms.
        available_models: Models reported by the provider during check (immutable tuple).
        error_message: Error details when is_healthy is False, None otherwise.

    Example:
        >>> health = ModelLlmHealthResponse(
        ...     is_healthy=True,
        ...     provider_name="openai-compatible",
        ...     response_time_ms=42.5,
        ...     available_models=["qwen2.5-coder-14b"],
        ... )
        >>> health.is_healthy
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    is_healthy: bool = Field(
        ...,
        description="Whether the LLM provider is healthy and operational.",
    )
    provider_name: str = Field(
        ...,
        min_length=1,
        description="Name of the LLM provider.",
    )
    response_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Response time of the health check in milliseconds.",
    )
    available_models: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Available models reported by the provider.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if health check failed, None if healthy.",
    )


__all__: list[str] = ["ModelLlmHealthResponse"]
