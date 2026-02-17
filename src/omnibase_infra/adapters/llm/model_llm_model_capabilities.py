# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Concrete Pydantic model implementing ProtocolModelCapabilities.

Provides a frozen, serializable model capabilities description that satisfies
the SPI structural protocol for model capability discovery.

Related:
    - ProtocolModelCapabilities (omnibase_spi.protocols.types.protocol_llm_types)
    - OMN-2319: Implement SPI LLM protocol adapters
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLlmModelCapabilities(BaseModel):
    """Concrete implementation of ProtocolModelCapabilities.

    Describes the capabilities of a specific model hosted by a provider,
    including context window size, supported features, and modalities.

    Attributes:
        model_name: Model identifier (e.g. 'qwen2.5-coder-14b').
        supports_streaming: Whether the model supports streaming responses.
        supports_function_calling: Whether tool/function calling is supported.
        max_context_length: Maximum context window size in tokens.
        supported_modalities: Input/output modalities as immutable tuple (e.g. ('text',), ('text', 'vision')).
        cost_per_1k_input_tokens: Cost per 1000 input tokens in USD (0.0 for local).
        cost_per_1k_output_tokens: Cost per 1000 output tokens in USD (0.0 for local).

    Example:
        >>> caps = ModelLlmModelCapabilities(
        ...     model_name="qwen2.5-coder-14b",
        ...     supports_streaming=True,
        ...     supports_function_calling=True,
        ...     max_context_length=32768,
        ...     supported_modalities=["text"],
        ... )
        >>> caps.max_context_length
        32768
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    model_name: str = Field(
        ...,
        min_length=1,
        description="Name of the model.",
    )
    supports_streaming: bool = Field(
        default=True,
        description="Whether the model supports streaming responses.",
    )
    supports_function_calling: bool = Field(
        default=False,
        description="Whether the model supports function/tool calling.",
    )
    max_context_length: int = Field(
        default=4096,
        ge=1,
        description="Maximum context length in tokens.",
    )
    supported_modalities: tuple[str, ...] = Field(
        default_factory=lambda: ("text",),
        description="Supported input/output modalities.",
    )
    cost_per_1k_input_tokens: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost per 1000 input tokens in USD. 0.0 for local models.",
    )
    cost_per_1k_output_tokens: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost per 1000 output tokens in USD. 0.0 for local models.",
    )


__all__: list[str] = ["ModelLlmModelCapabilities"]
