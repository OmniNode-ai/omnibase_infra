# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Adapter-level LLM request model implementing ProtocolLLMRequest.

Bridges the SPI protocol's field names (prompt, model_name, parameters) to
the infra layer's ModelLlmInferenceRequest field names (prompt, model, etc.).

Related:
    - ProtocolLLMRequest (omnibase_spi.protocols.types.protocol_llm_types)
    - ModelLlmInferenceRequest (omnibase_infra.nodes.effects.models)
    - OMN-2319: Implement SPI LLM protocol adapters (Gap 5)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelLlmAdapterRequest(BaseModel):
    """Request model satisfying ProtocolLLMRequest structural protocol.

    Maps the SPI-level request interface to concrete fields. The adapter
    layer translates these into ModelLlmInferenceRequest for the handlers.

    Attributes:
        prompt: Main prompt or query text.
        model_name: Model identifier to route to.
        parameters: Generation parameters as a JSON-compatible dictionary.
        max_tokens: Maximum tokens to generate, or None for default.
        temperature: Sampling temperature, or None for default.

    Example:
        >>> req = ModelLlmAdapterRequest(
        ...     prompt="Explain ONEX architecture",
        ...     model_name="qwen2.5-coder-14b",
        ...     temperature=0.7,
        ... )
        >>> req.model_name
        'qwen2.5-coder-14b'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    prompt: str = Field(
        ...,
        min_length=1,
        description="The main prompt or query.",
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Name of the model to use.",
    )
    # NOTE: Mutable dict accepted for SPI JsonType conformance (OMN-2319)
    parameters: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Generation parameters as JSON-compatible dictionary.",
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
        description="Temperature for generation.",
    )


__all__: list[str] = ["ModelLlmAdapterRequest"]
