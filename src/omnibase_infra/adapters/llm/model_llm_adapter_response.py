# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Adapter-level LLM response model implementing ProtocolLLMResponse.

Bridges the SPI protocol's field names (generated_text, model_used,
usage_statistics) to the infra layer's ModelLlmInferenceResponse.

Related:
    - ProtocolLLMResponse (omnibase_spi.protocols.types.protocol_llm_types)
    - ModelLlmInferenceResponse (omnibase_infra.nodes.effects.models)
    - OMN-2319: Implement SPI LLM protocol adapters (Gap 5)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelLlmAdapterResponse(BaseModel):
    """Response model satisfying ProtocolLLMResponse structural protocol.

    Maps the SPI-level response interface from infra-layer response models.
    The ``content`` property provides a convenience alias.

    Attributes:
        generated_text: The generated text content.
        model_used: Model identifier that produced the response.
        usage_statistics: Token usage as a JSON-compatible dictionary.
        finish_reason: Why generation stopped (e.g. 'stop', 'length').
        response_metadata: Additional response metadata.

    Example:
        >>> resp = ModelLlmAdapterResponse(
        ...     generated_text="Hello, world!",
        ...     model_used="qwen2.5-coder-14b",
        ...     usage_statistics={"prompt_tokens": 10, "completion_tokens": 5},
        ...     finish_reason="stop",
        ... )
        >>> resp.content
        'Hello, world!'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    generated_text: str = Field(
        default="",
        description="The generated text.",
    )
    model_used: str = Field(
        ...,
        min_length=1,
        description="Model that was used for generation.",
    )
    # NOTE: Mutable dict accepted for SPI JsonType conformance (OMN-2319)
    usage_statistics: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Usage statistics (tokens, time, etc.) as JSON dict.",
    )
    finish_reason: str = Field(
        default="unknown",
        description="Reason generation finished.",
    )
    # NOTE: Mutable dict accepted for SPI JsonType conformance (OMN-2319)
    response_metadata: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Additional response metadata.",
    )

    @property
    def content(self) -> str:
        """Convenience alias for generated_text.

        Matches the ProtocolLLMProvider docstring examples that reference
        ``response.content``.
        """
        return self.generated_text


__all__: list[str] = ["ModelLlmAdapterResponse"]
