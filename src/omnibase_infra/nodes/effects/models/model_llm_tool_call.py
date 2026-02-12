# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM tool call model for assistant message responses.

This module provides ModelLlmToolCall, representing a single tool call
returned by the model in an assistant message.

Related:
    - ModelLlmFunctionCall: The wrapped function invocation
    - OMN-2103: Phase 3 shared LLM models
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.effects.models.model_llm_function_call import (
    ModelLlmFunctionCall,
)


class ModelLlmToolCall(BaseModel):
    """A single tool call returned by the model in an assistant message.

    Attributes:
        id: Provider-assigned identifier for this tool call.
        type: Tool kind discriminator (currently always ``"function"``).
        function: The function invocation details.

    Example:
        >>> from omnibase_infra.nodes.effects.models.model_llm_function_call import (
        ...     ModelLlmFunctionCall,
        ... )
        >>> tc = ModelLlmToolCall(
        ...     id="call_abc123",
        ...     function=ModelLlmFunctionCall(name="search", arguments='{"q": "hello"}'),
        ... )
        >>> tc.id
        'call_abc123'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: str = Field(
        ...,
        min_length=1,
        description="Provider-assigned identifier for this tool call.",
    )
    type: Literal["function"] = Field(
        default="function",
        description="Tool kind discriminator.",
    )
    function: ModelLlmFunctionCall = Field(
        ...,
        description="The function invocation details.",
    )


__all__ = ["ModelLlmToolCall"]
