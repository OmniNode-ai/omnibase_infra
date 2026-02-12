# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM function call model for tool-calling interactions.

This module provides ModelLlmFunctionCall, representing a concrete
function invocation returned by the model.

Related:
    - ModelLlmToolCall: Wraps this model with an id and type discriminator
    - OMN-2103: Phase 3 shared LLM models
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLlmFunctionCall(BaseModel):
    """A concrete function invocation returned by the model.

    Attributes:
        name: Name of the function the model chose to call.
        arguments: Serialised JSON string of the call arguments.

    Example:
        >>> call = ModelLlmFunctionCall(name="get_weather", arguments='{"city": "London"}')
        >>> call.name
        'get_weather'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(
        ...,
        min_length=1,
        description="Name of the function the model chose to call.",
    )
    arguments: str = Field(
        ...,
        description=(
            "Serialised JSON string of the call arguments. "
            "No JSON validation is performed at the model level; "
            "callers are responsible for parsing and validating the content."
        ),
    )


__all__ = ["ModelLlmFunctionCall"]
