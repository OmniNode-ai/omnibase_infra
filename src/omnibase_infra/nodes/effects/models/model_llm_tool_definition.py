# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM tool definition model for request payloads.

This module provides ModelLlmToolDefinition, which wraps a function
definition with a type discriminator following the OpenAI tool-calling format.

Related:
    - ModelLlmFunctionDef: The wrapped function schema
    - OMN-2103: Phase 3 shared LLM models
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.effects.models.model_llm_function_def import (
    ModelLlmFunctionDef,
)


class ModelLlmToolDefinition(BaseModel):
    """A tool definition sent to the model in the request payload.

    Wraps a :class:`ModelLlmFunctionDef` with a ``type`` discriminator
    following the OpenAI tool-calling format.

    Attributes:
        type: Tool kind discriminator (currently always ``"function"``).
        function: The function schema to expose to the model.

    Example:
        >>> from omnibase_infra.nodes.effects.models.model_llm_function_def import (
        ...     ModelLlmFunctionDef,
        ... )
        >>> defn = ModelLlmToolDefinition(
        ...     function=ModelLlmFunctionDef(name="search", parameters={}),
        ... )
        >>> defn.type
        'function'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    type: Literal["function"] = Field(
        default="function",
        description="Tool kind discriminator.",
    )
    function: ModelLlmFunctionDef = Field(
        ...,
        description="The function schema to expose to the model.",
    )


__all__ = ["ModelLlmToolDefinition"]
