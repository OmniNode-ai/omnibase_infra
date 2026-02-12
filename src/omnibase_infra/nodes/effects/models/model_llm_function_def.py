# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM function definition model for tool-calling interactions.

This module provides ModelLlmFunctionDef, a JSON-Schema description
of a callable function that an LLM may invoke.

Related:
    - ModelLlmToolDefinition: Wraps this model with a type discriminator
    - OMN-2103: Phase 3 shared LLM models
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelLlmFunctionDef(BaseModel):
    """JSON-Schema description of a function that an LLM may call.

    Attributes:
        name: Machine-readable function name (e.g. ``"get_weather"``).
        description: Human-readable summary shown to the model.
        parameters: JSON Schema object describing accepted arguments.

    Example:
        >>> fn = ModelLlmFunctionDef(
        ...     name="get_weather",
        ...     description="Return current weather for a city.",
        ...     parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        ... )
        >>> fn.name
        'get_weather'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(
        ...,
        description="Machine-readable function name.",
    )
    description: str = Field(
        default="",
        description="Human-readable summary shown to the model.",
    )
    # ONEX_EXCLUDE: any_type - JSON Schema objects are inherently untyped dicts.
    # The parameters field mirrors the OpenAI function-calling spec where parameter
    # schemas are arbitrary JSON Schema objects with no fixed structure.
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema object describing accepted arguments.",
    )


__all__ = ["ModelLlmFunctionDef"]
