# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Output Validation Parameters Model.

Defines the parameter model for validating outputs against
execution shape constraints.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import (
    EnumMessageCategory,
    EnumNodeArchetype,
    EnumNodeOutputType,
)


class ModelOutputValidationParams(BaseModel):
    """Parameters for validating output against execution shape constraints.

    This model encapsulates the parameters needed for the validate_handler_output
    method of RuntimeShapeValidator.

    Attributes:
        node_archetype: The declared node archetype.
        output: The actual output object (used for context in violation message).
        output_category: The message category or node output type of the output.
        file_path: Optional file path for violation reporting.
        line_number: Optional line number for violation reporting.

    Example:
        >>> params = ModelOutputValidationParams(
        ...     node_archetype=EnumNodeArchetype.REDUCER,
        ...     output=some_event,
        ...     output_category=EnumMessageCategory.EVENT,
        ...     file_path="test_handler.py",
        ...     line_number=42,
        ... )
    """

    node_archetype: EnumNodeArchetype = Field(
        ...,
        description="The declared node archetype",
    )
    # NOTE: Using `object` per ONEX guidelines (not `Any`) - output can be any handler return value
    output: object = Field(
        ...,
        description="The actual output object (used for context in violation message)",
    )
    output_category: EnumMessageCategory | EnumNodeOutputType = Field(
        ...,
        description="The message category or node output type of the output",
    )
    file_path: str = Field(
        default="<runtime>",
        description="Optional file path for violation reporting",
    )
    line_number: int = Field(
        default=0,
        ge=0,
        description="Optional line number for violation reporting",
    )

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,  # Allow arbitrary types for output field
    )


__all__ = ["ModelOutputValidationParams"]
