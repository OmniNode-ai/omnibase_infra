# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validate and Raise Parameters Model.

Defines the parameter model for validating handler outputs and raising
exceptions on violations, supporting correlation ID for distributed tracing.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import (
    EnumMessageCategory,
    EnumNodeArchetype,
    EnumNodeOutputType,
)


class ModelValidateAndRaiseParams(BaseModel):
    """Parameters for validating handler output and raising exception if invalid.

    This model encapsulates the parameters needed for the validate_and_raise
    method of RuntimeShapeValidator, including optional correlation ID for
    distributed tracing.

    Attributes:
        node_archetype: The declared node archetype.
        output: The actual output object.
        output_category: The message category or node output type of the output.
        file_path: Optional file path for violation reporting.
        line_number: Optional line number for violation reporting.
        correlation_id: Optional correlation ID for distributed tracing.

    Example:
        >>> from uuid import uuid4
        >>> params = ModelValidateAndRaiseParams(
        ...     node_archetype=EnumNodeArchetype.REDUCER,
        ...     output=event_output,
        ...     output_category=EnumMessageCategory.EVENT,
        ...     file_path="handler.py",
        ...     line_number=42,
        ...     correlation_id=uuid4(),
        ... )
    """

    node_archetype: EnumNodeArchetype = Field(
        ...,
        description="The declared node archetype",
    )
    # NOTE: Using `object` per ONEX guidelines (not `Any`) - output can be any handler return value
    output: object = Field(
        ...,
        description="The actual output object",
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
    correlation_id: UUID | None = Field(
        default=None,
        description="Optional correlation ID for distributed tracing",
    )

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,  # Allow arbitrary types for output field
    )


__all__ = ["ModelValidateAndRaiseParams"]
