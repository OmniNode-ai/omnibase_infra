# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validation Error Parameters Model.

Defines the parameter model for creating security validation errors,
encapsulating all required and optional parameters for the
convert_to_validation_error function.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.handlers import ModelHandlerIdentifier


class ModelValidationErrorParams(BaseModel):
    """Parameters for creating a security validation error.

    This model encapsulates the parameters needed to create a
    ModelHandlerValidationError from security validation results.

    Attributes:
        rule_code: Unique code for the validation rule (e.g., "SECURITY-001").
        message: Human-readable error message.
        remediation_hint: Actionable fix suggestion.
        handler_identity: Handler identification information.
        file_path: Optional file path where error occurred.
        line_number: Optional line number (1-indexed).
        details: Optional additional context.

    Example:
        >>> params = ModelValidationErrorParams(
        ...     rule_code="SECURITY-001",
        ...     message="Handler exposes 'get_password' method",
        ...     remediation_hint="Prefix with underscore: '_get_password'",
        ...     handler_identity=ModelHandlerIdentifier.from_handler_id("auth"),
        ...     file_path="nodes/auth/handlers/handler_authenticate.py",
        ...     line_number=42,
        ... )
    """

    rule_code: str = Field(
        ...,
        description="Unique code for the validation rule (e.g., 'SECURITY-001')",
        min_length=1,
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        min_length=1,
    )
    remediation_hint: str = Field(
        ...,
        description="Actionable fix suggestion",
        min_length=1,
    )
    handler_identity: ModelHandlerIdentifier = Field(
        ...,
        description="Handler identification information",
    )
    file_path: str | None = Field(
        default=None,
        description="Optional file path where error occurred",
    )
    line_number: int | None = Field(
        default=None,
        ge=1,
        description="Optional line number (1-indexed)",
    )
    details: dict[str, object] | None = Field(
        default=None,
        description="Optional additional context",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


__all__ = ["ModelValidationErrorParams"]
