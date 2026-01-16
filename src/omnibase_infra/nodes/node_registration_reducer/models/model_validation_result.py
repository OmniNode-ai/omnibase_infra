# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validation result model for registration reducer.

This model represents the result of event validation with detailed error
information. It follows the ONEX pattern of using sentinel values instead
of nullable unions to minimize union count (OMN-1004).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

# Validation error codes for distinct error identification
ValidationErrorCode = Literal[
    "missing_node_id",
    "missing_node_type",
    "invalid_node_type",
]

# Sentinel value for "not set" state
_SENTINEL_STR: str = ""


class ModelValidationResult(BaseModel):
    """Result of event validation with detailed error information.

    This Pydantic model follows ONEX requirements for Pydantic-based
    data structures with sentinel values instead of nullable unions
    (OMN-1004).

    Sentinel Values:
        - Empty string ("") for field_name and error_message means "not set"
        - None for error_code (unavoidable for Literal type safety)
        - Use ``has_field_name``, ``has_error_message`` to check

    Input Normalization:
        Constructors accept ``None`` for string fields and convert to sentinel
        for developer convenience. This is NOT backwards compatibility (per
        CLAUDE.md policy) - it is input normalization that may be removed in
        future versions without notice.

    Attributes:
        is_valid: Whether the event passed validation.
        error_code: Distinct code identifying the validation failure (if any).
        field_name: Name of the field that failed validation. Empty string if not set.
        error_message: Human-readable error message for logging. Empty string if not set.

    .. versionchanged:: 0.7.0
        Refactored to use sentinel values for string fields (OMN-1004).
        Extracted to separate file (OMN-1104).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    is_valid: bool
    error_code: ValidationErrorCode | None = None
    field_name: str = _SENTINEL_STR
    error_message: str = _SENTINEL_STR

    # ---- Validators for None-to-Sentinel Conversion ----
    @field_validator("field_name", "error_message", mode="before")
    @classmethod
    def _convert_none_to_str_sentinel(cls, v: object) -> str:
        """Convert None to empty string sentinel for input normalization."""
        if v is None:
            return _SENTINEL_STR
        if isinstance(v, str):
            return v
        return str(v)

    # ---- Sentinel Check Properties ----
    @property
    def has_field_name(self) -> bool:
        """Check if field_name is set (not empty string)."""
        return self.field_name != _SENTINEL_STR

    @property
    def has_error_message(self) -> bool:
        """Check if error_message is set (not empty string)."""
        return self.error_message != _SENTINEL_STR

    @classmethod
    def success(cls) -> ModelValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True)

    @classmethod
    def failure(
        cls,
        error_code: ValidationErrorCode,
        field_name: str,
        error_message: str,
    ) -> ModelValidationResult:
        """Create a failed validation result with error details."""
        return cls(
            is_valid=False,
            error_code=error_code,
            field_name=field_name,
            error_message=error_message,
        )


# Convenience alias (not backwards compatibility - may be removed without notice)
ValidationResult = ModelValidationResult

__all__ = [
    "ModelValidationResult",
    "ValidationErrorCode",
    "ValidationResult",
]
