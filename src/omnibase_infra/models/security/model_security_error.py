# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security validation error model.

This module defines the ModelSecurityError model for capturing security
validation failures. Errors indicate policy violations that must be addressed
before handler registration or invocation can proceed.

Example:
    >>> error = ModelSecurityError(
    ...     code="SECRET_SCOPE_NOT_PERMITTED",
    ...     field="secret_scopes",
    ...     message="Secret scope 'database-admin' not permitted in production",
    ...     severity="error",
    ... )

See Also:
    - ModelSecurityWarning: Advisory warnings that don't block validation
    - ModelSecurityValidationResult: Complete validation result container
    - EnumSecurityRuleId: Security validation rule identifiers
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSecurityError(BaseModel):
    """A security validation error.

    Represents a security validation failure that should prevent
    handler registration or invocation. Errors indicate policy
    violations that must be addressed.

    Attributes:
        code: Error code identifier (e.g., "MISSING_SECRET_SCOPE", "INVALID_DOMAIN").
            Should be a stable identifier suitable for programmatic handling.
        field: The field or policy attribute that caused the error.
        message: Human-readable error description.
        severity: Error severity level. Currently accepts string values
            ("error", "critical") for flexibility. May be migrated to
            EnumValidationSeverity in future versions.

    Example:
        >>> error = ModelSecurityError(
        ...     code="SECRET_SCOPE_NOT_PERMITTED",
        ...     field="secret_scopes",
        ...     message="Secret scope 'database-admin' not permitted in production",
        ...     severity="error",
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    code: str = Field(
        description="Error code identifier for programmatic handling",
    )
    field: str = Field(
        description="The field or policy attribute that caused the error",
    )
    message: str = Field(
        description="Human-readable error description",
    )
    severity: str = Field(
        default="error",
        description="Error severity level (e.g., 'error', 'critical')",
    )


__all__ = [
    "ModelSecurityError",
]
