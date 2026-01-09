# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security validation warning model.

This module defines the ModelSecurityWarning model for capturing security
validation advisories. Warnings do not block handler registration or invocation,
but indicate potential issues that should be reviewed.

Example:
    >>> warning = ModelSecurityWarning(
    ...     code="WILDCARD_DOMAIN_ACCESS",
    ...     field="allowed_domains",
    ...     message="Handler permits access to all domains - consider restricting",
    ... )

See Also:
    - ModelSecurityError: Blocking errors that prevent validation
    - ModelSecurityValidationResult: Complete validation result container
    - EnumSecurityRuleId: Security validation rule identifiers
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSecurityWarning(BaseModel):
    """A security validation warning.

    Represents a security advisory that does not block handler
    registration or invocation, but indicates a potential issue
    that should be reviewed.

    Attributes:
        code: Warning code identifier (e.g., "BROAD_DOMAIN_ACCESS").
            Should be a stable identifier suitable for programmatic handling.
        field: The field or policy attribute that triggered the warning.
        message: Human-readable warning description.

    Example:
        >>> warning = ModelSecurityWarning(
        ...     code="WILDCARD_DOMAIN_ACCESS",
        ...     field="allowed_domains",
        ...     message="Handler permits access to all domains - consider restricting",
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    code: str = Field(
        description="Warning code identifier for programmatic handling",
    )
    field: str = Field(
        description="The field or policy attribute that triggered the warning",
    )
    message: str = Field(
        description="Human-readable warning description",
    )


__all__ = [
    "ModelSecurityWarning",
]
