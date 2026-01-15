# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validation severity levels for security and other validation errors.

This module defines the severity levels used in validation error models.
Severity indicates the impact and urgency of addressing a validation failure.

Severity Levels:
    - ERROR: Standard validation failure that must be addressed
    - CRITICAL: Severe failure that poses security or stability risk
    - WARNING: Advisory issue that should be reviewed but may not block
    - INFO: Informational only, does not block operations

Usage:
    Used in ModelSecurityError and other validation error models to
    provide structured severity classification.

See Also:
    - ModelSecurityError: Security validation error model
    - ModelSecurityValidationResult: Complete validation result container
"""

from enum import StrEnum


class EnumValidationSeverity(StrEnum):
    """Validation error severity levels.

    Classifies the severity of validation errors to indicate their
    impact and urgency. Used in security validation, architecture
    validation, and other validation contexts.

    Attributes:
        ERROR: Standard validation failure. The validation constraint
            was violated and must be addressed before proceeding.
            This is the default severity level. Blocks startup.
        CRITICAL: Severe validation failure. Indicates a security risk,
            data integrity issue, or system stability concern that
            requires immediate attention. Blocks startup.
        WARNING: Advisory issue. The validation identified a potential
            problem that should be reviewed but may not necessarily
            block the operation. Does not block startup.
        INFO: Informational only. Provides context or suggestions
            without indicating any problem. Does not block startup.

    Example:
        >>> from omnibase_infra.enums import EnumValidationSeverity
        >>> severity = EnumValidationSeverity.ERROR
        >>> severity.value
        'error'
        >>> severity == "error"
        True
        >>> severity.blocks_startup()
        True
        >>> EnumValidationSeverity.INFO.blocks_startup()
        False

    See Also:
        - ModelSecurityError: Uses this enum for severity classification
        - EnumSecurityRuleId: Security validation rule identifiers
    """

    ERROR = "error"
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

    def blocks_startup(self) -> bool:
        """Whether this severity level should block runtime startup.

        Returns:
            True if this severity level should prevent the runtime from
            starting when a violation is detected.
        """
        return self in (EnumValidationSeverity.ERROR, EnumValidationSeverity.CRITICAL)


__all__ = ["EnumValidationSeverity"]
