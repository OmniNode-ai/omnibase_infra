"""Validation severity enum for architecture validation rules.

This module defines severity levels that determine how architecture rule
violations should be handled during validation.
"""

from enum import Enum


class EnumValidationSeverity(str, Enum):
    """Severity levels for architecture validation rules.

    Determines how violations should be treated:
    - ERROR: Fail validation, block startup
    - WARNING: Log warning, allow startup
    - INFO: Informational only
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    def blocks_startup(self) -> bool:
        """Whether this severity level should block runtime startup.

        Returns:
            True if this severity level should prevent the runtime from
            starting when a violation is detected.
        """
        return self == EnumValidationSeverity.ERROR

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value
