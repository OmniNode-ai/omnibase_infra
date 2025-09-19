"""
Compliance level enum.

Strongly typed values for security compliance levels.
"""

from enum import Enum


class EnumComplianceLevel(str, Enum):
    """Strongly typed compliance level values."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    MAXIMUM = "maximum"


# Export for use
__all__ = ["EnumComplianceLevel"]
