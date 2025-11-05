"""Compliance level enumeration for security policy configuration."""

from enum import Enum


class EnumComplianceLevel(str, Enum):
    """Enumeration for security compliance levels."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    MAXIMUM = "maximum"
