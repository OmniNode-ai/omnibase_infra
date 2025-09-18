"""
Security protocol enum.

Strongly typed values for security protocols.
"""

from enum import Enum


class EnumSecurityProtocol(str, Enum):
    """Strongly typed security protocol values."""

    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"


# Export for use
__all__ = ["EnumSecurityProtocol"]