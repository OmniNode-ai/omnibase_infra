"""
SASL mechanism enum.

Strongly typed values for SASL authentication mechanisms.
"""

from enum import Enum


class EnumSaslMechanism(str, Enum):
    """Strongly typed SASL mechanism values."""

    PLAIN = "PLAIN"
    SCRAM_SHA_256 = "SCRAM-SHA-256"
    SCRAM_SHA_512 = "SCRAM-SHA-512"
    GSSAPI = "GSSAPI"
    OAUTHBEARER = "OAUTHBEARER"


# Export for use
__all__ = ["EnumSaslMechanism"]
