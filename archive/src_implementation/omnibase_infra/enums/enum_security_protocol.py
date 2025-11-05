"""Security protocol enumeration for Kafka configuration."""

from enum import Enum


class EnumSecurityProtocol(str, Enum):
    """Enumeration for Kafka security protocols."""

    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"
