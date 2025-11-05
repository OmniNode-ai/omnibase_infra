"""TLS version enumeration for security policy configuration."""

from enum import Enum


class EnumTlsVersion(str, Enum):
    """Enumeration for supported TLS versions."""

    TLS_1_2 = "1.2"
    TLS_1_3 = "1.3"
