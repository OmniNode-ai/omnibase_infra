"""Stub for omnibase_core.enums.enum_log_level"""

from enum import Enum


class EnumLogLevel(str, Enum):
    """Log level enum for structured logging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


__all__ = ["EnumLogLevel"]
