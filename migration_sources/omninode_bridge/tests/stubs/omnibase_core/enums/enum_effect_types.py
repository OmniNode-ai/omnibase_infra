"""Stub for omnibase_core.enums.enum_effect_types"""

from enum import Enum


class EnumCircuitBreakerState(str, Enum):
    """Circuit breaker state enum stub."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


__all__ = ["EnumCircuitBreakerState"]
