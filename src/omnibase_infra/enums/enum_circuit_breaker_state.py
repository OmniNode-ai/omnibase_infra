"""Circuit breaker state enumeration for fault tolerance monitoring."""

from enum import Enum


class EnumCircuitBreakerState(str, Enum):
    """Circuit breaker state enumeration for fault tolerance patterns."""

    CLOSED = "CLOSED"
    HALF_OPEN = "HALF_OPEN"
    OPEN = "OPEN"