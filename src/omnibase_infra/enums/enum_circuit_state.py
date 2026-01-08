# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Circuit breaker state enumeration.

This module defines the state enumeration for the circuit breaker pattern
used in infrastructure components for fault tolerance.

See Also:
    - MixinAsyncCircuitBreaker: Mixin implementing the circuit breaker pattern
    - docs/patterns/circuit_breaker_implementation.md: Implementation guide
"""

from __future__ import annotations

from enum import Enum


class EnumCircuitState(str, Enum):
    """Circuit breaker state machine.

    The circuit breaker implements a 3-state pattern for fault tolerance:

    States:
        CLOSED: Normal operation
            - All requests are allowed
            - Failures are counted
            - Transitions to OPEN when failure threshold is reached

        OPEN: Circuit tripped
            - All requests are blocked
            - Raises InfraUnavailableError immediately
            - Automatically transitions to HALF_OPEN after reset timeout

        HALF_OPEN: Testing recovery
            - Limited requests allowed for testing
            - First success transitions to CLOSED
            - First failure transitions back to OPEN

    State Transitions:
        CLOSED -> OPEN: Failure count >= threshold
        OPEN -> HALF_OPEN: Current time - last_failure_time > reset_timeout
        HALF_OPEN -> CLOSED: First successful operation
        HALF_OPEN -> OPEN: First failed operation
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


__all__: list[str] = ["EnumCircuitState"]
