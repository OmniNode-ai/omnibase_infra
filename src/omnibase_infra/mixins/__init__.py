# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infrastructure mixins for common cross-cutting concerns.

This module provides reusable mixins for infrastructure components to implement
common patterns such as circuit breakers, retry logic, health checks, and more.

Mixins follow ONEX patterns:
- Thread-safe async operations
- Infrastructure error integration
- Correlation ID propagation
- Configurable behavior
"""

from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    CircuitState,
    MixinAsyncCircuitBreaker,
)

__all__ = ["MixinAsyncCircuitBreaker", "CircuitState"]
