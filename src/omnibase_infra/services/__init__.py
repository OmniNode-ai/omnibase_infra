# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Services Module.

Provides high-level services that compose infrastructure components for
use by orchestrators and runtime hosts. Services provide clean interfaces
for common operations and encapsulate complexity.

Exports:
    ModelTimeoutEmissionConfig: Configuration for timeout emitter
    ModelTimeoutEmissionResult: Result model for timeout emission processing
    ModelTimeoutQueryResult: Result model for timeout queries
    TimeoutEmitter: Emitter for timeout events with markers
    TimeoutEmissionProcessor: Backwards compatibility alias for TimeoutEmitter
    ServiceTimeoutEmission: Backwards compatibility alias for TimeoutEmitter
    TimeoutScanner: Scanner for querying overdue registration entities

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
"""

from omnibase_infra.services.timeout_emitter import (
    ModelTimeoutEmissionConfig,
    ModelTimeoutEmissionResult,
    ServiceTimeoutEmission,
    TimeoutEmissionProcessor,
    TimeoutEmitter,
)
from omnibase_infra.services.timeout_scanner import (
    ModelTimeoutQueryResult,
    TimeoutScanner,
)

__all__ = [
    "ModelTimeoutEmissionConfig",
    "ModelTimeoutEmissionResult",
    "ModelTimeoutQueryResult",
    "ServiceTimeoutEmission",
    "TimeoutEmissionProcessor",
    "TimeoutEmitter",
    "TimeoutScanner",
]
