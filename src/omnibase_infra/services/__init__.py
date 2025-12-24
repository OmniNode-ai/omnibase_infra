# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Services Module.

Provides high-level services that compose infrastructure components for
use by orchestrators and runtime hosts. Services provide clean interfaces
for common operations and encapsulate complexity.

Exports:
    ModelTimeoutEmissionResult: Result model for timeout emission processing
    ModelTimeoutQueryResult: Result model for timeout queries
    ServiceTimeoutEmission: Service for emitting timeout events with markers
    ServiceTimeoutQuery: Service for querying overdue registration entities

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
"""

from omnibase_infra.services.service_timeout_emission import (
    ModelTimeoutEmissionResult,
    ServiceTimeoutEmission,
)
from omnibase_infra.services.service_timeout_query import (
    ModelTimeoutQueryResult,
    ServiceTimeoutQuery,
)

__all__ = [
    "ModelTimeoutEmissionResult",
    "ModelTimeoutQueryResult",
    "ServiceTimeoutEmission",
    "ServiceTimeoutQuery",
]
