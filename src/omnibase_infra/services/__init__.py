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
    ServiceTimeoutEmitter: Emitter for timeout events with markers
    ServiceTimeoutScanner: Scanner for querying overdue registration entities

ServiceHealth Import Guide
--------------------------
ServiceHealth and related constants are NOT exported from this __init__.py
to avoid circular imports with the runtime module.

**Direct import (REQUIRED)**::

    from omnibase_infra.services.service_health import ServiceHealth
    from omnibase_infra.services.service_health import DEFAULT_HTTP_HOST
    from omnibase_infra.services.service_health import DEFAULT_HTTP_PORT

Related Tickets:
    - OMN-529: ONEX Compliance container injection (ServiceHealth moved here)
    - OMN-932 (C2): Durable Timeout Handling
"""

from omnibase_infra.services.service_timeout_emitter import (
    ModelTimeoutEmissionConfig,
    ModelTimeoutEmissionResult,
    ServiceTimeoutEmitter,
)
from omnibase_infra.services.service_timeout_scanner import (
    ModelTimeoutQueryResult,
    ServiceTimeoutScanner,
)

__all__ = [
    "ModelTimeoutEmissionConfig",
    "ModelTimeoutEmissionResult",
    "ModelTimeoutQueryResult",
    "ServiceTimeoutEmitter",
    "ServiceTimeoutScanner",
]
