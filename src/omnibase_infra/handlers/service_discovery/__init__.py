# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Service Discovery Handlers Module.  # ai-slop-ok: pre-existing docstring opener

This module provides pluggable handler implementations for service discovery
operations, supporting the capability-oriented node architecture.

Handlers:
    - HandlerServiceDiscoveryMock: In-memory mock for testing

Models:
    - ModelServiceInfo: Service information model
    - ModelHandlerRegistrationResult: Handler-level registration operation result
    - ModelDiscoveryResult: Discovery operation result

Protocols:
    - ProtocolDiscoveryOperations: Discovery operations protocol definition
"""

from omnibase_infra.handlers.service_discovery.handler_service_discovery_mock import (
    HandlerServiceDiscoveryMock,
)
from omnibase_infra.handlers.service_discovery.models import (
    ModelDiscoveryResult,
    ModelHandlerRegistrationResult,
    ModelServiceInfo,
)
from omnibase_infra.handlers.service_discovery.protocol_discovery_operations import (
    ProtocolDiscoveryOperations,
)

__all__: list[str] = [
    "HandlerServiceDiscoveryMock",
    "ModelDiscoveryResult",
    "ModelHandlerRegistrationResult",
    "ModelServiceInfo",
    "ProtocolDiscoveryOperations",
]
