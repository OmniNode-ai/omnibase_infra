# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Discovery Handlers Module.

This module provides pluggable handler implementations for service discovery
operations, supporting the capability-oriented node architecture.

Handlers:
    - HandlerServiceDiscoveryConsul: Consul-backed service discovery
    - HandlerServiceDiscoveryMock: In-memory mock for testing

Models:
    - ModelServiceInfo: Service information model
    - ModelHandlerRegistrationResult: Handler-level registration operation result
    - ModelDiscoveryResult: Discovery operation result

Protocols:
    - ProtocolDiscoveryOperations: Discovery operations protocol definition
"""

from omnibase_infra.handlers.service_discovery.handler_service_discovery_consul import (
    HandlerServiceDiscoveryConsul,
)
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
    "HandlerServiceDiscoveryConsul",
    "HandlerServiceDiscoveryMock",
    "ModelDiscoveryResult",
    "ModelHandlerRegistrationResult",
    "ModelServiceInfo",
    "ProtocolDiscoveryOperations",
]
