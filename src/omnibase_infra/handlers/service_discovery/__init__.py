# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Discovery Handlers Module.

This module provides pluggable handler implementations for service discovery
operations, supporting the capability-oriented node architecture.

Handlers:
    - ConsulServiceDiscoveryHandler: Consul-backed service discovery
    - MockServiceDiscoveryHandler: In-memory mock for testing

Models:
    - ModelServiceInfo: Service information model
    - ModelHandlerRegistrationResult: Handler-level registration operation result
    - ModelDiscoveryResult: Discovery operation result

Protocols:
    - ProtocolServiceDiscoveryHandler: Handler protocol definition
"""

from omnibase_infra.handlers.service_discovery.handler_consul_service_discovery import (
    ConsulServiceDiscoveryHandler,
)
from omnibase_infra.handlers.service_discovery.handler_mock_service_discovery import (
    MockServiceDiscoveryHandler,
)
from omnibase_infra.handlers.service_discovery.models import (
    ModelDiscoveryResult,
    ModelHandlerRegistrationResult,
    ModelServiceInfo,
)
from omnibase_infra.handlers.service_discovery.protocol_service_discovery_handler import (
    ProtocolServiceDiscoveryHandler,
)

__all__: list[str] = [
    "ConsulServiceDiscoveryHandler",
    "MockServiceDiscoveryHandler",
    "ModelDiscoveryResult",
    "ModelHandlerRegistrationResult",
    "ModelServiceInfo",
    "ProtocolServiceDiscoveryHandler",
]
