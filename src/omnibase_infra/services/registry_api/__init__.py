# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry API Service Module.

Provides a FastAPI-based HTTP API for registry discovery operations,
exposing node registrations for dashboard consumption.

This module bridges the existing ProjectionReaderRegistration service
with a REST API layer. Consul instance discovery was removed (OMN-9545).

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from omnibase_infra.services.registry_api.main import create_app
from omnibase_infra.services.registry_api.models import (
    ModelPaginationInfo,
    ModelRegistryDiscoveryResponse,
    ModelRegistryHealthResponse,
    ModelRegistryNodeView,
    ModelRegistrySummary,
    ModelWarning,
    ModelWidgetMapping,
)
from omnibase_infra.services.registry_api.registry_discovery import (
    ServiceRegistryDiscovery,
)

__all__ = [
    "create_app",
    "ModelPaginationInfo",
    "ModelRegistryDiscoveryResponse",
    "ModelRegistryHealthResponse",
    "ModelRegistryNodeView",
    "ModelRegistrySummary",
    "ModelWarning",
    "ModelWidgetMapping",
    "ServiceRegistryDiscovery",
]
