# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry API Service Module.

Provides a FastAPI-based HTTP API for registry discovery operations,
exposing node registrations and live Consul instances for dashboard
consumption.

This module bridges the existing ProjectionReaderRegistration and
HandlerServiceDiscoveryConsul services with a REST API layer.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from omnibase_infra.services.registry_api.main import create_app
from omnibase_infra.services.registry_api.models import (
    ModelPaginationInfo,
    ModelRegistryDiscoveryResponse,
    ModelRegistryHealthResponse,
    ModelRegistryInstanceView,
    ModelRegistryNodeView,
    ModelRegistrySummary,
    ModelWarning,
    ModelWidgetMapping,
)
from omnibase_infra.services.registry_api.service import ServiceRegistryDiscovery

__all__ = [
    "create_app",
    "ModelPaginationInfo",
    "ModelRegistryDiscoveryResponse",
    "ModelRegistryHealthResponse",
    "ModelRegistryInstanceView",
    "ModelRegistryNodeView",
    "ModelRegistrySummary",
    "ModelWarning",
    "ModelWidgetMapping",
    "ServiceRegistryDiscovery",
]
