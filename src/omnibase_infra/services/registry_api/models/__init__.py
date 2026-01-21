# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry API Response Models.

Pydantic models for the Registry API HTTP responses. These models define
the JSON shape returned by each endpoint for dashboard consumption.

Design Principles:
    - Flat, dashboard-friendly structures (no deep nesting)
    - Explicit field descriptions for API documentation
    - Immutable (frozen) for thread safety
    - Strict validation (extra="forbid")

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from omnibase_infra.services.registry_api.models.model_capability_widget_mapping import (
    ModelCapabilityWidgetMapping,
)
from omnibase_infra.services.registry_api.models.model_pagination_info import (
    ModelPaginationInfo,
)
from omnibase_infra.services.registry_api.models.model_registry_discovery_response import (
    ModelRegistryDiscoveryResponse,
)
from omnibase_infra.services.registry_api.models.model_registry_health_response import (
    ModelRegistryHealthResponse,
)
from omnibase_infra.services.registry_api.models.model_registry_instance_view import (
    ModelRegistryInstanceView,
)
from omnibase_infra.services.registry_api.models.model_registry_node_view import (
    ModelRegistryNodeView,
)
from omnibase_infra.services.registry_api.models.model_registry_summary import (
    ModelRegistrySummary,
)
from omnibase_infra.services.registry_api.models.model_response_list_instances import (
    ModelResponseListInstances,
)
from omnibase_infra.services.registry_api.models.model_response_list_nodes import (
    ModelResponseListNodes,
)
from omnibase_infra.services.registry_api.models.model_warning import ModelWarning
from omnibase_infra.services.registry_api.models.model_widget_defaults import (
    ModelWidgetDefaults,
)
from omnibase_infra.services.registry_api.models.model_widget_mapping import (
    ModelWidgetMapping,
)

__all__ = [
    "ModelCapabilityWidgetMapping",
    "ModelPaginationInfo",
    "ModelRegistryDiscoveryResponse",
    "ModelRegistryHealthResponse",
    "ModelRegistryInstanceView",
    "ModelRegistryNodeView",
    "ModelRegistrySummary",
    "ModelResponseListInstances",
    "ModelResponseListNodes",
    "ModelWarning",
    "ModelWidgetDefaults",
    "ModelWidgetMapping",
]
