# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handlers package for NodeRegistryApiEffect.

Stub handlers for all 10 operations declared in ``contract.yaml``.
Each handler satisfies HandlerPluginLoader (class exists, correct type properties)
and raises NotImplementedError until full implementation is added.

Full business logic lives in ``services/registry_api/service.py`` and is
invoked via the FastAPI route layer.

Ticket: OMN-1441 (node), OMN-2909 (stubs)
"""

from __future__ import annotations

from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_contract import (
    HandlerRegistryApiGetContract,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_discovery import (
    HandlerRegistryApiGetDiscovery,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_health import (
    HandlerRegistryApiGetHealth,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_node import (
    HandlerRegistryApiGetNode,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_topic import (
    HandlerRegistryApiGetTopic,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_get_widget_mapping import (
    HandlerRegistryApiGetWidgetMapping,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_list_contracts import (
    HandlerRegistryApiListContracts,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_list_instances import (
    HandlerRegistryApiListInstances,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_list_nodes import (
    HandlerRegistryApiListNodes,
)
from omnibase_infra.nodes.node_registry_api_effect.handlers.handler_registry_api_list_topics import (
    HandlerRegistryApiListTopics,
)

__all__: list[str] = [
    "HandlerRegistryApiGetContract",
    "HandlerRegistryApiGetDiscovery",
    "HandlerRegistryApiGetHealth",
    "HandlerRegistryApiGetNode",
    "HandlerRegistryApiGetTopic",
    "HandlerRegistryApiGetWidgetMapping",
    "HandlerRegistryApiListContracts",
    "HandlerRegistryApiListInstances",
    "HandlerRegistryApiListNodes",
    "HandlerRegistryApiListTopics",
]
