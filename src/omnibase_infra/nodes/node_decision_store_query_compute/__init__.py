# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeDecisionStoreQueryCompute — cursor-paginated scope query node.

Queries decisions from the decision_store table with cursor-based pagination
and flexible scope filtering (ANY/ALL/EXACT modes for scope_services).

Architecture:
    - node.py: Declarative node shell extending NodeCompute
    - handlers/: Cursor-paginated query handler with dynamic WHERE
    - models/: Payload (filters+cursor) and result (decisions+cursor) models
    - registry/: Infrastructure registry for dependency injection
    - contract.yaml: Capability definitions and IO routing

Node Type: COMPUTE_GENERIC
Capability: decision_store.query_decisions

Supported filter parameters:
    - domain: scope_domain exact match
    - layer: scope_layer exact match
    - decision_type: OR-filtered type list
    - tags: ALL-required tag list
    - epic_id: exact match
    - status: lifecycle status filter (default: ACTIVE)
    - scope_services + scope_services_mode: ANY/ALL/EXACT filtering

Related:
    - OMN-2767: NodeDecisionStoreQueryCompute implementation
    - OMN-2765: NodeDecisionStoreEffect (write path)
    - OMN-2764: DB migrations (decision_store table + indexes)
    - OMN-2763: omnibase_core decision store models
"""

from omnibase_infra.nodes.node_decision_store_query_compute.handlers import (
    HandlerQueryDecisions,
    decode_cursor,
    encode_cursor,
)
from omnibase_infra.nodes.node_decision_store_query_compute.models import (
    ModelPayloadQueryDecisions,
    ModelResultDecisionList,
)
from omnibase_infra.nodes.node_decision_store_query_compute.node import (
    NodeDecisionStoreQueryCompute,
)
from omnibase_infra.nodes.node_decision_store_query_compute.registry import (
    RegistryInfraDecisionStoreQuery,
)

__all__: list[str] = [
    # Node
    "NodeDecisionStoreQueryCompute",
    # Handlers
    "HandlerQueryDecisions",
    # Cursor utilities
    "encode_cursor",
    "decode_cursor",
    # Models
    "ModelPayloadQueryDecisions",
    "ModelResultDecisionList",
    # Registry
    "RegistryInfraDecisionStoreQuery",
]
