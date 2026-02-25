# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeDecisionStoreQueryCompute — cursor-paginated scope query.

This compute node queries decisions from the decision_store table with
cursor-based pagination and flexible scope filtering.

Follows the ONEX declarative pattern:
    - DECLARATIVE compute driven by contract.yaml
    - Zero custom logic -- all behavior from handlers
    - Lightweight shell that delegates to handler implementations

Handlers:
    - HandlerQueryDecisions: Cursor-paginated decision store query with
      ANY/ALL/EXACT scope_services_mode semantics.

Design Decisions:
    - Node extends NodeCompute per the ticket spec (DB-DECISION-04).
    - The handler is EFFECT-classified (performs DB I/O) even though
      the node shell is NodeCompute.
    - Cursor encoding: base64(created_at_iso|decision_id) is stable
      under concurrent writes.
    - scope_services=None or [] returns all decisions (platform-wide).

Related:
    - contract.yaml: Capability definitions and IO operations
    - models/: Query payload and paginated result models
    - handlers/: Cursor-paginated query handler
    - registry/: DI registry

Tracking:
    - OMN-2767: DB-DECISION-04 — NodeDecisionStoreQueryCompute
    - OMN-2765: NodeDecisionStoreEffect (write path dependency)
    - OMN-2764: DB migrations (decision_store table and indexes)
    - OMN-2763: omnibase_core decision store models
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeDecisionStoreQueryCompute(NodeCompute):
    """Compute node for cursor-paginated decision store queries.

    Capability: decision_store.query_decisions

    Queries decisions from the decision_store table with flexible filter
    parameters (domain, layer, decision_type, tags, epic_id, status,
    scope_services) and cursor-based pagination for stable traversal
    under concurrent writes.

    All behavior is defined in contract.yaml and implemented through
    handlers. No custom logic exists in this class.

    Attributes:
        container: ONEX dependency injection container.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the decision store query compute node.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)


__all__: list[str] = ["NodeDecisionStoreQueryCompute"]
