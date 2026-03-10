# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""NodeDeltaBundleEffect -- declarative effect node for delta bundle writes.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom routing logic -- all behavior from handler_routing
    - Lightweight shell that delegates to handlers via container resolution
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for infrastructure I/O operations.
All handler routing is 100% driven by contract.yaml, not Python code.

Handler Routing Pattern:
    1. Receive event with typed payload (gate_decision or pr_outcome)
    2. Route to appropriate handler based on payload.intent_type
    3. Execute PostgreSQL I/O via handler
    4. Return structured ModelBackendResult

Design Decisions:
    - 100% Contract-Driven: All routing logic in YAML, not Python
    - Zero Custom Routing: Base class handles handler dispatch via contract
    - Idempotent Writes: ON CONFLICT DO NOTHING for gate-decision events
    - Fix-PR Detection: Parses stabilizes:<pr_ref> labels before insert

Related:
    - contract.yaml: Handler routing and I/O model definitions
    - handlers/: PostgreSQL operation handlers
    - OMN-3142: NodeDeltaBundleEffect implementation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.models.runtime.model_resolved_dependencies import (
        ModelResolvedDependencies,
    )


# ONEX_EXCLUDE: declarative_node -- OMN-1732 DEC-003 requires constructor injection
# for protocol dependencies. The _resolved_dependencies instance variable stores
# pre-resolved protocols from ContractDependencyResolver.
class NodeDeltaBundleEffect(NodeEffect):
    """Declarative effect node for delta bundle persistence.

    Routes gate-decision events through the idempotent bundle insert handler
    and pr-outcome events through the outcome update handler.

    NO custom routing logic -- all behavior is
    defined in contract.yaml.

    Args:
        container: ONEX dependency injection container.
        dependencies: Optional pre-resolved protocol dependencies. If provided,
            the node will use these instead of resolving from container.
            Part of OMN-1732 runtime dependency injection.
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        dependencies: ModelResolvedDependencies | None = None,
    ) -> None:
        """Initialise effect node with container dependency injection.

        Args:
            container: ONEX dependency injection container.
            dependencies: Optional pre-resolved protocol dependencies from
                ContractDependencyResolver. If provided, the node uses these
                instead of resolving from container. Part of OMN-1732.
        """
        super().__init__(container)
        self._resolved_dependencies = dependencies


__all__ = ["NodeDeltaBundleEffect"]
