# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Declarative EFFECT node for local Docker Compose service provisioning.

Owns all `docker compose` operations (up, down, status) for local services
defined in a ModelDeploymentTopology. All behavior is defined in contract.yaml
and delegated to handlers.

Ticket: OMN-3493
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeLocalProvisionEffect(NodeEffect):
    """Declarative effect node for local service provisioning.

    Handlers:
        - ``HandlerLocalProvision``: Runs ``docker compose up -d`` and polls ports.
        - ``HandlerLocalTeardown``: Runs ``docker compose down``.
        - ``HandlerLocalStatus``: Queries which services are currently running.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the local provision effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeLocalProvisionEffect"]
