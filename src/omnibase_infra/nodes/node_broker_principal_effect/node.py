# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative EFFECT node that creates, grants and verifies a developer principal.

All behavior is defined in contract.yaml and delegated to the three handlers.

Ticket: OMN-20822
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeBrokerPrincipalEffect(NodeEffect):
    """Declarative effect node for a developer principal on a broker.

    Handlers (``omnibase_infra.handlers``):
        - ``HandlerBrokerPrincipalList``: the principals under a prefix (Admin API check).
        - ``HandlerBrokerPrincipalApply``: drift check, ACLs, read-back, then password.
        - ``HandlerBrokerPrincipalVerify``: ACLs against the plan, then a login.
        - ``HandlerBrokerPrincipalDescribe``: the principal's ACLs, read only (OMN-20823).
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the broker principal effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeBrokerPrincipalEffect"]
