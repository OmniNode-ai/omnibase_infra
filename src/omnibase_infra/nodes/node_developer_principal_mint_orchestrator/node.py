# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative ORCHESTRATOR node that mints or audits a developer's broker principal.

All behavior is defined in contract.yaml and delegated to
``HandlerDeveloperPrincipalMint``, which sequences the plan compute node and the
broker principal effect in process. ``make mint-local`` runs it with ``onex node``
inside the stack's ``runtime-effects`` container.

Ticket: OMN-20823
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeDeveloperPrincipalMintOrchestrator(NodeOrchestrator):
    """Declarative orchestrator for one developer principal mint.

    Handler (``omnibase_infra.handlers``):
        - ``HandlerDeveloperPrincipalMint``: converge (list, plan, apply, verify)
          or audit (list, plan, describe), stopping at the first failure.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the developer principal mint orchestrator."""
        super().__init__(container)


__all__: list[str] = ["NodeDeveloperPrincipalMintOrchestrator"]
