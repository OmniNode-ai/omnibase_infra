# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Declarative EFFECT node for Infisical provisioning and seeding.

Bootstraps the Infisical secret store via provision + seed scripts as
part of the setup orchestration workflow.

Ticket: OMN-3494
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeInfisicalSetupEffect(NodeEffect):
    """Declarative effect node for Infisical provisioning and seeding.

    Handlers:
        - ``HandlerInfisicalFullSetup``: Runs provision-infisical.py then
          seed-infisical.py to bootstrap the Infisical secret store.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the Infisical setup effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeInfisicalSetupEffect"]
