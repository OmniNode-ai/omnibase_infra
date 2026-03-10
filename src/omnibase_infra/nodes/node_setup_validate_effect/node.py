# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Declarative EFFECT node for post-provision service health validation.

Validates TCP/HTTP health for all LOCAL-mode services in the topology
after provisioning is complete.

Ticket: OMN-3494
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeSetupValidateEffect(NodeEffect):
    """Declarative effect node for post-provision health validation.

    Handlers:
        - ``HandlerServiceValidate``: Performs TCP or HTTP health checks
          for all LOCAL-mode services defined in the topology.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the setup validate effect node."""
        super().__init__(container)


__all__: list[str] = ["NodeSetupValidateEffect"]
