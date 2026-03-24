# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative EFFECT node wrapping RuntimeContractConfigLoader.

Scans the filesystem for contract YAML files and emits discovered contracts
for the contract registry to process. All behavior is defined in contract.yaml
and delegated to handlers.

Ticket: OMN-6347
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeContractLoaderEffect(NodeEffect):
    """Declarative effect node for contract filesystem scanning.

    Handlers:
        - ``HandlerContractScan``: Scans directories for contract YAMLs.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the contract loader effect node."""
        super().__init__(container)
