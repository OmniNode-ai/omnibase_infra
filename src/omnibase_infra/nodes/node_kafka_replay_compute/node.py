# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka replay compute node shell.

All behavior is defined in contract.yaml and implemented by HandlerKafkaReplay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeKafkaReplayCompute(NodeCompute):
    """Declarative compute node for deterministic Kafka replay proofs."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeKafkaReplayCompute"]
