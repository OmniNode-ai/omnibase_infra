# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""NodeTopicMigrationProjection — declarative COMPUTE projection node.

Subscribes to the topic-migration lifecycle event topic and delegates all
projection logic to :class:`HandlerTopicMigrationProjection`, materializing the
``topic_migration_projection`` FSM table.

Subscribed Topic (via contract.yaml):
    - onex.evt.omnibase-infra.topic-migration-lifecycle.v1

Ticket: OMN-12623
"""

from __future__ import annotations

from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes.node_compute import NodeCompute


class NodeTopicMigrationProjection(NodeCompute):
    """Declarative COMPUTE node for topic-migration lifecycle projection.

    All behavior is defined in contract.yaml and delegated to
    HandlerTopicMigrationProjection. No custom logic beyond the DI constructor.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeTopicMigrationProjection"]
