# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""NodeTopicMigrationExecutorEffect — contract-native topic-migration executor.

Declarative EFFECT shell for executing a topic migration described by a
:class:`ModelTopicMigrationContract` (OMN-12621). All behavior lives in
``contract.yaml`` + :class:`HandlerTopicMigrationExecutor`; this class is a pure
shell with no custom routing logic.

Architecture:
    ModelTopicMigrationCommand
        -> NodeTopicMigrationExecutorEffect (this declarative shell)
        -> HandlerTopicMigrationExecutor
        -> provision new topic (TopicProvisioner / ModelTopicSpec)
        -> mint new group (compute_consumer_group_id)
        -> drain-proof gate (ServiceDrainProofGate) on cutover/complete
        -> ModelTopicMigrationLifecycleEvent (durable lifecycle log)

Related Tickets:
    - OMN-12623: migration executor + lag/drain + status projection + replay harness
    - OMN-12621: ModelTopicMigrationContract (core)
"""

from __future__ import annotations

from omnibase_core.models.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeTopicMigrationExecutorEffect(NodeEffect):
    """Declarative effect node for executing topic migrations.

    Lightweight shell defining the I/O contract for topic-migration execution.
    All routing and execution logic is driven by contract.yaml — this class
    contains NO custom routing code.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

    # Pure declarative shell — all behaviour defined in contract.yaml


__all__ = ["NodeTopicMigrationExecutorEffect"]
