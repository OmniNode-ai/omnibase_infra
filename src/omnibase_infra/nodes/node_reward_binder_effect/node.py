# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Declarative EFFECT node for structured reward event emission to Kafka.

After ``ScoringReducer`` produces an ``EvaluationResult``, the
``RewardBinder`` emits three structured events to Kafka:

  1. ``RunEvaluatedEvent``       → run summary with tamper-evident
     ``objective_fingerprint``
  2. ``RewardAssignedEvent``     → per-target reward with traceable
     ``evidence_refs``
  3. ``PolicyStateUpdatedEvent`` → policy state transition snapshot

This is the only node in the objective pipeline that performs I/O.
No scoring logic lives here — this node only emits what
``ScoringReducer`` produced.

All behaviour is defined in ``contract.yaml`` and delegated to
``HandlerRewardBinder``. No custom logic in this class.

Ticket: OMN-2552
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeRewardBinderEffect(NodeEffect):
    """Declarative effect node for reward event emission.

    Handlers:
        - ``HandlerRewardBinder``: Emits RunEvaluatedEvent,
          RewardAssignedEvent (one per target), and
          PolicyStateUpdatedEvent to their respective Kafka topics.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialise the reward binder effect node.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)


__all__: list[str] = ["NodeRewardBinderEffect"]
