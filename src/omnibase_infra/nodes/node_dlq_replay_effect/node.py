# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""NodeDlqReplayEffect — contract-native DLQ replay + quarantine (OMN-12619).

Declarative EFFECT shell for replaying Dead Letter Queue messages back to their
original topics and quarantining non-replayable messages to
``onex.dlq.omnibase-infra.quarantine.v1`` instead of dropping them.

All behaviour is defined in contract.yaml; this class contains no custom logic.

Architecture:
    onex.dlq.<category>.v1 (Kafka, persistent group onex-dlq-replay)
        -> NodeDlqReplayEffect (this declarative shell)
        -> HandlerDlqReplay
        -> original topic (replay)         when should_replay() is True
        -> onex.dlq.omnibase-infra.quarantine.v1          when should_replay() is False
        -> dlq_replay_history (PostgreSQL) for every terminal outcome

Related Tickets:
    - OMN-12619: contract-native DLQ replay node + quarantine
    - OMN-12618: Divergent Transport & Runtime Convergence (epic)
"""

from __future__ import annotations

from omnibase_core.models.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeDlqReplayEffect(NodeEffect):
    """Declarative effect node for DLQ replay and quarantine.

    Pure declarative shell — all routing/execution is contract-driven.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeDlqReplayEffect"]
