# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative coding-agent FSM REDUCER node (OMN-13247).

Owns the deterministic FSM, retry policy, circuit breaker, replay safety, and
the correlation-trace projection. Pure ``delta(state, event) ->
(state, intents[])`` with no I/O. contract.yaml + the handler drive behavior; the
node body stays empty per the declarative-node invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_reducer import NodeReducer

if TYPE_CHECKING:
    from omnibase_infra.models.coding_agent.model_coding_agent_fsm_state import (
        ModelCodingAgentFsmState,
    )


class NodeCodingAgentFsmReducer(
    NodeReducer["ModelCodingAgentFsmState", "ModelCodingAgentFsmState"]
):
    """Declarative pure FSM reducer node (zero custom logic)."""

    # Declarative node — all behavior defined in contract.yaml.


__all__: list[str] = ["NodeCodingAgentFsmReducer"]
