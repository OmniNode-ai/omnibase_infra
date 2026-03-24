# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure-function reducer for node graph lifecycle FSM [OMN-6349].

Follows the canonical pure-function reducer pattern from
node_contract_registry_reducer/reducer.py. The reducer is a stateless class --
all state is passed in and returned. No self.current_state tracking.

FSM Transitions:
    initializing -> wiring      (trigger: registry_ready)
    wiring       -> running     (trigger: wiring_complete)
    running      -> draining    (trigger: drain_requested)
    draining     -> stopped     (trigger: drain_complete)
    ANY          -> stopped     (trigger: fatal_error) -- wildcard

Ticket: OMN-6349
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from omnibase_infra.nodes.node_node_graph_reducer.models.model_node_graph_state import (
    ModelNodeGraphState,
)

logger = logging.getLogger(__name__)

# Valid FSM transitions: (current_state, event_type) -> next_state
_TRANSITIONS: dict[tuple[str, str], str] = {
    ("initializing", "registry_ready"): "wiring",
    ("wiring", "wiring_complete"): "running",
    ("running", "drain_requested"): "draining",
    ("draining", "drain_complete"): "stopped",
}

# Wildcard transitions: event_type -> next_state (from any state)
_WILDCARD_TRANSITIONS: dict[str, str] = {
    "fatal_error": "stopped",
}


@dataclass(frozen=True)
class ModelReducerOutput:
    """Output of a reduce() call: new state + optional intents."""

    result: ModelNodeGraphState
    intents: tuple[dict[str, object], ...] = ()


class NodeGraphReducer:
    """Stateless reducer for node graph lifecycle FSM.

    Pure function: reduce(state, event, metadata) -> ModelReducerOutput.
    No internal state tracking.
    """

    def reduce(
        self,
        state: ModelNodeGraphState,
        event: object,
        event_metadata: dict[
            str, object
        ],  # ONEX_EXCLUDE: dict_str_any - event metadata is protocol-level generic context
    ) -> ModelReducerOutput:
        """Apply an event to the current state, returning new state + intents.

        Args:
            state: Current FSM state (frozen, immutable).
            event: Event object with ``event_type`` attribute.
            event_metadata: Additional metadata about the event.

        Returns:
            New state and any emitted intents.

        Raises:
            ValueError: If the transition is invalid for the current state.
        """
        event_type: str = getattr(event, "event_type", "unknown")

        # Check wildcard transitions first
        if event_type in _WILDCARD_TRANSITIONS:
            next_state = _WILDCARD_TRANSITIONS[event_type]
            error_msg = str(getattr(event, "error_message", event))
            logger.warning(
                "Node graph FSM wildcard transition: %s -> %s (event=%s)",
                state.fsm_state,
                next_state,
                event_type,
            )
            return ModelReducerOutput(
                result=ModelNodeGraphState(
                    fsm_state=next_state,
                    nodes_loaded=state.nodes_loaded,
                    error_message=error_msg if event_type == "fatal_error" else None,
                ),
            )

        # Check normal transitions
        key = (state.fsm_state, event_type)
        if key not in _TRANSITIONS:
            msg = (
                f"Invalid FSM transition: state={state.fsm_state}, "
                f"event={event_type}. "
                f"Valid events from {state.fsm_state}: "
                f"{[k[1] for k in _TRANSITIONS if k[0] == state.fsm_state]}"
            )
            raise ValueError(msg)

        next_state = _TRANSITIONS[key]
        logger.debug(
            "Node graph FSM transition: %s -> %s (event=%s)",
            state.fsm_state,
            next_state,
            event_type,
        )

        return ModelReducerOutput(
            result=ModelNodeGraphState(
                fsm_state=next_state,
                nodes_loaded=state.nodes_loaded,
            ),
        )
