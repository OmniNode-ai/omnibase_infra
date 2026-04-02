# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for chain verification FSM state transitions.

This is a REDUCER handler -- pure state transitions, no I/O.
Implements delta(state, event) -> (new_state, intents[]).
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_orchestrator.models import EnumChainVerifyState

logger = logging.getLogger(__name__)

# Valid transitions: (from_state, trigger) -> to_state
_TRANSITIONS: dict[tuple[str, str], EnumChainVerifyState] = {
    ("pending", "cmd_received"): EnumChainVerifyState.RETRIEVING,
    ("retrieving", "retrieval_hit"): EnumChainVerifyState.REPLAYING,
    ("retrieving", "retrieval_miss"): EnumChainVerifyState.EXPLORING,
    ("replaying", "replay_complete"): EnumChainVerifyState.VERIFYING,
    ("exploring", "explore_complete"): EnumChainVerifyState.VERIFYING,
    ("verifying", "verify_success"): EnumChainVerifyState.COMPLETE,
    ("verifying", "verify_failed"): EnumChainVerifyState.FALLBACK,
    ("fallback", "fallback_to_explore"): EnumChainVerifyState.EXPLORING,
    # Error transitions from any non-terminal state
    ("retrieving", "error"): EnumChainVerifyState.FAILED,
    ("replaying", "error"): EnumChainVerifyState.FAILED,
    ("exploring", "error"): EnumChainVerifyState.FAILED,
    ("verifying", "error"): EnumChainVerifyState.FAILED,
    ("fallback", "error"): EnumChainVerifyState.FAILED,
}

_TERMINAL_STATES = frozenset(
    {EnumChainVerifyState.COMPLETE, EnumChainVerifyState.FAILED}
)


class HandlerChainVerify:
    """Pure FSM state transition handler for chain verification."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def delta(
        self,
        current_state: str,
        trigger: str,
    ) -> tuple[EnumChainVerifyState, list[str]]:
        """Compute state transition.

        Args:
            current_state: Current FSM state name.
            trigger: Event trigger name.

        Returns:
            Tuple of (new_state, intents). Intents are empty for this FSM
            since the orchestrator handles dispatching.

        Raises:
            ValueError: If the transition is invalid.
        """
        current = EnumChainVerifyState(current_state)

        if current in _TERMINAL_STATES:
            raise ValueError(f"Cannot transition from terminal state '{current_state}'")

        key = (current_state, trigger)
        new_state = _TRANSITIONS.get(key)

        if new_state is None:
            raise ValueError(
                f"Invalid transition: ({current_state}, {trigger}). "
                f"Valid triggers from '{current_state}': "
                f"{[t for (s, t) in _TRANSITIONS if s == current_state]}"
            )

        logger.info(
            "FSM transition: %s -[%s]-> %s",
            current_state,
            trigger,
            new_state.value,
        )

        return new_state, []
