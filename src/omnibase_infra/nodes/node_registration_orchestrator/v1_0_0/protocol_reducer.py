# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for the reducer that computes registration intents.

The reducer implements a pure function pattern: given the current state
and an incoming event, it returns updated state plus a list of typed
intents describing what infrastructure operations should occur.

Protocol Responsibilities:
    - Compute intents from introspection events
    - Maintain reducer state for deduplication and rate limiting
    - Filter duplicate or invalid events

Design Notes:
    - Uses @runtime_checkable for duck typing support
    - Type hints use forward references to avoid circular imports
    - The orchestrator owns the workflow; reducer is pluggable

Related Modules:
    - omnibase_infra.models.registration: Event models
    - OMN-889: Reducer implementation (pending)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_reducer_state import (
    ModelReducerState,
)
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_registration_intent import (
    ModelRegistrationIntent,
)

if TYPE_CHECKING:
    from omnibase_infra.models.registration import ModelNodeIntrospectionEvent


@runtime_checkable
class ProtocolReducer(Protocol):
    """Protocol for the reducer that computes registration intents.

    The reducer implements a pure function pattern: given the current state
    and an incoming event, it returns updated state plus a list of typed
    intents describing what infrastructure operations should occur.

    Contract:
        - Reducer MUST be deterministic (same inputs -> same outputs)
        - Reducer MUST NOT perform I/O operations
        - Reducer MUST return valid intents that the effect node can execute
        - Reducer MAY filter duplicate or invalid events

    Example:
        ```python
        class MyReducer:
            async def reduce(
                self,
                state: ModelReducerState,
                event: ModelNodeIntrospectionEvent,
            ) -> tuple[ModelReducerState, list[ModelRegistrationIntent]]:
                # Validate event
                if event.node_id in state.processed_node_ids:
                    return state, []  # Already processed, no-op

                # Generate intents
                intents = [
                    ModelRegistrationIntent(
                        kind="consul",
                        operation="register",
                        node_id=event.node_id,
                        correlation_id=event.correlation_id or uuid4(),
                        payload={"service_name": f"node-{event.node_type}"},
                    ),
                    ModelRegistrationIntent(
                        kind="postgres",
                        operation="upsert",
                        node_id=event.node_id,
                        correlation_id=event.correlation_id or uuid4(),
                        payload=event.model_dump(),
                    ),
                ]

                # Update state
                new_state = ModelReducerState(
                    last_event_timestamp=event.timestamp.isoformat(),
                    processed_node_ids=state.processed_node_ids | {event.node_id},
                    pending_registrations=state.pending_registrations + len(intents),
                )

                return new_state, intents
        ```
    """

    async def reduce(
        self,
        state: ModelReducerState,
        event: ModelNodeIntrospectionEvent,
    ) -> tuple[ModelReducerState, list[ModelRegistrationIntent]]:
        """Reduce an introspection event to state and intents.

        This method processes an incoming introspection event and produces:
        1. Updated reducer state (for deduplication, rate limiting, etc.)
        2. A list of intents describing infrastructure operations to perform

        Args:
            state: Current reducer state. Pass ModelReducerState.initial()
                for the first reduction.
            event: The introspection event to process. Contains node metadata,
                capabilities, and endpoints.

        Returns:
            A tuple of (new_state, intents) where:
                - new_state: Updated reducer state to pass to the next reduction
                - intents: List of intents for the effect node to execute.
                  May be empty if the event should be filtered.

        Raises:
            ValueError: If the event is malformed or cannot be processed.

        Note:
            This method is async to allow for potential future enhancements
            (e.g., async validation), but implementations MUST NOT perform
            actual I/O operations.
        """
        ...


__all__ = ["ProtocolReducer"]
