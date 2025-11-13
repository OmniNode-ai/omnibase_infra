#!/usr/bin/env python3
"""
Intent Types for Pure Reducer Operations.

Defines intent types that represent side effects to be performed
by EFFECT nodes, following ONEX v2.0 pure function architecture.

ONEX v2.0 Compliance:
- Enum-based naming: EnumIntentType
- Intent type definitions for COMPUTE → EFFECT separation
- Integration with Intent Publisher pattern
"""

from enum import Enum


class EnumIntentType(str, Enum):
    """
    Intent types for side effects in pure reducer.

    Intent Flow:
    1. COMPUTE node (reducer) emits intents during pure computation
    2. Orchestrator collects intents from output state
    3. EFFECT nodes consume intents and perform I/O operations

    Intent Categories:
    - Event Publishing: PublishEvent
    - State Persistence: PersistState, PersistFSMTransition
    - FSM Operations: RecoverFSMStates
    """

    PUBLISH_EVENT = "PublishEvent"
    """Publish event to event bus (Kafka)."""

    PERSIST_STATE = "PersistState"
    """Persist aggregated state to store (PostgreSQL)."""

    PERSIST_FSM_TRANSITION = "PersistFSMTransition"
    """Persist FSM state transition to store (PostgreSQL)."""

    RECOVER_FSM_STATES = "RecoverFSMStates"
    """Recover FSM states from store (PostgreSQL)."""

    def get_target_node_type(self) -> str:
        """
        Get the target node type for this intent.

        Returns:
            Node type that should handle this intent (effect, store_effect, event_bus)

        Example:
            >>> EnumIntentType.PUBLISH_EVENT.get_target_node_type()
            'event_bus'
        """
        target_map = {
            EnumIntentType.PUBLISH_EVENT: "event_bus",
            EnumIntentType.PERSIST_STATE: "store_effect",
            EnumIntentType.PERSIST_FSM_TRANSITION: "store_effect",
            EnumIntentType.RECOVER_FSM_STATES: "store_effect",
        }
        return target_map[self]

    @property
    def requires_effect_node(self) -> bool:
        """Check if this intent requires an EFFECT node to execute."""
        # All intents require EFFECT nodes (no intents are self-contained)
        return True
