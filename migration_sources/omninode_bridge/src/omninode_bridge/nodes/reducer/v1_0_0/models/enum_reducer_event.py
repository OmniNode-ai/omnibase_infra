#!/usr/bin/env python3
"""
Reducer Event Types for NodeBridgeReducer.

Defines Kafka event types published during aggregation and state reduction.
Part of the O.N.E. v0.1 compliant bridge architecture.

ONEX v2.0 Compliance:
- Enum-based naming: EnumReducerEvent
- Event type definitions for Kafka publishing
- Integration with ModelEventTypeSubcontract
"""

from enum import Enum


class EnumReducerEvent(str, Enum):
    """
    Kafka event types for metadata aggregation and state reduction.

    Event Publishing Flow:
    1. AGGREGATION_STARTED: Published when aggregation begins
    2. BATCH_PROCESSED: Published after each batch is aggregated
    3. STATE_PERSISTED: Published when state is saved to PostgreSQL
    4. AGGREGATION_COMPLETED: Published on successful aggregation completion
    5. AGGREGATION_FAILED: Published on error during aggregation
    6. FSM_STATE_INITIALIZED: Published when workflow FSM state is initialized
    7. FSM_STATE_TRANSITIONED: Published on FSM state transitions

    Usage:
        Used by NodeBridgeReducer to publish events via Kafka
        using ModelEventTypeSubcontract configuration.
    """

    AGGREGATION_STARTED = "aggregation_started"
    """Published when aggregation workflow begins."""

    BATCH_PROCESSED = "batch_processed"
    """Published after each batch of metadata is aggregated."""

    STATE_PERSISTED = "state_persisted"
    """Published when aggregated state is persisted to PostgreSQL."""

    AGGREGATION_COMPLETED = "aggregation_completed"
    """Published when aggregation completes successfully."""

    AGGREGATION_FAILED = "aggregation_failed"
    """Published when aggregation fails."""

    FSM_STATE_INITIALIZED = "fsm_state_initialized"
    """Published when a workflow is initialized in the FSM."""

    FSM_STATE_TRANSITIONED = "fsm_state_transitioned"
    """Published when a workflow transitions between FSM states."""

    # Node Introspection Events (shared with orchestrator)
    NODE_INTROSPECTION = "node_introspection"
    """Published when node broadcasts introspection data (capabilities, endpoints, metadata)."""

    REGISTRY_REQUEST_INTROSPECTION = "registry_request_introspection"
    """Published when registry requests all nodes to broadcast introspection data."""

    NODE_HEARTBEAT = "node_heartbeat"
    """Published periodically by nodes to indicate health and availability."""

    def get_topic_name(self, namespace: str = "dev") -> str:
        """
        Generate Kafka topic name for this event using standardized ONEX format.

        Args:
            namespace: Environment namespace (default: dev)

        Returns:
            Fully qualified Kafka topic name

        Example:
            >>> EnumReducerEvent.AGGREGATION_STARTED.get_topic_name()
            'dev.omninode_bridge.onex.evt.aggregation-started.v1'
        """
        # Convert event value to match topic naming (replace underscores with hyphens)
        event_slug = self.value.replace("_", "-")
        return f"{namespace}.omninode_bridge.onex.evt.{event_slug}.v1"

    @property
    def is_terminal_event(self) -> bool:
        """Check if this event indicates aggregation termination."""
        return self in (
            EnumReducerEvent.AGGREGATION_COMPLETED,
            EnumReducerEvent.AGGREGATION_FAILED,
        )
