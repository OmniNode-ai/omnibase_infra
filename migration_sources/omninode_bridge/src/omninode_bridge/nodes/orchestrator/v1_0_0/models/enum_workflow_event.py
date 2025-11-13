#!/usr/bin/env python3
"""
Workflow Event Types for NodeBridgeOrchestrator.

Defines Kafka event types published during workflow orchestration.
Part of the O.N.E. v0.1 compliant bridge architecture.

ONEX v2.0 Compliance:
- Enum-based naming: EnumWorkflowEvent
- Event type definitions for Kafka publishing
- Integration with ModelEventTypeSubcontract
"""

from enum import Enum


class EnumWorkflowEvent(str, Enum):
    """
    Kafka event types for stamping workflow orchestration.

    Event Publishing Flow:
    1. WORKFLOW_STARTED: Published when workflow begins (PENDING → PROCESSING)
    2. WORKFLOW_COMPLETED: Published on successful completion (PROCESSING → COMPLETED)
    3. WORKFLOW_FAILED: Published on error (PROCESSING → FAILED)
    4. STEP_COMPLETED: Published after each workflow step completes
    5. INTELLIGENCE_REQUESTED: Published when OnexTree analysis is requested
    6. STAMP_CREATED: Published when metadata stamp is generated

    Usage:
        Used by NodeBridgeOrchestrator to publish events via Kafka
        using ModelEventTypeSubcontract configuration.
    """

    WORKFLOW_STARTED = "stamp_workflow_started"
    """Published when workflow execution begins."""

    WORKFLOW_COMPLETED = "stamp_workflow_completed"
    """Published when workflow completes successfully."""

    WORKFLOW_FAILED = "stamp_workflow_failed"
    """Published when workflow execution fails."""

    STEP_COMPLETED = "workflow_step_completed"
    """Published after each workflow step completes."""

    INTELLIGENCE_REQUESTED = "onextree_intelligence_requested"
    """Published when OnexTree AI intelligence analysis is requested."""

    INTELLIGENCE_RECEIVED = "onextree_intelligence_received"
    """Published when OnexTree AI intelligence analysis completes."""

    STAMP_CREATED = "metadata_stamp_created"
    """Published when BLAKE3 metadata stamp is successfully created."""

    HASH_GENERATED = "blake3_hash_generated"
    """Published when BLAKE3 hash generation completes."""

    STATE_TRANSITION = "workflow_state_transition"
    """Published on FSM state transitions."""

    # Node Introspection Events
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
            >>> EnumWorkflowEvent.WORKFLOW_STARTED.get_topic_name()
            'dev.omninode_bridge.onex.evt.stamp-workflow-started.v1'
        """
        # Convert event value to match topic naming (replace underscores with hyphens)
        event_slug = self.value.replace("_", "-")
        return f"{namespace}.omninode_bridge.onex.evt.{event_slug}.v1"

    @property
    def is_terminal_event(self) -> bool:
        """Check if this event indicates workflow termination."""
        return self in (
            EnumWorkflowEvent.WORKFLOW_COMPLETED,
            EnumWorkflowEvent.WORKFLOW_FAILED,
        )
