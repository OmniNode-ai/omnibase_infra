"""
Entity Type Enum for Database Adapter.

Defines the entity types that can be persisted through the generic
CRUD handlers using EntityRegistry for type-safe operations.

ONEX v2.0 Compliance:
- Enum-based type safety
- Clear entity-to-table mapping
- Extensible for new entity types
"""

from enum import Enum


class EnumEntityType(str, Enum):
    """
    Database entity types for generic CRUD operations.

    Maps entity types to Pydantic models and database tables through
    the EntityRegistry system.
    """

    WORKFLOW_EXECUTION = "workflow_execution"
    """Workflow execution records (workflow_executions table)."""

    WORKFLOW_STEP = "workflow_step"
    """Workflow step history (workflow_steps table)."""

    METADATA_STAMP = "metadata_stamp"
    """Metadata stamp audit records (metadata_stamps table)."""

    FSM_TRANSITION = "fsm_transition"
    """FSM state transition history (fsm_transitions table)."""

    BRIDGE_STATE = "bridge_state"
    """Bridge aggregation state (bridge_states table)."""

    NODE_HEARTBEAT = "node_heartbeat"
    """Node heartbeat and health (node_registrations table)."""

    NODE_HEALTH_METRICS = "node_health_metrics"
    """Node health and performance metrics (node_registrations table - virtual entity)."""
