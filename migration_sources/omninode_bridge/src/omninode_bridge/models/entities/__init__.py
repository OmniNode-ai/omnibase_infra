"""Pydantic entity models for database tables.

This module provides strongly-typed Pydantic models that map to database entities,
enabling type-safe database operations throughout the omninode_bridge system.

Each entity model corresponds to a database table and provides:
- Type validation via Pydantic
- Field-level constraints and validation
- JSON schema generation for API documentation
- Serialization/deserialization for database operations

Entity Models:
    - ModelWorkflowExecution: Workflow execution tracking (workflow_executions table)
    - ModelWorkflowStep: Individual workflow step tracking (workflow_steps table)
    - ModelMetadataStamp: Metadata stamp audit trail (metadata_stamps table)
    - ModelFSMTransition: FSM state transitions (fsm_transitions table)
    - ModelBridgeState: Bridge aggregation state (bridge_states table)
    - ModelNodeHeartbeat: Node registration and health (node_registrations table)
    - ModelNodeHealthMetrics: Node health metrics (node_health_metrics table)

Usage:
    ```python
    from omninode_bridge.models.entities import ModelWorkflowExecution
    from uuid import uuid4
    from datetime import datetime, timezone

    # Create strongly-typed entity
    workflow = ModelWorkflowExecution(
        correlation_id=uuid4(),
        workflow_type="metadata_stamping",
        current_state="PROCESSING",
        namespace="test_app",
        started_at=datetime.now(timezone.utc)
    )

    # Validation happens automatically
    print(workflow.correlation_id)  # Type-safe access
    ```
"""

from typing import Union

from .model_bridge_state import ModelBridgeState
from .model_fsm_transition import ModelFSMTransition
from .model_metadata_stamp import ModelMetadataStamp
from .model_node_health_metrics import ModelNodeHealthMetrics
from .model_node_heartbeat import ModelNodeHeartbeat
from .model_workflow_execution import ModelWorkflowExecution
from .model_workflow_step import ModelWorkflowStep

# Union type for strongly-typed entity validation
EntityUnion = Union[
    ModelWorkflowExecution,
    ModelWorkflowStep,
    ModelMetadataStamp,
    ModelFSMTransition,
    ModelBridgeState,
    ModelNodeHeartbeat,
    ModelNodeHealthMetrics,
]

__all__ = [
    "ModelBridgeState",
    "ModelFSMTransition",
    "ModelMetadataStamp",
    "ModelNodeHealthMetrics",
    "ModelNodeHeartbeat",
    "ModelWorkflowExecution",
    "ModelWorkflowStep",
    "EntityUnion",
]
