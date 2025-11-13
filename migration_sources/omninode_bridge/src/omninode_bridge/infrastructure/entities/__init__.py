"""
Entity models for database table representations.

This module contains Pydantic models that directly map to PostgreSQL tables.
These are used with the EntityRegistry for type-safe database operations.

Entity Models:
- ModelWorkflowExecution: workflow_executions table
- ModelWorkflowStep: workflow_steps table
- ModelMetadataStamp: metadata_stamps table
- ModelFSMTransition: fsm_transitions table
- ModelBridgeState: bridge_states table
- ModelNodeHeartbeat: node_registrations table
- ModelNodeHealthMetrics: node_health_metrics table
- ModelWorkflowState: workflow_state table (Pure Reducer Refactor)
- ModelWorkflowProjection: workflow_projection table (Pure Reducer Refactor)
- ModelProjectionWatermark: projection_watermarks table (Pure Reducer Refactor)
- ModelActionDedup: action_dedup_log table (Pure Reducer Refactor)
- ModelStateCommittedEvent: StateCommitted event (Pure Reducer Refactor)

ONEX v2.0 Compliance:
- Suffix-based naming convention
- Strong typing with Pydantic v2
- Comprehensive field validation
- Database schema mapping
"""

from .model_action_dedup import ModelActionDedup
from .model_bridge_state import ModelBridgeState
from .model_fsm_transition import ModelFSMTransition
from .model_metadata_stamp import ModelMetadataStamp
from .model_node_health_metrics import ModelNodeHealthMetrics
from .model_node_heartbeat import ModelNodeHeartbeat
from .model_projection_watermark import ModelProjectionWatermark
from .model_state_committed_event import ModelStateCommittedEvent
from .model_workflow_execution import ModelWorkflowExecution
from .model_workflow_projection import ModelWorkflowProjection
from .model_workflow_state import ModelWorkflowState
from .model_workflow_step import ModelWorkflowStep

__all__ = [
    "ModelWorkflowExecution",
    "ModelWorkflowStep",
    "ModelMetadataStamp",
    "ModelFSMTransition",
    "ModelBridgeState",
    "ModelNodeHeartbeat",
    "ModelNodeHealthMetrics",
    "ModelWorkflowState",
    "ModelWorkflowProjection",
    "ModelProjectionWatermark",
    "ModelActionDedup",
    "ModelStateCommittedEvent",
]
