"""
Entity models for database adapter effect node.

Strongly-typed Pydantic models representing database entities.
Each entity model maps directly to a PostgreSQL table and includes
all fields including auto-generated database columns (id, created_at, updated_at).

Entity Models:
- ModelWorkflowExecution: Workflow execution records
- ModelWorkflowStep: Workflow step history
- ModelBridgeState: Bridge aggregation state (reducer)
- ModelFSMTransition: FSM state transition history
- ModelMetadataStamp: Metadata stamp audit records
- ModelNodeHeartbeat: Node heartbeat and registration

Entity Union:
- EntityUnion: Union type of all entity models for type-safe operations

Usage:
    >>> from models.entities import ModelWorkflowExecution, EntityUnion
    >>> workflow = ModelWorkflowExecution(
    ...     correlation_id=uuid4(),
    ...     workflow_type="metadata_stamping",
    ...     current_state="PROCESSING",
    ...     namespace="production"
    ... )
"""

from typing import Union

from omninode_bridge.infrastructure.entities.model_bridge_state import ModelBridgeState
from omninode_bridge.infrastructure.entities.model_fsm_transition import (
    ModelFSMTransition,
)
from omninode_bridge.infrastructure.entities.model_metadata_stamp import (
    ModelMetadataStamp,
)
from omninode_bridge.infrastructure.entities.model_node_heartbeat import (
    ModelNodeHeartbeat,
)

# Import infrastructure entities that match database schema (UUID primary keys)
from omninode_bridge.infrastructure.entities.model_workflow_execution import (
    ModelWorkflowExecution,
)
from omninode_bridge.infrastructure.entities.model_workflow_step import (
    ModelWorkflowStep,
)

# Union of all entity types for strongly-typed database operations
EntityUnion = Union[
    ModelWorkflowExecution,
    ModelWorkflowStep,
    ModelBridgeState,
    ModelFSMTransition,
    ModelMetadataStamp,
    ModelNodeHeartbeat,
]

__all__ = [
    # Entity models
    "ModelWorkflowExecution",
    "ModelWorkflowStep",
    "ModelBridgeState",
    "ModelFSMTransition",
    "ModelMetadataStamp",
    "ModelNodeHeartbeat",
    # Union type
    "EntityUnion",
]
