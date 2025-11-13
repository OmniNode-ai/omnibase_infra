#!/usr/bin/env python3
"""
Database Adapter Effect Node - Input Models Package.

This package contains all input models for the database adapter effect node,
providing type-safe routing and validation for database operations triggered
by Kafka events from bridge nodes.

ONEX v2.0 Compliance:
- All models use Pydantic v2 BaseModel
- Suffix-based naming convention (Model* prefix)
- UUID correlation tracking across all operations
- Comprehensive field validation

Input Models:
- ModelDatabaseOperationInput: Base routing model with operation_type
- ModelWorkflowExecutionInput: Workflow execution persistence
- ModelWorkflowStepInput: Workflow step history tracking
- ModelBridgeStateInput: Bridge state UPSERT operations
- ModelFSMTransitionInput: FSM transition audit trail
- ModelMetadataStampInput: Metadata stamp audit records
- ModelNodeHeartbeatInput: Node heartbeat updates

Usage:
    >>> from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs import (
    ...     ModelDatabaseOperationInput,
    ...     ModelWorkflowExecutionInput,
    ...     ModelBridgeStateInput,
    ... )
    >>> from uuid import uuid4
    >>>
    >>> # Route to workflow execution operation
    >>> operation = ModelDatabaseOperationInput(
    ...     operation_type="persist_workflow_execution",
    ...     correlation_id=uuid4(),
    ...     workflow_execution_data={
    ...         "workflow_type": "metadata_stamping",
    ...         "current_state": "PROCESSING",
    ...         "namespace": "production"
    ...     }
    ... )
"""

from .model_bridge_state_input import ModelBridgeStateInput
from .model_database_operation_input import ModelDatabaseOperationInput
from .model_fsm_transition_input import ModelFSMTransitionInput
from .model_metadata_stamp_input import ModelMetadataStampInput
from .model_node_heartbeat_input import ModelNodeHeartbeatInput
from .model_workflow_execution_input import ModelWorkflowExecutionInput
from .model_workflow_step_input import ModelWorkflowStepInput

__all__ = [
    "ModelDatabaseOperationInput",
    "ModelWorkflowExecutionInput",
    "ModelWorkflowStepInput",
    "ModelBridgeStateInput",
    "ModelFSMTransitionInput",
    "ModelMetadataStampInput",
    "ModelNodeHeartbeatInput",
]
