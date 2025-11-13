"""
Persistence Layer for OmniNode Bridge.

High-level CRUD operations with transaction support for bridge nodes.

This module provides convenience wrapper functions around the generic
CRUD handlers in the database_adapter_effect node, offering:

- Type-safe entity operations
- Comprehensive error handling with OnexError
- UUID correlation tracking
- Transaction management patterns
- Performance logging and metrics

ONEX v2.0 Compliance:
- Uses EntityRegistry for type-safe operations
- OnexError exception handling with context
- UUID correlation tracking
- Performance monitoring

Modules:
    - bridge_state_crud: CRUD operations for ModelBridgeState
    - workflow_execution_crud: CRUD operations for ModelWorkflowExecution

Usage Example:
    >>> from omninode_bridge.persistence import (
    ...     create_bridge_state,
    ...     create_workflow_execution,
    ...     update_workflow_execution
    ... )
    >>> from uuid import uuid4
    >>> from datetime import datetime, UTC
    >>>
    >>> # Create workflow
    >>> workflow = await create_workflow_execution(
    ...     correlation_id=uuid4(),
    ...     workflow_type="metadata_stamping",
    ...     current_state="PENDING",
    ...     namespace="production",
    ...     node=db_node,
    ...     request_correlation_id=uuid4()
    ... )
    >>>
    >>> # Update to PROCESSING
    >>> await update_workflow_execution(
    ...     correlation_id=workflow.correlation_id,
    ...     updates={
    ...         "current_state": "PROCESSING",
    ...         "started_at": datetime.now(UTC)
    ...     },
    ...     node=db_node,
    ...     request_correlation_id=uuid4()
    ... )
    >>>
    >>> # Create bridge state
    >>> bridge = await create_bridge_state(
    ...     bridge_id=uuid4(),
    ...     namespace="production",
    ...     current_fsm_state="IDLE",
    ...     node=db_node,
    ...     correlation_id=uuid4()
    ... )
"""

# Bridge State CRUD operations
from omninode_bridge.persistence.bridge_state_crud import (
    create_bridge_state,
    delete_bridge_state,
    get_bridge_state,
    list_bridge_states,
    update_bridge_state,
    upsert_bridge_state,
)

# Workflow Execution CRUD operations
from omninode_bridge.persistence.workflow_execution_crud import (
    create_workflow_execution,
    delete_workflow_execution,
    get_workflow_execution,
    list_workflow_executions,
    update_workflow_execution,
)

__all__ = [
    # Bridge State operations
    "create_bridge_state",
    "update_bridge_state",
    "get_bridge_state",
    "list_bridge_states",
    "delete_bridge_state",
    "upsert_bridge_state",
    # Workflow Execution operations
    "create_workflow_execution",
    "update_workflow_execution",
    "get_workflow_execution",
    "list_workflow_executions",
    "delete_workflow_execution",
]
