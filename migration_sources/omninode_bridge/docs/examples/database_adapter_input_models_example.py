#!/usr/bin/env python3
"""
Database Adapter Effect Node - Input Models Usage Examples

This file demonstrates practical usage patterns for all input models,
showing how they route database operations and validate inputs.

ONEX v2.0 Compliance Examples for Phase 1 - Agent 2
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs import (
    ModelBridgeStateInput,
    ModelDatabaseOperationInput,
    ModelFSMTransitionInput,
    ModelMetadataStampInput,
    ModelNodeHeartbeatInput,
    ModelWorkflowExecutionInput,
    ModelWorkflowStepInput,
)


def example_1_workflow_started():
    """Example 1: Workflow started event → persist_workflow_execution"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Workflow Started Event")
    print("=" * 80)

    correlation_id = uuid4()

    # Base operation routing
    operation = ModelDatabaseOperationInput(
        operation_type="persist_workflow_execution",
        correlation_id=correlation_id,
        workflow_execution_data={
            "workflow_type": "metadata_stamping",
            "current_state": "PROCESSING",
            "namespace": "production",
        },
    )

    # Specific workflow execution data
    workflow_data = ModelWorkflowExecutionInput(
        correlation_id=correlation_id,
        workflow_type="metadata_stamping",
        current_state="PROCESSING",
        namespace="production",
        started_at=datetime.now(UTC),
        metadata={
            "source": "api",
            "user_id": "user_123",
            "api_version": "v1",
        },
    )

    print(f"Operation Type: {operation.operation_type}")
    print(f"Correlation ID: {correlation_id}")
    print(f"Workflow Type: {workflow_data.workflow_type}")
    print(f"Current State: {workflow_data.current_state}")
    print(f"Namespace: {workflow_data.namespace}")
    print(f"Started At: {workflow_data.started_at}")
    print(f"Metadata: {workflow_data.metadata}")


def example_2_workflow_step_completed():
    """Example 2: Step completed event → persist_workflow_step"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Workflow Step Completed Event")
    print("=" * 80)

    workflow_id = uuid4()
    correlation_id = uuid4()

    # Hash generation step
    hash_step = ModelWorkflowStepInput(
        workflow_id=workflow_id,
        step_name="generate_blake3_hash",
        step_order=1,
        status="COMPLETED",
        execution_time_ms=2,
        step_data={
            "file_hash": "abc123def456789abcdef0123456789",
            "file_size_bytes": 1024,
            "performance_grade": "A",
        },
    )

    # Stamping step
    stamp_step = ModelWorkflowStepInput(
        workflow_id=workflow_id,
        step_name="stamp_content",
        step_order=2,
        status="COMPLETED",
        execution_time_ms=8,
        step_data={
            "stamp_id": str(uuid4()),
            "stamp_type": "inline",
            "content_length": 500,
        },
    )

    print(f"Workflow ID: {workflow_id}")
    print(f"\nStep 1: {hash_step.step_name}")
    print(f"  - Order: {hash_step.step_order}")
    print(f"  - Status: {hash_step.status}")
    print(f"  - Execution Time: {hash_step.execution_time_ms}ms")
    print(f"  - Data: {hash_step.step_data}")

    print(f"\nStep 2: {stamp_step.step_name}")
    print(f"  - Order: {stamp_step.step_order}")
    print(f"  - Status: {stamp_step.status}")
    print(f"  - Execution Time: {stamp_step.execution_time_ms}ms")
    print(f"  - Data: {stamp_step.step_data}")


def example_3_bridge_state_aggregation():
    """Example 3: Aggregation completed → persist_bridge_state (UPSERT)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Bridge State Aggregation (UPSERT)")
    print("=" * 80)

    bridge_id = uuid4()

    # Initial state
    initial_state = ModelBridgeStateInput(
        bridge_id=bridge_id,
        namespace="production",
        total_workflows_processed=150,
        total_items_aggregated=750,
        aggregation_metadata={
            "file_type_distribution": {"jpeg": 500, "pdf": 250},
            "avg_file_size_bytes": 102400,
        },
        current_fsm_state="aggregating",
        last_aggregation_timestamp=datetime.now(UTC),
    )

    # Updated state (UPSERT will update)
    updated_state = ModelBridgeStateInput(
        bridge_id=bridge_id,  # Same ID
        namespace="production",
        total_workflows_processed=200,  # Incremented
        total_items_aggregated=1000,  # Incremented
        aggregation_metadata={
            "file_type_distribution": {"jpeg": 650, "pdf": 350},
            "avg_file_size_bytes": 105000,
        },
        current_fsm_state="idle",
        last_aggregation_timestamp=datetime.now(UTC),
    )

    print(f"Bridge ID: {bridge_id}")
    print("\nInitial State:")
    print(f"  - Workflows Processed: {initial_state.total_workflows_processed}")
    print(f"  - Items Aggregated: {initial_state.total_items_aggregated}")
    print(f"  - FSM State: {initial_state.current_fsm_state}")

    print("\nUpdated State (UPSERT):")
    print(f"  - Workflows Processed: {updated_state.total_workflows_processed}")
    print(f"  - Items Aggregated: {updated_state.total_items_aggregated}")
    print(f"  - FSM State: {updated_state.current_fsm_state}")


def example_4_fsm_transition():
    """Example 4: State transition event → persist_fsm_transition"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: FSM State Transition Tracking")
    print("=" * 80)

    entity_id = uuid4()

    # Workflow state transitions
    transitions = [
        ModelFSMTransitionInput(
            entity_id=entity_id,
            entity_type="workflow",
            from_state=None,  # Initial state
            to_state="PENDING",
            transition_event="workflow_created",
            transition_data={"namespace": "production"},
        ),
        ModelFSMTransitionInput(
            entity_id=entity_id,
            entity_type="workflow",
            from_state="PENDING",
            to_state="PROCESSING",
            transition_event="workflow_started",
            transition_data={"started_at": str(datetime.now(UTC))},
        ),
        ModelFSMTransitionInput(
            entity_id=entity_id,
            entity_type="workflow",
            from_state="PROCESSING",
            to_state="COMPLETED",
            transition_event="workflow_completed",
            transition_data={
                "execution_time_ms": 1234,
                "steps_completed": 5,
                "items_processed": 10,
            },
        ),
    ]

    print(f"Entity ID: {entity_id}")
    print("Entity Type: workflow")
    print("\nState Transition Timeline:")

    for i, transition in enumerate(transitions, 1):
        print(f"  {i}. {transition.from_state or 'INITIAL'} → {transition.to_state}")
        print(f"     Event: {transition.transition_event}")
        print(f"     Data: {transition.transition_data}")


def example_5_metadata_stamp_audit():
    """Example 5: Stamp created event → persist_metadata_stamp"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Metadata Stamp Audit Trail")
    print("=" * 80)

    workflow_id = uuid4()

    stamp = ModelMetadataStampInput(
        workflow_id=workflow_id,
        file_hash="a" * 64,  # Valid 64-char hex hash
        stamp_data={
            "stamp_type": "inline",
            "stamp_position": "header",
            "file_metadata": {
                "file_size_bytes": 1024,
                "content_type": "image/jpeg",
            },
            "stamping_metadata": {
                "service": "metadata-stamping",
                "version": "1.0.0",
                "execution_time_ms": 2,
            },
            "onex_compliance": {
                "protocol_version": "1.0",
                "metadata_version": "0.1",
            },
        },
        namespace="production",
    )

    print(f"Workflow ID: {workflow_id}")
    print(f"File Hash: {stamp.file_hash}")
    print(f"Namespace: {stamp.namespace}")
    print(f"Stamp Data: {stamp.stamp_data}")
    print(f"Created At: {stamp.created_at}")


def example_6_node_heartbeat():
    """Example 6: Node heartbeat event → update_node_heartbeat"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Node Heartbeat Update")
    print("=" * 80)

    # Healthy node
    healthy_heartbeat = ModelNodeHeartbeatInput(
        node_id="orchestrator-01",
        health_status="HEALTHY",
        metadata={
            "version": "1.0.0",
            "uptime_seconds": 3600,
            "memory_usage_mb": 256,
            "cpu_usage_percent": 15.5,
            "active_workflows": 42,
            "events_processed": 1000,
        },
    )

    # Degraded node
    degraded_heartbeat = ModelNodeHeartbeatInput(
        node_id="reducer-02",
        health_status="DEGRADED",
        metadata={
            "version": "1.0.0",
            "uptime_seconds": 7200,
            "memory_usage_mb": 480,
            "cpu_usage_percent": 85.0,
            "warnings": ["High CPU usage", "Growing backlog"],
        },
    )

    print(f"Node 1: {healthy_heartbeat.node_id}")
    print(f"  - Status: {healthy_heartbeat.health_status}")
    print(f"  - Uptime: {healthy_heartbeat.metadata['uptime_seconds']}s")
    print(f"  - CPU: {healthy_heartbeat.metadata['cpu_usage_percent']}%")

    print(f"\nNode 2: {degraded_heartbeat.node_id}")
    print(f"  - Status: {degraded_heartbeat.health_status}")
    print(f"  - Uptime: {degraded_heartbeat.metadata['uptime_seconds']}s")
    print(f"  - CPU: {degraded_heartbeat.metadata['cpu_usage_percent']}%")
    print(f"  - Warnings: {degraded_heartbeat.metadata['warnings']}")


def example_7_validation_errors():
    """Example 7: Validation error handling"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Validation Error Handling")
    print("=" * 80)

    print("\n1. Invalid operation_type:")
    try:
        invalid_op = ModelDatabaseOperationInput(
            operation_type="invalid_operation",
            correlation_id=uuid4(),
        )
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: Operation type not in allowed list")

    print("\n2. Negative execution_time_ms:")
    try:
        invalid_time = ModelWorkflowExecutionInput(
            correlation_id=uuid4(),
            workflow_type="test",
            current_state="PROCESSING",
            namespace="test",
            execution_time_ms=-100,
        )
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: execution_time_ms must be >= 0")

    print("\n3. Invalid step_order:")
    try:
        invalid_step = ModelWorkflowStepInput(
            workflow_id=uuid4(),
            step_name="test",
            step_order=0,
            status="COMPLETED",
        )
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: step_order must be >= 1")

    print("\n4. Invalid hash pattern:")
    try:
        invalid_hash = ModelMetadataStampInput(
            file_hash="INVALID_HASH",
            stamp_data={},
            namespace="test",
        )
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: Hash must be 64-128 char hex")

    print("\n5. Extra fields (extra='forbid'):")
    try:
        invalid_extra = ModelNodeHeartbeatInput(
            node_id="test",
            health_status="HEALTHY",
            invalid_field="should fail",
        )
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: Extra fields not allowed")


if __name__ == "__main__":
    print("=" * 80)
    print("DATABASE ADAPTER EFFECT NODE - INPUT MODELS USAGE EXAMPLES")
    print("=" * 80)

    example_1_workflow_started()
    example_2_workflow_step_completed()
    example_3_bridge_state_aggregation()
    example_4_fsm_transition()
    example_5_metadata_stamp_audit()
    example_6_node_heartbeat()
    example_7_validation_errors()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 80)
