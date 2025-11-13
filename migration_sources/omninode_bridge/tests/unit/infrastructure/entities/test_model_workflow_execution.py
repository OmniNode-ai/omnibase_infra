#!/usr/bin/env python3
"""Unit tests for ModelWorkflowExecution.

Tests cover:
- Model instantiation with all fields
- Model validation
- Default values
- JSON serialization/deserialization
- Field validation
- Model configuration
"""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omninode_bridge.infrastructure.entities.model_workflow_execution import (
    ModelWorkflowExecution,
)


class TestModelWorkflowExecution:
    """Test suite for ModelWorkflowExecution."""

    def test_model_instantiation_with_all_fields(self):
        """Test model instantiation with all required fields."""
        id = uuid4()
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)
        execution_time_ms = 5000
        error_message = None
        metadata = {"key": "value"}
        created_at = datetime.now(UTC)
        updated_at = datetime.now(UTC)

        execution = ModelWorkflowExecution(
            id=id,
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
            completed_at=completed_at,
            execution_time_ms=execution_time_ms,
            error_message=error_message,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
        )

        assert execution.id == id
        assert execution.correlation_id == correlation_id
        assert execution.workflow_type == workflow_type
        assert execution.current_state == current_state
        assert execution.namespace == namespace
        assert execution.started_at == started_at
        assert execution.completed_at == completed_at
        assert execution.execution_time_ms == execution_time_ms
        assert execution.error_message == error_message
        assert execution.metadata == metadata
        assert execution.created_at == created_at
        assert execution.updated_at == updated_at

    def test_model_instantiation_with_required_fields_only(self):
        """Test model instantiation with only required fields."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)

        execution = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
        )

        assert execution.id is None  # Default value
        assert execution.correlation_id == correlation_id
        assert execution.workflow_type == workflow_type
        assert execution.current_state == current_state
        assert execution.namespace == namespace
        assert execution.started_at == started_at
        assert execution.completed_at is None  # Default value
        assert execution.execution_time_ms is None  # Default value
        assert execution.error_message is None  # Default value
        assert execution.metadata == {}  # Default value
        assert execution.created_at is None  # Default value
        assert execution.updated_at is None  # Default value

    def test_model_validation_missing_required_fields(self):
        """Test model validation with missing required fields."""
        correlation_id = uuid4()
        # Missing workflow_type, current_state, namespace, started_at

        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecution(correlation_id=correlation_id)

        errors = exc_info.value.errors()
        assert len(errors) == 4
        assert any("workflow_type" in str(error) for error in errors)
        assert any("current_state" in str(error) for error in errors)
        assert any("namespace" in str(error) for error in errors)
        assert any("started_at" in str(error) for error in errors)

    def test_model_validation_invalid_workflow_type(self):
        """Test model validation with invalid workflow_type."""
        correlation_id = uuid4()
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)

        # Empty workflow_type
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecution(
                correlation_id=correlation_id,
                workflow_type="",
                current_state=current_state,
                namespace=namespace,
                started_at=started_at,
            )

        assert "workflow_type" in str(exc_info.value)

        # Too long workflow_type
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecution(
                correlation_id=correlation_id,
                workflow_type="a" * 101,  # 101 characters, max is 100
                current_state=current_state,
                namespace=namespace,
                started_at=started_at,
            )

        assert "workflow_type" in str(exc_info.value)

    def test_model_validation_invalid_current_state(self):
        """Test model validation with invalid current_state."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        namespace = "production"
        started_at = datetime.now(UTC)

        # Empty current_state
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecution(
                correlation_id=correlation_id,
                workflow_type=workflow_type,
                current_state="",
                namespace=namespace,
                started_at=started_at,
            )

        assert "current_state" in str(exc_info.value)

        # Too long current_state
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecution(
                correlation_id=correlation_id,
                workflow_type=workflow_type,
                current_state="a" * 51,  # 51 characters, max is 50
                namespace=namespace,
                started_at=started_at,
            )

        assert "current_state" in str(exc_info.value)

    def test_model_validation_invalid_namespace(self):
        """Test model validation with invalid namespace."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        started_at = datetime.now(UTC)

        # Empty namespace
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecution(
                correlation_id=correlation_id,
                workflow_type=workflow_type,
                current_state=current_state,
                namespace="",
                started_at=started_at,
            )

        assert "namespace" in str(exc_info.value)

        # Too long namespace
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecution(
                correlation_id=correlation_id,
                workflow_type=workflow_type,
                current_state=current_state,
                namespace="a" * 256,  # 256 characters, max is 255
                started_at=started_at,
            )

        assert "namespace" in str(exc_info.value)

    def test_model_validation_invalid_execution_time_ms(self):
        """Test model validation with invalid execution_time_ms."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)

        # Negative execution_time_ms
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecution(
                correlation_id=correlation_id,
                workflow_type=workflow_type,
                current_state=current_state,
                namespace=namespace,
                started_at=started_at,
                execution_time_ms=-1,
            )

        assert "execution_time_ms" in str(exc_info.value)

    def test_model_serialization(self):
        """Test model serialization to JSON."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)

        execution = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
        )

        # Test model_dump
        data = execution.model_dump()
        assert (
            data["correlation_id"] == correlation_id
        )  # UUID is not converted to string in model_dump
        assert data["workflow_type"] == workflow_type
        assert data["current_state"] == current_state
        assert data["namespace"] == namespace
        assert (
            data["started_at"] == started_at
        )  # datetime is not converted to string in model_dump
        assert data["completed_at"] is None
        assert data["execution_time_ms"] is None
        assert data["error_message"] is None
        assert data["metadata"] == {}

        # Test model_dump_json
        json_str = execution.model_dump_json()
        parsed_data = json.loads(json_str)
        assert parsed_data["correlation_id"] == str(correlation_id)
        assert parsed_data["workflow_type"] == workflow_type
        assert parsed_data["current_state"] == current_state

    def test_model_deserialization(self):
        """Test model deserialization from JSON."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)
        execution_time_ms = 5000
        error_message = "Test error"
        metadata = {"key": "value"}

        data = {
            "correlation_id": str(correlation_id),
            "workflow_type": workflow_type,
            "current_state": current_state,
            "namespace": namespace,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "execution_time_ms": execution_time_ms,
            "error_message": error_message,
            "metadata": metadata,
        }

        # Test model_validate
        execution = ModelWorkflowExecution.model_validate(data)
        assert execution.correlation_id == correlation_id
        assert execution.workflow_type == workflow_type
        assert execution.current_state == current_state
        assert execution.namespace == namespace
        assert execution.started_at == started_at
        assert execution.completed_at == completed_at
        assert execution.execution_time_ms == execution_time_ms
        assert execution.error_message == error_message
        assert execution.metadata == metadata

    def test_model_roundtrip(self):
        """Test model serialization and deserialization roundtrip."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)
        metadata = {"key": "value"}

        original = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
            metadata=metadata,
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = ModelWorkflowExecution.model_validate(data)

        # Verify all fields match
        assert restored.correlation_id == original.correlation_id
        assert restored.workflow_type == original.workflow_type
        assert restored.current_state == original.current_state
        assert restored.namespace == original.namespace
        assert restored.started_at == original.started_at
        assert restored.metadata == original.metadata

    def test_model_equality(self):
        """Test model equality comparison."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)

        execution1 = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
        )

        execution2 = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
        )

        execution3 = ModelWorkflowExecution(
            correlation_id=uuid4(),  # Different ID
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
        )

        # Test equality
        assert execution1.model_dump() == execution2.model_dump()
        assert execution1.model_dump() != execution3.model_dump()

    def test_model_copy(self):
        """Test model copying."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)

        original = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
        )

        # Test model_copy
        copied = original.model_copy()
        assert copied.correlation_id == original.correlation_id
        assert copied.workflow_type == original.workflow_type
        assert copied.current_state == original.current_state
        assert copied.namespace == original.namespace
        assert copied.started_at == original.started_at
        assert copied is not original  # Different instances

    def test_model_update(self):
        """Test model field updates."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)

        execution = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
        )

        # Update fields
        updated = execution.model_copy(
            update={
                "current_state": "COMPLETED",
                "completed_at": datetime.now(UTC),
                "execution_time_ms": 5000,
            }
        )

        assert updated.current_state == "COMPLETED"
        assert updated.completed_at is not None
        assert updated.execution_time_ms == 5000
        # Original should be unchanged
        assert execution.current_state == "PROCESSING"
        assert execution.completed_at is None
        assert execution.execution_time_ms is None

    def test_model_metadata(self):
        """Test metadata field."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)

        # Test with empty metadata
        execution = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
        )

        assert execution.metadata == {}

        # Test with metadata
        metadata = {
            "input_data": {"key": "value"},
            "processing_steps": ["step1", "step2"],
            "result": {"output": "success"},
        }

        execution = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
            metadata=metadata,
        )

        assert execution.metadata == metadata

    def test_model_jsonb_validation(self):
        """Test JSONB field validation."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "PROCESSING"
        namespace = "production"
        started_at = datetime.now(UTC)

        # This should not raise an error if validation is working correctly
        execution = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
            metadata={"test": "value"},
        )

        # The _validate_jsonb_fields method should be called during initialization
        # If it's working correctly, no exception should be raised

    def test_model_workflow_states(self):
        """Test model with different workflow states."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        namespace = "production"
        started_at = datetime.now(UTC)

        # Test all valid workflow states
        states = ["PENDING", "PROCESSING", "COMPLETED", "FAILED"]

        for state in states:
            execution = ModelWorkflowExecution(
                correlation_id=correlation_id,
                workflow_type=workflow_type,
                current_state=state,
                namespace=namespace,
                started_at=started_at,
            )

            assert execution.current_state == state

    def test_model_error_handling(self):
        """Test model with error information."""
        correlation_id = uuid4()
        workflow_type = "metadata_stamping"
        current_state = "FAILED"
        namespace = "production"
        started_at = datetime.now(UTC)
        completed_at = datetime.now(UTC)
        error_message = "Workflow failed due to invalid input"

        execution = ModelWorkflowExecution(
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            current_state=current_state,
            namespace=namespace,
            started_at=started_at,
            completed_at=completed_at,
            error_message=error_message,
        )

        assert execution.current_state == "FAILED"
        assert execution.completed_at == completed_at
        assert execution.error_message == error_message
