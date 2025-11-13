#!/usr/bin/env python3
"""Unit tests for ModelWorkflowStep.

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

from omninode_bridge.infrastructure.entities.model_workflow_step import (
    ModelWorkflowStep,
)


class TestModelWorkflowStep:
    """Test suite for ModelWorkflowStep."""

    def test_model_instantiation_with_all_fields(self):
        """Test model instantiation with all required fields."""
        id = uuid4()
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"
        execution_time_ms = 500
        step_data = {"input": "data", "output": "result"}
        error_message = None
        created_at = datetime.now(UTC)

        step = ModelWorkflowStep(
            id=id,
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
            execution_time_ms=execution_time_ms,
            step_data=step_data,
            error_message=error_message,
            created_at=created_at,
        )

        assert step.id == id
        assert step.workflow_id == workflow_id
        assert step.step_name == step_name
        assert step.step_order == step_order
        assert step.status == status
        assert step.execution_time_ms == execution_time_ms
        assert step.step_data == step_data
        assert step.error_message == error_message
        assert step.created_at == created_at

    def test_model_instantiation_with_required_fields_only(self):
        """Test model instantiation with only required fields."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"

        step = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
        )

        assert step.id is None  # Default value
        assert step.workflow_id == workflow_id
        assert step.step_name == step_name
        assert step.step_order == step_order
        assert step.status == status
        assert step.execution_time_ms is None  # Default value
        assert step.step_data == {}  # Default value
        assert step.error_message is None  # Default value
        assert step.created_at is None  # Default value

    def test_model_validation_missing_required_fields(self):
        """Test model validation with missing required fields."""
        workflow_id = uuid4()
        # Missing step_name, step_order, status

        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowStep(workflow_id=workflow_id)

        errors = exc_info.value.errors()
        assert len(errors) == 3
        assert any("step_name" in str(error) for error in errors)
        assert any("step_order" in str(error) for error in errors)
        assert any("status" in str(error) for error in errors)

    def test_model_validation_invalid_step_name(self):
        """Test model validation with invalid step_name."""
        workflow_id = uuid4()
        step_order = 1
        status = "completed"

        # Empty step_name
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name="",
                step_order=step_order,
                status=status,
            )

        assert "step_name" in str(exc_info.value)

        # Too long step_name
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name="a" * 101,  # 101 characters, max is 100
                step_order=step_order,
                status=status,
            )

        assert "step_name" in str(exc_info.value)

    def test_model_validation_invalid_step_order(self):
        """Test model validation with invalid step_order."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        status = "completed"

        # Negative step_order
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=step_name,
                step_order=-1,
                status=status,
            )

        assert "step_order" in str(exc_info.value)

    def test_model_validation_invalid_status(self):
        """Test model validation with invalid status."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1

        # Empty status
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=step_name,
                step_order=step_order,
                status="",
            )

        assert "status" in str(exc_info.value)

        # Too long status
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=step_name,
                step_order=step_order,
                status="a" * 51,  # 51 characters, max is 50
            )

        assert "status" in str(exc_info.value)

    def test_model_validation_invalid_execution_time_ms(self):
        """Test model validation with invalid execution_time_ms."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"

        # Negative execution_time_ms
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=step_name,
                step_order=step_order,
                status=status,
                execution_time_ms=-1,
            )

        assert "execution_time_ms" in str(exc_info.value)

    def test_model_serialization(self):
        """Test model serialization to JSON."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"

        step = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
        )

        # Test model_dump
        data = step.model_dump()
        assert data["workflow_id"] == workflow_id
        assert data["step_name"] == step_name
        assert data["step_order"] == step_order
        assert data["status"] == status
        assert data["execution_time_ms"] is None
        assert data["step_data"] == {}
        assert data["error_message"] is None
        assert data["created_at"] is None

        # Test model_dump_json
        json_str = step.model_dump_json()
        parsed_data = json.loads(json_str)
        assert parsed_data["workflow_id"] == str(workflow_id)
        assert parsed_data["step_name"] == step_name
        assert parsed_data["step_order"] == step_order
        assert parsed_data["status"] == status

    def test_model_deserialization(self):
        """Test model deserialization from JSON."""
        id = uuid4()
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"
        execution_time_ms = 500
        step_data = {"input": "data", "output": "result"}
        error_message = None
        created_at = datetime.now(UTC)

        data = {
            "id": str(id),
            "workflow_id": str(workflow_id),
            "step_name": step_name,
            "step_order": step_order,
            "status": status,
            "execution_time_ms": execution_time_ms,
            "step_data": step_data,
            "error_message": error_message,
            "created_at": created_at.isoformat(),
        }

        # Test model_validate
        step = ModelWorkflowStep.model_validate(data)
        assert step.id == id
        assert step.workflow_id == workflow_id
        assert step.step_name == step_name
        assert step.step_order == step_order
        assert step.status == status
        assert step.execution_time_ms == execution_time_ms
        assert step.step_data == step_data
        assert step.error_message == error_message
        assert step.created_at == created_at

    def test_model_roundtrip(self):
        """Test model serialization and deserialization roundtrip."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"
        step_data = {"input": "data", "output": "result"}

        original = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
            step_data=step_data,
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = ModelWorkflowStep.model_validate(data)

        # Verify all fields match
        assert restored.workflow_id == original.workflow_id
        assert restored.step_name == original.step_name
        assert restored.step_order == original.step_order
        assert restored.status == original.status
        assert restored.step_data == original.step_data

    def test_model_equality(self):
        """Test model equality comparison."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"

        step1 = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
        )

        step2 = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
        )

        step3 = ModelWorkflowStep(
            workflow_id=uuid4(),  # Different ID
            step_name=step_name,
            step_order=step_order,
            status=status,
        )

        # Test equality
        assert step1.model_dump() == step2.model_dump()
        assert step1.model_dump() != step3.model_dump()

    def test_model_copy(self):
        """Test model copying."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"

        original = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
        )

        # Test model_copy
        copied = original.model_copy()
        assert copied.workflow_id == original.workflow_id
        assert copied.step_name == original.step_name
        assert copied.step_order == original.step_order
        assert copied.status == original.status
        assert copied is not original  # Different instances

    def test_model_update(self):
        """Test model field updates."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"

        step = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
        )

        # Update fields
        updated = step.model_copy(
            update={
                "status": "failed",
                "execution_time_ms": 1000,
                "error_message": "Step failed due to error",
            }
        )

        assert updated.status == "failed"
        assert updated.execution_time_ms == 1000
        assert updated.error_message == "Step failed due to error"
        # Original should be unchanged
        assert step.status == "completed"
        assert step.execution_time_ms is None
        assert step.error_message is None

    def test_model_step_data(self):
        """Test step_data field."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"

        # Test with empty step_data
        step = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
        )

        assert step.step_data == {}

        # Test with step_data
        step_data = {
            "input_data": {"file_path": "/path/to/file"},
            "processing_params": {"algorithm": "blake3"},
            "result": {"hash": "abc123def456"},
        }

        step = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
            step_data=step_data,
        )

        assert step.step_data == step_data

    def test_model_jsonb_validation(self):
        """Test JSONB field validation."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "completed"

        # This should not raise an error if validation is working correctly
        step = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
            step_data={"test": "value"},
        )

        # The _validate_jsonb_fields method should be called during initialization
        # If it's working correctly, no exception should be raised

    def test_model_step_statuses(self):
        """Test model with different step statuses."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1

        # Test all valid step statuses
        statuses = ["pending", "running", "completed", "failed", "skipped"]

        for status in statuses:
            step = ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=step_name,
                step_order=step_order,
                status=status,
            )

            assert step.status == status

    def test_model_error_handling(self):
        """Test model with error information."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        step_order = 1
        status = "failed"
        execution_time_ms = 1000
        error_message = "Step failed due to invalid input"

        step = ModelWorkflowStep(
            workflow_id=workflow_id,
            step_name=step_name,
            step_order=step_order,
            status=status,
            execution_time_ms=execution_time_ms,
            error_message=error_message,
        )

        assert step.status == "failed"
        assert step.execution_time_ms == execution_time_ms
        assert step.error_message == error_message

    def test_model_step_order_sequence(self):
        """Test model with sequential step orders."""
        workflow_id = uuid4()
        step_name = "hash_generation"
        status = "completed"

        # Test with different step orders
        step_orders = [0, 1, 5, 10, 100]

        for step_order in step_orders:
            step = ModelWorkflowStep(
                workflow_id=workflow_id,
                step_name=f"{step_name}_{step_order}",
                step_order=step_order,
                status=status,
            )

            assert step.step_order == step_order
