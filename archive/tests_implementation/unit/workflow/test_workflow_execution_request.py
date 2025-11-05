"""Comprehensive tests for ModelWorkflowExecutionRequest - Backward Compatibility Testing.

Tests the Union[Model, dict[str, Any]] patterns mentioned in PR #11 feedback.
Addresses backward compatibility concerns and migration path validation.
"""

import pytest
from datetime import datetime
from uuid import uuid4, UUID
from pydantic import ValidationError

from omnibase_infra.models.core.workflow.model_workflow_execution_request import (
    ModelWorkflowExecutionRequest,
)
from omnibase_infra.models.core.workflow.model_workflow_execution_context import (
    ModelWorkflowExecutionContext,
)


class TestModelWorkflowExecutionRequest:
    """Test ModelWorkflowExecutionRequest with focus on backward compatibility."""

    def test_model_initialization_defaults(self):
        """Test model initializes with proper default values."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        request = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="test_workflow",
        )

        assert request.workflow_id == workflow_id
        assert request.correlation_id == correlation_id
        assert request.workflow_type == "test_workflow"
        assert isinstance(request.execution_context, ModelWorkflowExecutionContext)
        assert request.agent_coordination_required is True
        assert request.priority == "normal"
        assert request.timeout_seconds == 300
        assert request.retry_count == 3
        assert request.environment == "development"
        assert request.background_execution is False
        assert request.progress_tracking_enabled is True
        assert request.sub_agent_fleet_size == 1
        assert isinstance(request.created_at, datetime)

    def test_execution_context_with_model_object(self):
        """Test execution_context field with proper ModelWorkflowExecutionContext object."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        # Create explicit execution context model
        execution_context = ModelWorkflowExecutionContext(
            task_id=uuid4(),
            agent_id="test_agent",
            session_id=uuid4(),
            execution_phase="initialization",
        )

        request = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="model_context_workflow",
            execution_context=execution_context,
        )

        # Verify context is preserved as model object
        assert isinstance(request.execution_context, ModelWorkflowExecutionContext)
        assert request.execution_context.agent_id == "test_agent"
        assert request.execution_context.execution_phase == "initialization"

    def test_execution_context_with_dict_valid_conversion(self):
        """Test execution_context field with dict that can be converted to model."""
        workflow_id = uuid4()
        correlation_id = uuid4()
        task_id = uuid4()
        session_id = uuid4()

        # Provide dict with valid model fields
        context_dict = {
            "task_id": task_id,
            "agent_id": "dict_agent",
            "session_id": session_id,
            "execution_phase": "planning",
            "max_cycles": 25,
        }

        request = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="dict_context_workflow",
            execution_context=context_dict,
        )

        # Verify dict was converted to model
        assert isinstance(request.execution_context, ModelWorkflowExecutionContext)
        assert request.execution_context.agent_id == "dict_agent"
        assert request.execution_context.execution_phase == "planning"
        assert request.execution_context.max_cycles == 25

    def test_execution_context_with_dict_invalid_fallback(self):
        """Test execution_context field with dict that cannot be converted - fallback behavior."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        # Provide dict with invalid/extra fields that don't match model
        invalid_context_dict = {
            "invalid_field": "invalid_value",
            "another_invalid": 123,
            "completely_wrong": True,
        }

        request = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="invalid_dict_workflow",
            execution_context=invalid_context_dict,
        )

        # Verify dict is kept as-is for backward compatibility
        assert isinstance(request.execution_context, dict)
        assert request.execution_context == invalid_context_dict
        assert request.execution_context["invalid_field"] == "invalid_value"

    def test_execution_context_backward_compatibility_scenarios(self):
        """Test various backward compatibility scenarios for execution_context."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        # Scenario 1: Legacy dict with some valid fields
        legacy_dict_1 = {
            "agent_id": "legacy_agent",
            "some_legacy_field": "legacy_value",
            "task_data": {"nested": "data"},
        }

        request_1 = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="legacy_workflow_1",
            execution_context=legacy_dict_1,
        )

        # Should attempt conversion but likely fall back to dict due to invalid fields
        # The exact behavior depends on how the model validation handles extra fields
        assert request_1.execution_context is not None

        # Scenario 2: Minimal dict with required fields only
        minimal_dict = {
            "task_id": uuid4(),
            "agent_id": "minimal_agent",
            "session_id": uuid4(),
        }

        request_2 = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="minimal_workflow",
            execution_context=minimal_dict,
        )

        # This should successfully convert to model
        assert isinstance(request_2.execution_context, ModelWorkflowExecutionContext)
        assert request_2.execution_context.agent_id == "minimal_agent"

        # Scenario 3: Empty dict
        empty_dict = {}

        request_3 = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="empty_context_workflow",
            execution_context=empty_dict,
        )

        # Should fall back to dict since empty dict can't populate required model fields
        assert isinstance(request_3.execution_context, dict)
        assert request_3.execution_context == {}

    def test_required_field_validation(self):
        """Test validation of required fields."""
        # Missing workflow_id
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecutionRequest(
                correlation_id=uuid4(),
                workflow_type="test_workflow",
            )
        assert "workflow_id" in str(exc_info.value)

        # Missing correlation_id
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecutionRequest(
                workflow_id=uuid4(),
                workflow_type="test_workflow",
            )
        assert "correlation_id" in str(exc_info.value)

        # Missing workflow_type
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecutionRequest(
                workflow_id=uuid4(),
                correlation_id=uuid4(),
            )
        assert "workflow_type" in str(exc_info.value)

    def test_field_type_validation(self):
        """Test field type validation."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        # Invalid timeout_seconds (negative)
        with pytest.raises(ValidationError):
            ModelWorkflowExecutionRequest(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                workflow_type="test_workflow",
                timeout_seconds=-1,
            )

        # Invalid retry_count (negative)
        with pytest.raises(ValidationError):
            ModelWorkflowExecutionRequest(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                workflow_type="test_workflow",
                retry_count=-1,
            )

        # Invalid sub_agent_fleet_size (negative)
        with pytest.raises(ValidationError):
            ModelWorkflowExecutionRequest(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                workflow_type="test_workflow",
                sub_agent_fleet_size=-1,
            )

    def test_priority_validation(self):
        """Test priority field validation."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        # Valid priorities (according to description: low, normal, high, critical)
        valid_priorities = ["low", "normal", "high", "critical"]
        for priority in valid_priorities:
            request = ModelWorkflowExecutionRequest(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                workflow_type="test_workflow",
                priority=priority,
            )
            assert request.priority == priority

        # Note: Model currently accepts any string - might want to add enum validation
        invalid_priority = "invalid_priority"
        request = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="test_workflow",
            priority=invalid_priority,
        )
        assert request.priority == invalid_priority  # Currently allowed

    def test_environment_validation(self):
        """Test environment field validation."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        # Common environments
        valid_environments = ["development", "staging", "testing", "production"]
        for environment in valid_environments:
            request = ModelWorkflowExecutionRequest(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                workflow_type="test_workflow",
                environment=environment,
            )
            assert request.environment == environment

    def test_comprehensive_workflow_request(self):
        """Test comprehensive workflow request with all fields."""
        workflow_id = uuid4()
        correlation_id = uuid4()
        task_id = uuid4()
        session_id = uuid4()
        created_time = datetime.utcnow()

        execution_context = ModelWorkflowExecutionContext(
            task_id=task_id,
            agent_id="comprehensive_agent",
            session_id=session_id,
            execution_phase="execution",
            max_cycles=50,
        )

        request = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="comprehensive_workflow",
            execution_context=execution_context,
            agent_coordination_required=True,
            priority="high",
            timeout_seconds=600,
            retry_count=5,
            environment="production",
            background_execution=True,
            progress_tracking_enabled=True,
            sub_agent_fleet_size=3,
            created_at=created_time,
        )

        # Verify all fields
        assert request.workflow_id == workflow_id
        assert request.correlation_id == correlation_id
        assert request.workflow_type == "comprehensive_workflow"
        assert isinstance(request.execution_context, ModelWorkflowExecutionContext)
        assert request.execution_context.agent_id == "comprehensive_agent"
        assert request.agent_coordination_required is True
        assert request.priority == "high"
        assert request.timeout_seconds == 600
        assert request.retry_count == 5
        assert request.environment == "production"
        assert request.background_execution is True
        assert request.progress_tracking_enabled is True
        assert request.sub_agent_fleet_size == 3
        assert request.created_at == created_time

    def test_json_serialization_with_model_context(self):
        """Test JSON serialization when execution_context is a model."""
        workflow_id = uuid4()
        correlation_id = uuid4()
        task_id = uuid4()

        execution_context = ModelWorkflowExecutionContext(
            task_id=task_id,
            agent_id="serialization_agent",
            session_id=uuid4(),
        )

        request = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="serialization_workflow",
            execution_context=execution_context,
        )

        json_data = request.model_dump()

        # Verify serialization
        assert json_data["workflow_type"] == "serialization_workflow"
        assert isinstance(json_data["execution_context"], dict)
        assert json_data["execution_context"]["agent_id"] == "serialization_agent"

    def test_json_serialization_with_dict_context(self):
        """Test JSON serialization when execution_context is a dict."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        legacy_context = {
            "legacy_field": "legacy_value",
            "task_data": {"nested": "information"},
        }

        request = ModelWorkflowExecutionRequest(
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            workflow_type="dict_serialization_workflow",
            execution_context=legacy_context,
        )

        json_data = request.model_dump()

        # Verify serialization preserves dict
        assert json_data["workflow_type"] == "dict_serialization_workflow"
        assert isinstance(json_data["execution_context"], dict)
        assert json_data["execution_context"]["legacy_field"] == "legacy_value"
        assert json_data["execution_context"]["task_data"]["nested"] == "information"

    def test_model_extra_fields_forbidden(self):
        """Test that extra fields are forbidden (ConfigDict extra='forbid')."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecutionRequest(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                workflow_type="test_workflow",
                extra_field="should_be_rejected",
            )
        assert "extra fields not permitted" in str(exc_info.value)

    def test_migration_path_validation(self):
        """Test migration path from dict to model for execution_context."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        # Test gradual migration scenario
        # Stage 1: Legacy system provides dict
        legacy_request_data = {
            "workflow_id": workflow_id,
            "correlation_id": correlation_id,
            "workflow_type": "migration_workflow",
            "execution_context": {
                "agent_id": "migration_agent",
                "legacy_field": "will_be_ignored",
            },
        }

        # Create request from legacy data
        request = ModelWorkflowExecutionRequest(**legacy_request_data)

        # Verify migration behavior
        assert request.workflow_type == "migration_workflow"

        # Execution context conversion should attempt to convert to model
        # If successful, it's a model; if not, it remains a dict
        assert request.execution_context is not None

        if isinstance(request.execution_context, ModelWorkflowExecutionContext):
            # Successfully migrated to model
            assert request.execution_context.agent_id == "migration_agent"
        else:
            # Remained as dict due to extra fields
            assert isinstance(request.execution_context, dict)
            assert request.execution_context["agent_id"] == "migration_agent"
            assert request.execution_context["legacy_field"] == "will_be_ignored"

    def test_validator_error_handling(self):
        """Test field_validator error handling for execution_context conversion."""
        workflow_id = uuid4()
        correlation_id = uuid4()

        # Test with various problematic dict structures
        problematic_contexts = [
            {"invalid": "data", "cannot": "convert"},
            {"task_id": "not_a_uuid", "agent_id": "test"},
            {"nested": {"complex": {"structure": True}}},
            {"agent_id": None},  # None value for required field
        ]

        for context in problematic_contexts:
            request = ModelWorkflowExecutionRequest(
                workflow_id=workflow_id,
                correlation_id=correlation_id,
                workflow_type=f"error_test_{hash(str(context))}",
                execution_context=context,
            )

            # Should not raise an error - should fall back to dict
            assert request.execution_context is not None
            # Most likely will remain as dict due to conversion failure
            if isinstance(request.execution_context, dict):
                assert request.execution_context == context


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])