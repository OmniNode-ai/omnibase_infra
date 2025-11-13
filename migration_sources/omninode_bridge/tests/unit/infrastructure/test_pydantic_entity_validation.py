#!/usr/bin/env python3
"""
Unit Tests for Pydantic Entity Model Validation and Type Safety.

This test module verifies that all database adapter entity models
enforce strong typing, field validation, and Pydantic v2 compliance.

Test Coverage:
- Required field validation
- Type validation (UUID, str, int, datetime)
- Field constraints (min_length, max_length, ge, pattern)
- Extra fields forbidden
- Invalid type rejection
- Pydantic ValidationError handling
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

# ===========================
# Workflow Execution Entity Tests
# ===========================


class TestModelWorkflowExecutionInputValidation:
    """Test Pydantic validation for ModelWorkflowExecutionInput."""

    def test_valid_entity_creation(self, sample_correlation_id):
        """Test creating a valid workflow execution entity."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_workflow_execution_input import (
            ModelWorkflowExecutionInput,
        )

        entity = ModelWorkflowExecutionInput(
            correlation_id=sample_correlation_id,
            workflow_type="metadata_stamping",
            current_state="PROCESSING",
            namespace="test_namespace",
            started_at=datetime.now(UTC),
            metadata={"test": True},
        )

        assert isinstance(entity.correlation_id, UUID)
        assert entity.workflow_type == "metadata_stamping"
        assert entity.current_state == "PROCESSING"
        assert entity.namespace == "test_namespace"
        assert isinstance(entity.started_at, datetime)

    def test_required_field_missing(self):
        """Test that missing required fields raise ValidationError."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_workflow_execution_input import (
            ModelWorkflowExecutionInput,
        )

        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecutionInput(
                correlation_id=uuid4(),
                workflow_type="test",
                # Missing required 'current_state'
                namespace="test",
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("current_state",)
        assert errors[0]["type"] == "missing"

    def test_invalid_uuid_type(self):
        """Test that invalid UUID type raises ValidationError."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_workflow_execution_input import (
            ModelWorkflowExecutionInput,
        )

        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecutionInput(
                correlation_id="not-a-uuid",  # Invalid UUID
                workflow_type="test",
                current_state="PROCESSING",
                namespace="test",
            )

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("correlation_id",) for error in errors)

    def test_field_constraint_min_length(self):
        """Test that field constraints are enforced (min_length)."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_workflow_execution_input import (
            ModelWorkflowExecutionInput,
        )

        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecutionInput(
                correlation_id=uuid4(),
                workflow_type="",  # Empty string violates min_length=1
                current_state="PROCESSING",
                namespace="test",
            )

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("workflow_type",) for error in errors)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden (Pydantic extra='forbid')."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_workflow_execution_input import (
            ModelWorkflowExecutionInput,
        )

        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecutionInput(
                correlation_id=uuid4(),
                workflow_type="test",
                current_state="PROCESSING",
                namespace="test",
                extra_field="not_allowed",  # Extra field forbidden
            )

        errors = exc_info.value.errors()
        assert any(error["type"] == "extra_forbidden" for error in errors)


# ===========================
# Bridge State Entity Tests
# ===========================


class TestModelBridgeStateInputValidation:
    """Test Pydantic validation for ModelBridgeStateInput."""

    def test_valid_entity_creation(self):
        """Test creating a valid bridge state entity."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_bridge_state_input import (
            ModelBridgeStateInput,
        )

        entity = ModelBridgeStateInput(
            bridge_id=uuid4(),
            namespace="test",
            total_workflows_processed=100,
            total_items_aggregated=500,
            aggregation_metadata={"test": True},
            current_fsm_state="aggregating",
        )

        assert isinstance(entity.bridge_id, UUID)
        assert entity.total_workflows_processed == 100
        assert entity.total_items_aggregated == 500
        assert entity.current_fsm_state == "aggregating"

    def test_numeric_constraint_ge_zero(self):
        """Test that numeric constraints are enforced (ge=0)."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_bridge_state_input import (
            ModelBridgeStateInput,
        )

        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeStateInput(
                bridge_id=uuid4(),
                namespace="test",
                total_workflows_processed=-1,  # Negative value violates ge=0
                total_items_aggregated=500,
                aggregation_metadata={},
                current_fsm_state="idle",
            )

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("total_workflows_processed",) for error in errors)


# ===========================
# Metadata Stamp Entity Tests
# ===========================


class TestModelMetadataStampInputValidation:
    """Test Pydantic validation for ModelMetadataStampInput."""

    def test_valid_entity_creation(self, sample_correlation_id):
        """Test creating a valid metadata stamp entity."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_metadata_stamp_input import (
            ModelMetadataStampInput,
        )

        entity = ModelMetadataStampInput(
            workflow_id=sample_correlation_id,
            file_hash="abc123def456789abcdef0123456789abcdef0123456789abcdef01234567890",  # 64 chars
            stamp_data={"stamp_type": "inline"},
            namespace="test",
        )

        assert isinstance(entity.workflow_id, UUID)
        assert len(entity.file_hash) == 64
        assert entity.namespace == "test"

    def test_file_hash_pattern_validation(self):
        """Test that file hash pattern validation works."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_metadata_stamp_input import (
            ModelMetadataStampInput,
        )

        with pytest.raises(ValidationError) as exc_info:
            ModelMetadataStampInput(
                workflow_id=uuid4(),
                file_hash="INVALID_HASH_WITH_CAPS",  # Invalid pattern
                stamp_data={},
                namespace="test",
            )

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("file_hash",) for error in errors)

    def test_file_hash_min_length(self):
        """Test that file hash minimum length is enforced."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_metadata_stamp_input import (
            ModelMetadataStampInput,
        )

        with pytest.raises(ValidationError) as exc_info:
            ModelMetadataStampInput(
                workflow_id=uuid4(),
                file_hash="short",  # Too short (min 64 chars)
                stamp_data={},
                namespace="test",
            )

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("file_hash",) for error in errors)


# ===========================
# FSM Transition Entity Tests
# ===========================


class TestModelFSMTransitionInputValidation:
    """Test Pydantic validation for ModelFSMTransitionInput."""

    def test_valid_entity_creation(self, sample_correlation_id):
        """Test creating a valid FSM transition entity."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_fsm_transition_input import (
            ModelFSMTransitionInput,
        )

        entity = ModelFSMTransitionInput(
            entity_id=sample_correlation_id,
            entity_type="workflow",
            from_state="PENDING",
            to_state="PROCESSING",
            transition_event="start",
            transition_data={},
        )

        assert isinstance(entity.entity_id, UUID)
        assert entity.from_state == "PENDING"
        assert entity.to_state == "PROCESSING"

    def test_initial_transition_null_from_state(self):
        """Test that initial transitions can have null from_state."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_fsm_transition_input import (
            ModelFSMTransitionInput,
        )

        entity = ModelFSMTransitionInput(
            entity_id=uuid4(),
            entity_type="workflow",
            from_state=None,  # Initial state - null is valid
            to_state="PENDING",
            transition_event="created",
            transition_data={},
        )

        assert entity.from_state is None
        assert entity.to_state == "PENDING"


# ===========================
# Node Heartbeat Entity Tests
# ===========================


class TestModelNodeHeartbeatInputValidation:
    """Test Pydantic validation for ModelNodeHeartbeatInput."""

    def test_valid_entity_creation(self):
        """Test creating a valid node heartbeat entity."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_node_heartbeat_input import (
            ModelNodeHeartbeatInput,
        )

        entity = ModelNodeHeartbeatInput(
            node_id="test_node_01",
            health_status="HEALTHY",
            metadata={"version": "1.0"},
        )

        assert entity.node_id == "test_node_01"
        assert entity.health_status == "HEALTHY"
        assert isinstance(entity.last_heartbeat, datetime)


# ===========================
# Workflow Step Entity Tests
# ===========================


class TestModelWorkflowStepInputValidation:
    """Test Pydantic validation for ModelWorkflowStepInput."""

    def test_valid_entity_creation(self, sample_correlation_id):
        """Test creating a valid workflow step entity."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_workflow_step_input import (
            ModelWorkflowStepInput,
        )

        entity = ModelWorkflowStepInput(
            workflow_id=sample_correlation_id,
            step_name="test_step",
            step_order=1,
            status="COMPLETED",
            execution_time_ms=10,
            step_data={},
        )

        assert isinstance(entity.workflow_id, UUID)
        assert entity.step_order == 1
        assert entity.execution_time_ms == 10

    def test_step_order_ge_constraint(self):
        """Test that step_order must be >= 1."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_workflow_step_input import (
            ModelWorkflowStepInput,
        )

        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowStepInput(
                workflow_id=uuid4(),
                step_name="test",
                step_order=0,  # Invalid: must be >= 1
                status="PENDING",
                step_data={},
            )

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("step_order",) for error in errors)


# ===========================
# Cross-Entity Type Safety Tests
# ===========================


@pytest.mark.pydantic_validation
class TestCrossEntityTypeSafety:
    """Test type safety across all entity models."""

    def test_all_entities_use_uuid_types(
        self,
        sample_workflow_execution_entity,
        sample_bridge_state_entity,
        sample_metadata_stamp_entity,
        sample_fsm_transition_entity,
        sample_workflow_step_entity,
    ):
        """Verify all entities use UUID types for identifiers."""
        assert isinstance(sample_workflow_execution_entity.correlation_id, UUID)
        assert isinstance(sample_bridge_state_entity.bridge_id, UUID)
        assert isinstance(sample_metadata_stamp_entity.workflow_id, UUID)
        assert isinstance(sample_fsm_transition_entity.entity_id, UUID)
        assert isinstance(sample_workflow_step_entity.workflow_id, UUID)

    def test_all_entities_reject_dict_input(self):
        """Verify that entities reject raw dict input (no implicit conversion)."""
        from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_workflow_execution_input import (
            ModelWorkflowExecutionInput,
        )

        # Pydantic v2 should NOT accept raw dict for UUID field
        with pytest.raises(ValidationError):
            ModelWorkflowExecutionInput(
                correlation_id={"not": "a_uuid"},  # Dict instead of UUID
                workflow_type="test",
                current_state="PROCESSING",
                namespace="test",
            )

    def test_pydantic_v2_config_enforcement(self, sample_workflow_execution_entity):
        """Verify Pydantic v2 config is properly enforced."""
        # Test that validate_assignment is True (mutations are validated)
        with pytest.raises(ValidationError):
            sample_workflow_execution_entity.workflow_type = (
                ""  # Empty string violates min_length
            )
