#!/usr/bin/env python3
"""
Unit tests for EntityRegistry.

Tests type-safe entity resolution, validation, and serialization.

ONEX v2.0 Compliance:
- Comprehensive test coverage
- Type safety validation
- Error handling verification
- JSONB field handling tests
"""

import json
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field, ValidationError

from omninode_bridge.infrastructure.entities.model_bridge_state import ModelBridgeState
from omninode_bridge.infrastructure.entities.model_fsm_transition import (
    ModelFSMTransition,
)
from omninode_bridge.infrastructure.entities.model_metadata_stamp import (
    ModelMetadataStamp,
)
from omninode_bridge.infrastructure.entities.model_node_health_metrics import (
    ModelNodeHealthMetrics,
)
from omninode_bridge.infrastructure.entities.model_node_heartbeat import (
    ModelNodeHeartbeat,
)
from omninode_bridge.infrastructure.entities.model_workflow_execution import (
    ModelWorkflowExecution,
)
from omninode_bridge.infrastructure.entities.model_workflow_step import (
    ModelWorkflowStep,
)
from omninode_bridge.infrastructure.entity_registry import EntityRegistry
from omninode_bridge.infrastructure.enum_entity_type import EnumEntityType


class TestEntityRegistryModelMapping:
    """Test entity type to model mapping."""

    def test_get_model_workflow_execution(self):
        """Test getting ModelWorkflowExecution for WORKFLOW_EXECUTION type."""
        model = EntityRegistry.get_model(EnumEntityType.WORKFLOW_EXECUTION)
        assert model == ModelWorkflowExecution
        assert model.__name__ == "ModelWorkflowExecution"

    def test_get_model_workflow_step(self):
        """Test getting ModelWorkflowStep for WORKFLOW_STEP type."""
        model = EntityRegistry.get_model(EnumEntityType.WORKFLOW_STEP)
        assert model == ModelWorkflowStep
        assert model.__name__ == "ModelWorkflowStep"

    def test_get_model_metadata_stamp(self):
        """Test getting ModelMetadataStamp for METADATA_STAMP type."""
        model = EntityRegistry.get_model(EnumEntityType.METADATA_STAMP)
        assert model == ModelMetadataStamp
        assert model.__name__ == "ModelMetadataStamp"

    def test_get_model_fsm_transition(self):
        """Test getting ModelFSMTransition for FSM_TRANSITION type."""
        model = EntityRegistry.get_model(EnumEntityType.FSM_TRANSITION)
        assert model == ModelFSMTransition
        assert model.__name__ == "ModelFSMTransition"

    def test_get_model_bridge_state(self):
        """Test getting ModelBridgeState for BRIDGE_STATE type."""
        model = EntityRegistry.get_model(EnumEntityType.BRIDGE_STATE)
        assert model == ModelBridgeState
        assert model.__name__ == "ModelBridgeState"

    def test_get_model_node_heartbeat(self):
        """Test getting ModelNodeHeartbeat for NODE_HEARTBEAT type."""
        model = EntityRegistry.get_model(EnumEntityType.NODE_HEARTBEAT)
        assert model == ModelNodeHeartbeat
        assert model.__name__ == "ModelNodeHeartbeat"

    def test_get_model_node_health_metrics(self):
        """Test getting ModelNodeHealthMetrics for NODE_HEALTH_METRICS type."""
        model = EntityRegistry.get_model(EnumEntityType.NODE_HEALTH_METRICS)
        assert model == ModelNodeHealthMetrics
        assert model.__name__ == "ModelNodeHealthMetrics"


class TestEntityRegistryTableMapping:
    """Test entity type to table name mapping."""

    def test_get_table_name_workflow_execution(self):
        """Test getting table name for WORKFLOW_EXECUTION type."""
        table = EntityRegistry.get_table_name(EnumEntityType.WORKFLOW_EXECUTION)
        assert table == "workflow_executions"

    def test_get_table_name_workflow_step(self):
        """Test getting table name for WORKFLOW_STEP type."""
        table = EntityRegistry.get_table_name(EnumEntityType.WORKFLOW_STEP)
        assert table == "workflow_steps"

    def test_get_table_name_metadata_stamp(self):
        """Test getting table name for METADATA_STAMP type."""
        table = EntityRegistry.get_table_name(EnumEntityType.METADATA_STAMP)
        assert table == "metadata_stamps"

    def test_get_table_name_fsm_transition(self):
        """Test getting table name for FSM_TRANSITION type."""
        table = EntityRegistry.get_table_name(EnumEntityType.FSM_TRANSITION)
        assert table == "fsm_transitions"

    def test_get_table_name_bridge_state(self):
        """Test getting table name for BRIDGE_STATE type."""
        table = EntityRegistry.get_table_name(EnumEntityType.BRIDGE_STATE)
        assert table == "bridge_states"

    def test_get_table_name_node_heartbeat(self):
        """Test getting table name for NODE_HEARTBEAT type."""
        table = EntityRegistry.get_table_name(EnumEntityType.NODE_HEARTBEAT)
        assert table == "node_registrations"

    def test_get_table_name_node_health_metrics(self):
        """Test getting table name for NODE_HEALTH_METRICS type."""
        table = EntityRegistry.get_table_name(EnumEntityType.NODE_HEALTH_METRICS)
        assert table == "node_registrations"  # Virtual entity, same table


class TestEntityRegistryValidation:
    """Test entity validation functionality."""

    def test_validate_workflow_execution_valid(self):
        """Test validating valid workflow execution data."""
        entity_data = {
            "correlation_id": uuid4(),
            "workflow_type": "metadata_stamping",
            "current_state": "PROCESSING",
            "namespace": "production",
            "started_at": datetime.now(UTC),
        }

        validated = EntityRegistry.validate_entity(
            EnumEntityType.WORKFLOW_EXECUTION, entity_data
        )

        assert isinstance(validated, ModelWorkflowExecution)
        assert validated.workflow_type == "metadata_stamping"
        assert validated.current_state == "PROCESSING"
        assert validated.namespace == "production"

    def test_validate_workflow_step_valid(self):
        """Test validating valid workflow step data."""
        entity_data = {
            "workflow_id": uuid4(),
            "step_name": "hash_generation",
            "step_order": 1,
            "status": "completed",
        }

        validated = EntityRegistry.validate_entity(
            EnumEntityType.WORKFLOW_STEP, entity_data
        )

        assert isinstance(validated, ModelWorkflowStep)
        assert validated.step_name == "hash_generation"
        assert validated.step_order == 1
        assert validated.status == "completed"

    def test_validate_metadata_stamp_valid(self):
        """Test validating valid metadata stamp data."""
        entity_data = {
            "file_hash": "a" * 64,  # 64-char hex hash
            "stamp_data": {"stamp_type": "inline"},
            "namespace": "production",
        }

        validated = EntityRegistry.validate_entity(
            EnumEntityType.METADATA_STAMP, entity_data
        )

        assert isinstance(validated, ModelMetadataStamp)
        assert validated.file_hash == "a" * 64
        assert validated.namespace == "production"

    def test_validate_fsm_transition_valid(self):
        """Test validating valid FSM transition data."""
        entity_data = {
            "entity_id": uuid4(),
            "entity_type": "workflow_execution",
            "to_state": "COMPLETED",
            "transition_event": "WORKFLOW_COMPLETED",
        }

        validated = EntityRegistry.validate_entity(
            EnumEntityType.FSM_TRANSITION, entity_data
        )

        assert isinstance(validated, ModelFSMTransition)
        assert validated.entity_type == "workflow_execution"
        assert validated.to_state == "COMPLETED"

    def test_validate_bridge_state_valid(self):
        """Test validating valid bridge state data."""
        entity_data = {
            "bridge_id": uuid4(),
            "namespace": "production",
            "current_fsm_state": "PROCESSING",
        }

        validated = EntityRegistry.validate_entity(
            EnumEntityType.BRIDGE_STATE, entity_data
        )

        assert isinstance(validated, ModelBridgeState)
        assert validated.namespace == "production"
        assert validated.current_fsm_state == "PROCESSING"

    def test_validate_node_heartbeat_valid(self):
        """Test validating valid node heartbeat data."""
        entity_data = {
            "node_id": "orchestrator-001",
            "node_type": "orchestrator",
            "node_version": "1.0.0",
        }

        validated = EntityRegistry.validate_entity(
            EnumEntityType.NODE_HEARTBEAT, entity_data
        )

        assert isinstance(validated, ModelNodeHeartbeat)
        assert validated.node_id == "orchestrator-001"
        assert validated.node_type == "orchestrator"

    def test_validate_node_health_metrics_valid(self):
        """Test validating valid node health metrics data."""
        entity_data = {
            "node_id": "orchestrator-001",
            "node_type": "orchestrator",
            "health_status": "HEALTHY",
        }

        validated = EntityRegistry.validate_entity(
            EnumEntityType.NODE_HEALTH_METRICS, entity_data
        )

        assert isinstance(validated, ModelNodeHealthMetrics)
        assert validated.node_id == "orchestrator-001"
        assert validated.health_status == "HEALTHY"

    def test_validate_entity_invalid_data(self):
        """Test validation with invalid data raises ValidationError."""
        entity_data = {
            # Missing required fields
            "workflow_type": "test",
        }

        with pytest.raises(ValidationError):
            EntityRegistry.validate_entity(
                EnumEntityType.WORKFLOW_EXECUTION, entity_data
            )


class TestEntityRegistrySerialization:
    """Test entity serialization functionality."""

    def test_serialize_entity_excludes_auto_managed_fields(self):
        """Test that serialization excludes auto-managed fields."""
        execution = ModelWorkflowExecution(
            id=uuid4(),
            correlation_id=uuid4(),
            workflow_type="metadata_stamping",
            current_state="PROCESSING",
            namespace="production",
            started_at=datetime.now(UTC),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        serialized = EntityRegistry.serialize_entity(execution)

        # Auto-managed fields should be excluded
        assert "id" not in serialized
        assert "created_at" not in serialized
        assert "updated_at" not in serialized

        # Required fields should be included
        assert "correlation_id" in serialized
        assert "workflow_type" in serialized
        assert "current_state" in serialized
        assert "namespace" in serialized
        assert "started_at" in serialized

    def test_serialize_entity_excludes_none_values(self):
        """Test that serialization excludes None values."""
        execution = ModelWorkflowExecution(
            correlation_id=uuid4(),
            workflow_type="metadata_stamping",
            current_state="PROCESSING",
            namespace="production",
            started_at=datetime.now(UTC),
            completed_at=None,  # None value should be excluded
            error_message=None,  # None value should be excluded
        )

        serialized = EntityRegistry.serialize_entity(execution)

        # None values should be excluded
        assert "completed_at" not in serialized
        assert "error_message" not in serialized

        # Non-None required fields should be included
        assert "correlation_id" in serialized
        assert "workflow_type" in serialized

    def test_serialize_entity_with_jsonb_fields(self):
        """Test serialization with JSONB fields converts dicts to JSON strings."""
        # Create metadata stamp with JSONB fields
        stamp = ModelMetadataStamp(
            file_hash="a" * 64,
            stamp_data={"stamp_type": "inline", "protocol_version": "1.0"},
            namespace="production",
        )

        serialized = EntityRegistry.serialize_entity(stamp)

        # JSONB fields should be JSON strings
        assert "stamp_data" in serialized
        assert isinstance(serialized["stamp_data"], str)
        parsed_stamp = json.loads(serialized["stamp_data"])
        assert parsed_stamp["stamp_type"] == "inline"
        assert parsed_stamp["protocol_version"] == "1.0"

    def test_serialize_entity_different_entity_types(self):
        """Test serialization works for all entity types."""
        # Test workflow execution
        execution = ModelWorkflowExecution(
            correlation_id=uuid4(),
            workflow_type="test",
            current_state="PROCESSING",
            namespace="test",
            started_at=datetime.now(UTC),
        )
        serialized_exec = EntityRegistry.serialize_entity(execution)
        assert "correlation_id" in serialized_exec
        assert "id" not in serialized_exec

        # Test workflow step
        step = ModelWorkflowStep(
            workflow_id=uuid4(),
            step_name="test_step",
            step_order=1,
            status="pending",
        )
        serialized_step = EntityRegistry.serialize_entity(step)
        assert "workflow_id" in serialized_step
        assert "id" not in serialized_step

        # Test FSM transition
        transition = ModelFSMTransition(
            entity_id=uuid4(),
            entity_type="workflow",
            to_state="COMPLETED",
            transition_event="COMPLETED",
        )
        serialized_trans = EntityRegistry.serialize_entity(transition)
        assert "entity_id" in serialized_trans
        assert "id" not in serialized_trans


class TestEntityRegistryDeserialization:
    """Test entity deserialization functionality."""

    def test_deserialize_row_basic(self):
        """Test deserializing database row to entity model."""
        row = {
            "id": uuid4(),
            "correlation_id": uuid4(),
            "workflow_type": "metadata_stamping",
            "current_state": "PROCESSING",
            "namespace": "production",
            "started_at": datetime.now(UTC),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }

        entity = EntityRegistry.deserialize_row(EnumEntityType.WORKFLOW_EXECUTION, row)

        assert isinstance(entity, ModelWorkflowExecution)
        assert entity.id == row["id"]
        assert entity.correlation_id == row["correlation_id"]
        assert entity.workflow_type == row["workflow_type"]

    def test_deserialize_row_with_jsonb_fields(self):
        """Test deserializing row with JSONB fields as JSON strings."""
        # Simulate database row with JSONB fields as JSON strings
        row = {
            "id": uuid4(),
            "file_hash": "a" * 64,
            "stamp_data": json.dumps(
                {"stamp_type": "inline", "protocol_version": "1.0"}
            ),
            "namespace": "production",
            "created_at": datetime.now(UTC),
        }

        entity = EntityRegistry.deserialize_row(EnumEntityType.METADATA_STAMP, row)

        assert isinstance(entity, ModelMetadataStamp)
        assert isinstance(entity.stamp_data, dict)
        assert entity.stamp_data["stamp_type"] == "inline"
        assert entity.stamp_data["protocol_version"] == "1.0"

    def test_deserialize_row_with_malformed_json(self):
        """Test deserializing row with malformed JSON in JSONB fields."""
        row = {
            "id": uuid4(),
            "file_hash": "a" * 64,
            "stamp_data": "not valid json{",  # Invalid JSON
            "namespace": "production",
            "created_at": datetime.now(UTC),
        }

        # Should raise ValidationError due to malformed JSON
        with pytest.raises(ValidationError):
            EntityRegistry.deserialize_row(EnumEntityType.METADATA_STAMP, row)

    def test_deserialize_row_with_dict_jsonb_fields(self):
        """Test deserializing row where JSONB fields are already dicts."""
        # Some database drivers might return JSONB as dicts directly
        row = {
            "id": uuid4(),
            "file_hash": "a" * 64,
            "stamp_data": {"stamp_type": "inline", "version": "1.0"},  # Already a dict
            "namespace": "production",
            "created_at": datetime.now(UTC),
        }

        entity = EntityRegistry.deserialize_row(EnumEntityType.METADATA_STAMP, row)

        assert isinstance(entity, ModelMetadataStamp)
        assert entity.stamp_data == {"stamp_type": "inline", "version": "1.0"}


class TestEntityRegistryJSONBFieldDetection:
    """Test JSONB field detection logic."""

    def test_is_jsonb_field_with_explicit_metadata(self):
        """Test _is_jsonb_field with explicit json_schema_extra metadata."""

        class TestModel(BaseModel):
            jsonb_field: dict[str, Any] = Field(
                default_factory=dict, json_schema_extra={"db_type": "jsonb"}
            )
            regular_field: str = "test"

        field_info_jsonb = TestModel.model_fields["jsonb_field"]
        field_info_regular = TestModel.model_fields["regular_field"]

        assert EntityRegistry._is_jsonb_field(field_info_jsonb) is True
        assert EntityRegistry._is_jsonb_field(field_info_regular) is False

    def test_is_jsonb_field_with_dict_str_any_type(self):
        """Test _is_jsonb_field with dict[str, Any] type annotation."""

        class TestModel(BaseModel):
            dict_field: dict[str, Any] = Field(default_factory=dict)
            str_field: str = "test"
            int_field: int = 0

        field_info_dict = TestModel.model_fields["dict_field"]
        field_info_str = TestModel.model_fields["str_field"]
        field_info_int = TestModel.model_fields["int_field"]

        assert EntityRegistry._is_jsonb_field(field_info_dict) is True
        assert EntityRegistry._is_jsonb_field(field_info_str) is False
        assert EntityRegistry._is_jsonb_field(field_info_int) is False

    def test_is_jsonb_field_with_optional_dict(self):
        """Test _is_jsonb_field with Optional[dict[str, Any]] type."""

        class TestModel(BaseModel):
            optional_dict: Optional[dict[str, Any]] = None
            required_dict: dict[str, Any] = Field(default_factory=dict)

        field_info_optional = TestModel.model_fields["optional_dict"]
        field_info_required = TestModel.model_fields["required_dict"]

        assert EntityRegistry._is_jsonb_field(field_info_optional) is True
        assert EntityRegistry._is_jsonb_field(field_info_required) is True

    def test_is_jsonb_field_with_dict_int_any(self):
        """Test _is_jsonb_field with dict[int, Any] (should be False)."""

        class TestModel(BaseModel):
            int_key_dict: dict[int, Any] = Field(default_factory=dict)

        field_info = TestModel.model_fields["int_key_dict"]

        # Should be False because key type is not str
        assert EntityRegistry._is_jsonb_field(field_info) is False

    def test_is_jsonb_field_with_dict_str_int(self):
        """Test _is_jsonb_field with dict[str, int] (should be False)."""

        class TestModel(BaseModel):
            str_int_dict: dict[str, int] = Field(default_factory=dict)

        field_info = TestModel.model_fields["str_int_dict"]

        # Should be False because value type is not Any
        assert EntityRegistry._is_jsonb_field(field_info) is False


class TestEntityRegistryUtilityMethods:
    """Test utility methods."""

    def test_is_registered_all_types(self):
        """Test that all entity types are registered."""
        for entity_type in EnumEntityType:
            assert EntityRegistry.is_registered(entity_type) is True

    def test_get_primary_key_field(self):
        """Test getting primary key field name."""
        pk_field = EntityRegistry.get_primary_key_field(
            EnumEntityType.WORKFLOW_EXECUTION
        )
        assert pk_field == "id"

    def test_get_primary_key_field_all_types(self):
        """Test primary key field for all entity types."""
        for entity_type in EnumEntityType:
            pk_field = EntityRegistry.get_primary_key_field(entity_type)
            assert pk_field == "id"

    def test_get_entity_fields(self):
        """Test getting entity field names."""
        fields = EntityRegistry.get_entity_fields(EnumEntityType.WORKFLOW_EXECUTION)

        assert "id" in fields
        assert "correlation_id" in fields
        assert "workflow_type" in fields
        assert "current_state" in fields
        assert "namespace" in fields
        assert isinstance(fields, list)

    def test_get_entity_fields_all_types(self):
        """Test getting fields for all entity types."""
        for entity_type in EnumEntityType:
            fields = EntityRegistry.get_entity_fields(entity_type)
            assert isinstance(fields, list)
            assert len(fields) > 0
            # All entity types should have some identifier field
            # (most use 'id', but ModelBridgeState uses 'bridge_id')
            pk_field = EntityRegistry.get_primary_key_field(entity_type)
            # Primary key is 'id' by convention, but we just check fields exist
            assert len(fields) >= 1


class TestEntityRegistryErrorHandling:
    """Test error handling."""

    def test_get_model_invalid_type_raises_value_error(self):
        """Test that get_model with invalid type raises ValueError."""
        # This would only happen if someone bypasses the enum
        # but we test it for completeness
        with pytest.raises(ValueError, match="No model registered for entity type"):
            # Trying to use a string instead of enum will raise ValueError
            EntityRegistry.get_model("invalid_type")  # type: ignore

    def test_get_table_name_invalid_type_raises_value_error(self):
        """Test that get_table_name with invalid type raises ValueError."""
        with pytest.raises(ValueError, match="No table registered for entity type"):
            EntityRegistry.get_table_name("invalid_type")  # type: ignore

    def test_validate_entity_missing_required_fields(self):
        """Test validation with missing required fields."""
        with pytest.raises(ValidationError):
            EntityRegistry.validate_entity(
                EnumEntityType.WORKFLOW_EXECUTION,
                {
                    "workflow_type": "test"
                },  # Missing correlation_id, current_state, etc.
            )

    def test_validate_entity_invalid_field_types(self):
        """Test validation with invalid field types."""
        with pytest.raises(ValidationError):
            EntityRegistry.validate_entity(
                EnumEntityType.WORKFLOW_EXECUTION,
                {
                    "correlation_id": "not-a-uuid",  # Should be UUID
                    "workflow_type": "test",
                    "current_state": "PROCESSING",
                    "namespace": "test",
                    "started_at": datetime.now(UTC),
                },
            )


class TestEntityRegistryIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_execution_cycle(self):
        """Test complete workflow: validate → serialize → deserialize."""
        # Step 1: Validate entity data
        entity_data = {
            "correlation_id": uuid4(),
            "workflow_type": "metadata_stamping",
            "current_state": "PROCESSING",
            "namespace": "production",
            "started_at": datetime.now(UTC),
        }

        validated = EntityRegistry.validate_entity(
            EnumEntityType.WORKFLOW_EXECUTION, entity_data
        )

        # Step 2: Serialize for database insertion
        serialized = EntityRegistry.serialize_entity(validated)

        assert "id" not in serialized
        assert "correlation_id" in serialized

        # Step 3: Simulate database row with auto-managed fields
        row = {
            "id": uuid4(),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            **serialized,
        }

        # Step 4: Deserialize database row back to entity
        deserialized = EntityRegistry.deserialize_row(
            EnumEntityType.WORKFLOW_EXECUTION, row
        )

        assert isinstance(deserialized, ModelWorkflowExecution)
        assert deserialized.correlation_id == entity_data["correlation_id"]
        assert deserialized.workflow_type == entity_data["workflow_type"]

    def test_complete_metadata_stamp_cycle_with_jsonb(self):
        """Test complete workflow with JSONB fields."""
        # Step 1: Validate entity data with JSONB fields
        entity_data = {
            "file_hash": "a" * 64,
            "stamp_data": {"stamp_type": "inline", "protocol_version": "1.0"},
            "namespace": "production",
        }

        validated = EntityRegistry.validate_entity(
            EnumEntityType.METADATA_STAMP, entity_data
        )

        # Step 2: Serialize for database insertion (JSONB → JSON strings)
        serialized = EntityRegistry.serialize_entity(validated)

        assert isinstance(serialized["stamp_data"], str)

        # Step 3: Simulate database row
        row = {
            "id": uuid4(),
            "created_at": datetime.now(UTC),
            **serialized,
        }

        # Step 4: Deserialize database row back to entity (JSON strings → dicts)
        deserialized = EntityRegistry.deserialize_row(
            EnumEntityType.METADATA_STAMP, row
        )

        assert isinstance(deserialized, ModelMetadataStamp)
        assert isinstance(deserialized.stamp_data, dict)
        assert deserialized.stamp_data["stamp_type"] == "inline"
        assert deserialized.stamp_data["protocol_version"] == "1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
