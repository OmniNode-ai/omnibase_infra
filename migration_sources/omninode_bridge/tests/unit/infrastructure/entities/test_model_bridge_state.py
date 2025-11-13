#!/usr/bin/env python3
"""Unit tests for ModelBridgeState.

Tests cover:
- Model instantiation with all fields
- Model validation
- Default values
- JSON serialization/deserialization
- Field validation
- Model configuration
"""

import json
from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omninode_bridge.infrastructure.entities.model_bridge_state import ModelBridgeState


class TestModelBridgeState:
    """Test suite for ModelBridgeState."""

    def test_model_instantiation_with_all_fields(self):
        """Test model instantiation with all required fields."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"
        last_aggregation_timestamp = datetime.now()
        created_at = datetime.now()
        updated_at = datetime.now()

        bridge_state = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            total_workflows_processed=100,
            total_items_aggregated=1000,
            aggregation_metadata={"avg_processing_time": 1.5},
            current_fsm_state=current_fsm_state,
            last_aggregation_timestamp=last_aggregation_timestamp,
            created_at=created_at,
            updated_at=updated_at,
        )

        assert bridge_state.bridge_id == bridge_id
        assert bridge_state.namespace == namespace
        assert bridge_state.total_workflows_processed == 100
        assert bridge_state.total_items_aggregated == 1000
        assert bridge_state.aggregation_metadata == {"avg_processing_time": 1.5}
        assert bridge_state.current_fsm_state == current_fsm_state
        assert bridge_state.last_aggregation_timestamp == last_aggregation_timestamp
        assert bridge_state.created_at == created_at
        assert bridge_state.updated_at == updated_at

    def test_model_instantiation_with_required_fields_only(self):
        """Test model instantiation with only required fields."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"

        bridge_state = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            current_fsm_state=current_fsm_state,
        )

        assert bridge_state.bridge_id == bridge_id
        assert bridge_state.namespace == namespace
        assert bridge_state.current_fsm_state == current_fsm_state
        assert bridge_state.total_workflows_processed == 0  # Default value
        assert bridge_state.total_items_aggregated == 0  # Default value
        assert bridge_state.aggregation_metadata == {}  # Default value
        assert bridge_state.last_aggregation_timestamp is None  # Default value
        assert bridge_state.created_at is None  # Default value
        assert bridge_state.updated_at is None  # Default value

    def test_model_validation_missing_required_fields(self):
        """Test model validation with missing required fields."""
        bridge_id = uuid4()
        # Missing namespace and current_fsm_state

        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeState(bridge_id=bridge_id)

        errors = exc_info.value.errors()
        assert len(errors) == 2
        assert any("namespace" in str(error) for error in errors)
        assert any("current_fsm_state" in str(error) for error in errors)

    def test_model_validation_invalid_namespace(self):
        """Test model validation with invalid namespace."""
        bridge_id = uuid4()
        current_fsm_state = "PROCESSING"

        # Empty namespace
        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeState(
                bridge_id=bridge_id, namespace="", current_fsm_state=current_fsm_state
            )

        assert "namespace" in str(exc_info.value)

        # Too long namespace
        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeState(
                bridge_id=bridge_id,
                namespace="a" * 256,  # 256 characters, max is 255
                current_fsm_state=current_fsm_state,
            )

        assert "namespace" in str(exc_info.value)

    def test_model_validation_invalid_counters(self):
        """Test model validation with invalid counter values."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"

        # Negative total_workflows_processed
        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeState(
                bridge_id=bridge_id,
                namespace=namespace,
                total_workflows_processed=-1,
                current_fsm_state=current_fsm_state,
            )

        assert "total_workflows_processed" in str(exc_info.value)

        # Negative total_items_aggregated
        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeState(
                bridge_id=bridge_id,
                namespace=namespace,
                total_items_aggregated=-1,
                current_fsm_state=current_fsm_state,
            )

        assert "total_items_aggregated" in str(exc_info.value)

    def test_model_validation_invalid_fsm_state(self):
        """Test model validation with invalid FSM state."""
        bridge_id = uuid4()
        namespace = "test.namespace"

        # Empty current_fsm_state
        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeState(
                bridge_id=bridge_id, namespace=namespace, current_fsm_state=""
            )

        assert "current_fsm_state" in str(exc_info.value)

        # Too long current_fsm_state
        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeState(
                bridge_id=bridge_id,
                namespace=namespace,
                current_fsm_state="a" * 51,  # 51 characters, max is 50
            )

        assert "current_fsm_state" in str(exc_info.value)

    def test_model_serialization(self):
        """Test model serialization to JSON."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"

        bridge_state = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            current_fsm_state=current_fsm_state,
        )

        # Test model_dump
        data = bridge_state.model_dump()
        assert (
            data["bridge_id"] == bridge_id
        )  # UUID is not converted to string in model_dump
        assert data["namespace"] == namespace
        assert data["current_fsm_state"] == current_fsm_state
        assert data["total_workflows_processed"] == 0
        assert data["total_items_aggregated"] == 0
        assert data["aggregation_metadata"] == {}

        # Test model_dump_json
        json_str = bridge_state.model_dump_json()
        parsed_data = json.loads(json_str)
        assert parsed_data["bridge_id"] == str(
            bridge_id
        )  # UUID is converted to string in JSON
        assert parsed_data["namespace"] == namespace
        assert parsed_data["current_fsm_state"] == current_fsm_state

    def test_model_deserialization(self):
        """Test model deserialization from JSON."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"

        data = {
            "bridge_id": str(bridge_id),
            "namespace": namespace,
            "current_fsm_state": current_fsm_state,
            "total_workflows_processed": 50,
            "total_items_aggregated": 500,
            "aggregation_metadata": {"avg_time": 1.2},
        }

        # Test model_validate
        bridge_state = ModelBridgeState.model_validate(data)
        assert bridge_state.bridge_id == bridge_id
        assert bridge_state.namespace == namespace
        assert bridge_state.current_fsm_state == current_fsm_state
        assert bridge_state.total_workflows_processed == 50
        assert bridge_state.total_items_aggregated == 500
        assert bridge_state.aggregation_metadata == {"avg_time": 1.2}

    def test_model_roundtrip(self):
        """Test model serialization and deserialization roundtrip."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"

        original = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            total_workflows_processed=100,
            total_items_aggregated=1000,
            aggregation_metadata={"avg_processing_time": 1.5},
            current_fsm_state=current_fsm_state,
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = ModelBridgeState.model_validate(data)

        # Verify all fields match
        assert restored.bridge_id == original.bridge_id
        assert restored.namespace == original.namespace
        assert restored.total_workflows_processed == original.total_workflows_processed
        assert restored.total_items_aggregated == original.total_items_aggregated
        assert restored.aggregation_metadata == original.aggregation_metadata
        assert restored.current_fsm_state == original.current_fsm_state

    def test_model_equality(self):
        """Test model equality comparison."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"

        bridge_state1 = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            current_fsm_state=current_fsm_state,
        )

        bridge_state2 = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            current_fsm_state=current_fsm_state,
        )

        bridge_state3 = ModelBridgeState(
            bridge_id=uuid4(),  # Different ID
            namespace=namespace,
            current_fsm_state=current_fsm_state,
        )

        # Test equality
        assert bridge_state1.model_dump() == bridge_state2.model_dump()
        assert bridge_state1.model_dump() != bridge_state3.model_dump()

    def test_model_copy(self):
        """Test model copying."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"

        original = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            total_workflows_processed=100,
            current_fsm_state=current_fsm_state,
        )

        # Test model_copy
        copied = original.model_copy()
        assert copied.bridge_id == original.bridge_id
        assert copied.namespace == original.namespace
        assert copied.total_workflows_processed == original.total_workflows_processed
        assert copied.current_fsm_state == original.current_fsm_state
        assert copied is not original  # Different instances

    def test_model_update(self):
        """Test model field updates."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"

        bridge_state = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            current_fsm_state=current_fsm_state,
        )

        # Update fields
        updated = bridge_state.model_copy(
            update={"total_workflows_processed": 200, "current_fsm_state": "COMPLETED"}
        )

        assert updated.total_workflows_processed == 200
        assert updated.current_fsm_state == "COMPLETED"
        # Original should be unchanged
        assert bridge_state.total_workflows_processed == 0
        assert bridge_state.current_fsm_state == "PROCESSING"

    def test_model_aggregation_metadata(self):
        """Test aggregation metadata field."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"

        # Test with empty metadata
        bridge_state = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            current_fsm_state=current_fsm_state,
        )

        assert bridge_state.aggregation_metadata == {}

        # Test with metadata
        metadata = {
            "avg_processing_time": 1.5,
            "max_batch_size": 100,
            "namespaces": ["ns1", "ns2"],
        }

        bridge_state = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            current_fsm_state=current_fsm_state,
            aggregation_metadata=metadata,
        )

        assert bridge_state.aggregation_metadata == metadata

    def test_model_jsonb_validation(self):
        """Test JSONB field validation."""
        bridge_id = uuid4()
        namespace = "test.namespace"
        current_fsm_state = "PROCESSING"

        # This should not raise an error if validation is working correctly
        bridge_state = ModelBridgeState(
            bridge_id=bridge_id,
            namespace=namespace,
            current_fsm_state=current_fsm_state,
            aggregation_metadata={"test": "value"},
        )

        # The _validate_jsonb_fields method should be called during initialization
        # If it's working correctly, no exception should be raised
