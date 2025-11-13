#!/usr/bin/env python3
"""Unit tests for ModelWorkflowState.

Tests cover:
- Model instantiation with all fields
- Model validation
- Default values
- JSON serialization/deserialization
- Field validation (version, state, provenance)
- Model configuration
- Optimistic concurrency control patterns

Reference: docs/planning/PURE_REDUCER_REFACTOR_PLAN.md (Wave 1, Workstream 1A)
"""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from omninode_bridge.infrastructure.entities.model_workflow_state import (
    ModelWorkflowState,
)


class TestModelWorkflowState:
    """Test suite for ModelWorkflowState."""

    def test_model_instantiation_with_all_fields(self):
        """Test model instantiation with all required fields."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": [], "count": 0, "namespace": "production"}
        updated_at = datetime.now(UTC)
        schema_version = 1
        provenance = {
            "effect_id": "effect-456",
            "timestamp": "2025-10-21T12:00:00Z",
            "action_id": "action-789",
        }

        workflow_state = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            schema_version=schema_version,
            provenance=provenance,
        )

        assert workflow_state.workflow_key == workflow_key
        assert workflow_state.version == version
        assert workflow_state.state == state
        assert workflow_state.updated_at == updated_at
        assert workflow_state.schema_version == schema_version
        assert workflow_state.provenance == provenance

    def test_model_instantiation_with_required_fields_only(self):
        """Test model instantiation with only required fields."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": [], "count": 0}
        updated_at = datetime.now(UTC)
        provenance = {
            "effect_id": "effect-456",
            "timestamp": "2025-10-21T12:00:00Z",
        }

        workflow_state = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        assert workflow_state.workflow_key == workflow_key
        assert workflow_state.version == version
        assert workflow_state.state == state
        assert workflow_state.updated_at == updated_at
        assert workflow_state.schema_version == 1  # Default value
        assert workflow_state.provenance == provenance

    def test_model_validation_missing_required_fields(self):
        """Test model validation with missing required fields."""
        workflow_key = "workflow-123"
        # Missing version, state, updated_at, provenance

        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowState(workflow_key=workflow_key)

        errors = exc_info.value.errors()
        assert len(errors) == 4
        assert any("version" in str(error) for error in errors)
        assert any("state" in str(error) for error in errors)
        assert any("updated_at" in str(error) for error in errors)
        assert any("provenance" in str(error) for error in errors)

    def test_model_validation_invalid_workflow_key(self):
        """Test model validation with invalid workflow_key."""
        version = 1
        state = {"items": []}
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        # Empty workflow_key
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowState(
                workflow_key="",
                version=version,
                state=state,
                updated_at=updated_at,
                provenance=provenance,
            )

        assert "workflow_key" in str(exc_info.value)

        # Too long workflow_key
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowState(
                workflow_key="a" * 256,  # 256 characters, max is 255
                version=version,
                state=state,
                updated_at=updated_at,
                provenance=provenance,
            )

        assert "workflow_key" in str(exc_info.value)

    def test_model_validation_invalid_version(self):
        """Test model validation with invalid version."""
        workflow_key = "workflow-123"
        state = {"items": []}
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        # Zero version
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowState(
                workflow_key=workflow_key,
                version=0,
                state=state,
                updated_at=updated_at,
                provenance=provenance,
            )

        assert "version" in str(exc_info.value).lower()

        # Negative version
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowState(
                workflow_key=workflow_key,
                version=-1,
                state=state,
                updated_at=updated_at,
                provenance=provenance,
            )

        assert "version" in str(exc_info.value).lower()

    def test_model_validation_empty_state(self):
        """Test model validation with empty state."""
        workflow_key = "workflow-123"
        version = 1
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        # Empty state dict
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowState(
                workflow_key=workflow_key,
                version=version,
                state={},
                updated_at=updated_at,
                provenance=provenance,
            )

        assert "state" in str(exc_info.value).lower()

    def test_model_validation_invalid_schema_version(self):
        """Test model validation with invalid schema_version."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": []}
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        # Zero schema_version
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowState(
                workflow_key=workflow_key,
                version=version,
                state=state,
                updated_at=updated_at,
                schema_version=0,
                provenance=provenance,
            )

        assert "schema_version" in str(exc_info.value).lower()

        # Negative schema_version
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowState(
                workflow_key=workflow_key,
                version=version,
                state=state,
                updated_at=updated_at,
                schema_version=-1,
                provenance=provenance,
            )

        assert "schema_version" in str(exc_info.value).lower()

    def test_model_validation_missing_provenance_fields(self):
        """Test model validation with missing required provenance fields."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": []}
        updated_at = datetime.now(UTC)

        # Missing effect_id
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowState(
                workflow_key=workflow_key,
                version=version,
                state=state,
                updated_at=updated_at,
                provenance={"timestamp": "2025-10-21T12:00:00Z"},
            )

        assert "effect_id" in str(exc_info.value)

        # Missing timestamp
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowState(
                workflow_key=workflow_key,
                version=version,
                state=state,
                updated_at=updated_at,
                provenance={"effect_id": "effect-456"},
            )

        assert "timestamp" in str(exc_info.value)

    def test_model_serialization(self):
        """Test model serialization to JSON."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": [], "count": 0}
        updated_at = datetime.now(UTC)
        provenance = {
            "effect_id": "effect-456",
            "timestamp": "2025-10-21T12:00:00Z",
        }

        workflow_state = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        # Test model_dump
        data = workflow_state.model_dump()
        assert data["workflow_key"] == workflow_key
        assert data["version"] == version
        assert data["state"] == state
        assert data["updated_at"] == updated_at
        assert data["schema_version"] == 1
        assert data["provenance"] == provenance

        # Test model_dump_json
        json_str = workflow_state.model_dump_json()
        parsed_data = json.loads(json_str)
        assert parsed_data["workflow_key"] == workflow_key
        assert parsed_data["version"] == version
        assert parsed_data["state"] == state
        assert parsed_data["schema_version"] == 1
        assert parsed_data["provenance"] == provenance

    def test_model_deserialization(self):
        """Test model deserialization from JSON."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": [], "count": 0}
        updated_at = datetime.now(UTC)
        provenance = {
            "effect_id": "effect-456",
            "timestamp": "2025-10-21T12:00:00Z",
        }

        data = {
            "workflow_key": workflow_key,
            "version": version,
            "state": state,
            "updated_at": updated_at.isoformat(),
            "schema_version": 1,
            "provenance": provenance,
        }

        # Test model_validate
        workflow_state = ModelWorkflowState.model_validate(data)
        assert workflow_state.workflow_key == workflow_key
        assert workflow_state.version == version
        assert workflow_state.state == state
        assert workflow_state.updated_at == updated_at
        assert workflow_state.schema_version == 1
        assert workflow_state.provenance == provenance

    def test_model_roundtrip(self):
        """Test model serialization and deserialization roundtrip."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": [], "count": 0, "namespace": "production"}
        updated_at = datetime.now(UTC)
        provenance = {
            "effect_id": "effect-456",
            "timestamp": "2025-10-21T12:00:00Z",
            "action_id": "action-789",
        }

        original = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = ModelWorkflowState.model_validate(data)

        # Verify all fields match
        assert restored.workflow_key == original.workflow_key
        assert restored.version == original.version
        assert restored.state == original.state
        assert restored.updated_at == original.updated_at
        assert restored.schema_version == original.schema_version
        assert restored.provenance == original.provenance

    def test_model_equality(self):
        """Test model equality comparison."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": []}
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        workflow_state1 = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        workflow_state2 = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        workflow_state3 = ModelWorkflowState(
            workflow_key="workflow-456",  # Different key
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        # Test equality
        assert workflow_state1.model_dump() == workflow_state2.model_dump()
        assert workflow_state1.model_dump() != workflow_state3.model_dump()

    def test_model_copy(self):
        """Test model copying."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": []}
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        original = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        # Test model_copy
        copied = original.model_copy()
        assert copied.workflow_key == original.workflow_key
        assert copied.version == original.version
        assert copied.state == original.state
        assert copied.updated_at == original.updated_at
        assert copied.provenance == original.provenance
        assert copied is not original  # Different instances

    def test_model_update_version(self):
        """Test model field updates for version increment."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": [], "count": 0}
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        workflow_state = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        # Update version and state
        new_updated_at = datetime.now(UTC)
        new_state = {"items": ["item1"], "count": 1}
        new_provenance = {
            "effect_id": "effect-789",
            "timestamp": "2025-10-21T12:05:00Z",
        }

        updated = workflow_state.model_copy(
            update={
                "version": version + 1,
                "state": new_state,
                "updated_at": new_updated_at,
                "provenance": new_provenance,
            }
        )

        assert updated.version == version + 1
        assert updated.state == new_state
        assert updated.updated_at == new_updated_at
        assert updated.provenance == new_provenance
        # Original should be unchanged
        assert workflow_state.version == version
        assert workflow_state.state == state

    def test_model_state_variations(self):
        """Test model with different state structures."""
        workflow_key = "workflow-123"
        version = 1
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        # Test with different state structures
        states = [
            {"items": [], "count": 0},
            {"namespace": "production", "aggregated_data": {"total_size": 1024}},
            {
                "workflow_id": "123",
                "current_state": "PROCESSING",
                "metadata": {"key": "value"},
            },
            {
                "complex": {
                    "nested": {"structure": {"with": ["arrays", "and", "dicts"]}}
                }
            },
        ]

        for state in states:
            workflow_state = ModelWorkflowState(
                workflow_key=workflow_key,
                version=version,
                state=state,
                updated_at=updated_at,
                provenance=provenance,
            )

            assert workflow_state.state == state

    def test_model_provenance_variations(self):
        """Test model with different provenance structures."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": []}
        updated_at = datetime.now(UTC)

        # Minimal required provenance
        minimal_provenance = {
            "effect_id": "effect-456",
            "timestamp": "2025-10-21T12:00:00Z",
        }

        workflow_state = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=minimal_provenance,
        )

        assert workflow_state.provenance == minimal_provenance

        # Extended provenance with optional fields
        extended_provenance = {
            "effect_id": "effect-456",
            "timestamp": "2025-10-21T12:00:00Z",
            "action_id": "action-789",
            "correlation_id": "correlation-012",
            "metadata": {"key": "value"},
        }

        workflow_state = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=extended_provenance,
        )

        assert workflow_state.provenance == extended_provenance

    def test_model_jsonb_validation(self):
        """Test JSONB field validation."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": [], "count": 0}
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        # This should not raise an error if validation is working correctly
        workflow_state = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        # The _validate_jsonb_fields method should be called during initialization
        # Verify JSONB fields have proper annotations
        state_field = ModelWorkflowState.model_fields["state"]
        provenance_field = ModelWorkflowState.model_fields["provenance"]

        assert state_field.json_schema_extra.get("db_type") == "jsonb"
        assert provenance_field.json_schema_extra.get("db_type") == "jsonb"

    def test_model_optimistic_concurrency_pattern(self):
        """Test optimistic concurrency control pattern."""
        workflow_key = "workflow-123"
        initial_version = 1
        state = {"items": [], "count": 0}
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        # Create initial state
        initial_state = ModelWorkflowState(
            workflow_key=workflow_key,
            version=initial_version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        # Simulate state update (version increment)
        updated_state = initial_state.model_copy(
            update={
                "version": initial_version + 1,
                "state": {"items": ["item1"], "count": 1},
                "updated_at": datetime.now(UTC),
                "provenance": {
                    "effect_id": "effect-789",
                    "timestamp": "2025-10-21T12:05:00Z",
                },
            }
        )

        # Verify version increment
        assert updated_state.version == initial_version + 1
        assert updated_state.state != initial_state.state

        # Simulate another concurrent update (version conflict scenario)
        concurrent_state = initial_state.model_copy(
            update={
                "version": initial_version + 1,
                "state": {"items": ["item2"], "count": 1},
                "updated_at": datetime.now(UTC),
                "provenance": {
                    "effect_id": "effect-012",
                    "timestamp": "2025-10-21T12:06:00Z",
                },
            }
        )

        # Both updates have version 2, but different state
        # In practice, the second commit would fail due to version conflict
        assert concurrent_state.version == updated_state.version
        assert concurrent_state.state != updated_state.state

    def test_model_schema_versioning(self):
        """Test schema versioning for future migrations."""
        workflow_key = "workflow-123"
        version = 1
        state = {"items": []}
        updated_at = datetime.now(UTC)
        provenance = {"effect_id": "effect-456", "timestamp": "2025-10-21T12:00:00Z"}

        # Test with default schema_version
        workflow_state_v1 = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            provenance=provenance,
        )

        assert workflow_state_v1.schema_version == 1

        # Test with explicit schema_version (future migration)
        workflow_state_v2 = ModelWorkflowState(
            workflow_key=workflow_key,
            version=version,
            state=state,
            updated_at=updated_at,
            schema_version=2,
            provenance=provenance,
        )

        assert workflow_state_v2.schema_version == 2
