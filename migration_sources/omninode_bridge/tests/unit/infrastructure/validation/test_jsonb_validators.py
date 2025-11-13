#!/usr/bin/env python3
"""
Unit tests for JSONB field validation utilities.

Tests verify that JSONB validators correctly enforce PostgreSQL JSONB type
annotations on Pydantic entity models.

ONEX v2.0 Compliance:
- Comprehensive validation testing
- Error message verification
- Multiple usage pattern testing
"""

from datetime import UTC
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

from omninode_bridge.infrastructure.validation import (
    JsonbField,
    JsonbValidatedModel,
    validate_jsonb_fields,
)


class TestJsonbFieldHelper:
    """Test suite for JsonbField() helper function."""

    def test_jsonb_field_adds_annotation(self):
        """Verify JsonbField() automatically adds json_schema_extra."""

        class TestModel(BaseModel):
            metadata: dict[str, Any] = JsonbField(
                default_factory=dict, description="Test metadata"
            )

        # Verify annotation is present
        field_info = TestModel.model_fields["metadata"]
        assert field_info.json_schema_extra == {"db_type": "jsonb"}

    def test_jsonb_field_preserves_other_config(self):
        """Verify JsonbField() preserves other field configuration."""

        class TestModel(BaseModel):
            metadata: dict[str, Any] = JsonbField(
                default_factory=dict,
                description="Test metadata",
                examples=[{"key": "value"}],
            )

        field_info = TestModel.model_fields["metadata"]
        assert field_info.description == "Test metadata"
        assert field_info.examples == [{"key": "value"}]

    def test_jsonb_field_with_default_value(self):
        """Verify JsonbField() works with default values."""

        class TestModel(BaseModel):
            metadata: dict[str, Any] = JsonbField(
                default={"initial": "value"}, description="Test metadata"
            )

        model = TestModel()
        assert model.metadata == {"initial": "value"}
        assert TestModel.model_fields["metadata"].json_schema_extra == {
            "db_type": "jsonb"
        }

    def test_jsonb_field_multiple_fields(self):
        """Verify multiple JsonbField() fields in same model."""

        class TestModel(BaseModel):
            metadata: dict[str, Any] = JsonbField(default_factory=dict)
            config: dict[str, Any] = JsonbField(default_factory=dict)
            settings: dict[str, Any] = JsonbField(default_factory=dict)

        for field_name in ["metadata", "config", "settings"]:
            field_info = TestModel.model_fields[field_name]
            assert field_info.json_schema_extra == {"db_type": "jsonb"}


class TestValidateJsonbFields:
    """Test suite for validate_jsonb_fields() validator function."""

    def test_validator_accepts_correct_annotation(self):
        """Verify validator accepts fields with correct JSONB annotation."""

        class TestModel(BaseModel):
            metadata: dict[str, Any] = Field(
                default_factory=dict, json_schema_extra={"db_type": "jsonb"}
            )

            @model_validator(mode="after")
            def _validate(self) -> "TestModel":
                return validate_jsonb_fields(self)

        # Should not raise any errors
        model = TestModel(metadata={"key": "value"})
        assert model.metadata == {"key": "value"}

    def test_validator_rejects_missing_annotation(self):
        """Verify validator rejects dict fields without JSONB annotation."""

        with pytest.raises(ValidationError) as exc_info:

            class TestModel(BaseModel):
                metadata: dict[str, Any] = Field(default_factory=dict)

                @model_validator(mode="after")
                def _validate(self) -> "TestModel":
                    return validate_jsonb_fields(self)

            TestModel()

        # Verify error message contains helpful guidance
        error_msg = str(exc_info.value)
        assert "missing required json_schema_extra" in error_msg
        assert "JsonbField()" in error_msg or "json_schema_extra" in error_msg

    def test_validator_rejects_wrong_db_type(self):
        """Verify validator rejects incorrect db_type value."""

        with pytest.raises(ValidationError) as exc_info:

            class TestModel(BaseModel):
                metadata: dict[str, Any] = Field(
                    default_factory=dict, json_schema_extra={"db_type": "json"}
                )

                @model_validator(mode="after")
                def _validate(self) -> "TestModel":
                    return validate_jsonb_fields(self)

            TestModel()

        error_msg = str(exc_info.value)
        assert "missing required json_schema_extra" in error_msg

    def test_validator_handles_non_dict_fields(self):
        """Verify validator ignores non-dict fields."""

        class TestModel(BaseModel):
            name: str = Field(...)
            count: int = Field(...)
            metadata: dict[str, Any] = JsonbField(default_factory=dict)

            @model_validator(mode="after")
            def _validate(self) -> "TestModel":
                return validate_jsonb_fields(self)

        # Should not raise errors for non-dict fields
        model = TestModel(name="test", count=42)
        assert model.name == "test"
        assert model.count == 42

    def test_validator_handles_invalid_json_schema_extra(self):
        """Verify validator handles invalid json_schema_extra gracefully."""

        with pytest.raises(ValidationError) as exc_info:

            class TestModel(BaseModel):
                metadata: dict[str, Any] = Field(
                    default_factory=dict, json_schema_extra="invalid"  # Should be dict
                )

                @model_validator(mode="after")
                def _validate(self) -> "TestModel":
                    return validate_jsonb_fields(self)

            TestModel()

        error_msg = str(exc_info.value)
        assert "invalid json_schema_extra" in error_msg


class TestJsonbValidatedModel:
    """Test suite for JsonbValidatedModel base class."""

    def test_base_class_validates_automatically(self):
        """Verify JsonbValidatedModel validates without explicit decorator."""

        class TestModel(JsonbValidatedModel):
            metadata: dict[str, Any] = JsonbField(default_factory=dict)

        # Validation happens automatically via base class
        model = TestModel(metadata={"key": "value"})
        assert model.metadata == {"key": "value"}

    def test_base_class_rejects_missing_annotation(self):
        """Verify base class rejects missing annotations."""

        with pytest.raises(ValidationError) as exc_info:

            class TestModel(JsonbValidatedModel):
                metadata: dict[str, Any] = Field(default_factory=dict)

            TestModel()

        error_msg = str(exc_info.value)
        assert "missing required json_schema_extra" in error_msg

    def test_base_class_with_multiple_fields(self):
        """Verify base class handles multiple JSONB fields."""

        class TestModel(JsonbValidatedModel):
            metadata: dict[str, Any] = JsonbField(default_factory=dict)
            config: dict[str, Any] = JsonbField(default_factory=dict)
            settings: dict[str, Any] = JsonbField(default_factory=dict)

        model = TestModel(metadata={"m": 1}, config={"c": 2}, settings={"s": 3})
        assert model.metadata == {"m": 1}
        assert model.config == {"c": 2}
        assert model.settings == {"s": 3}


class TestEntityModelCompliance:
    """Test suite verifying entity models have correct JSONB annotations."""

    def test_model_bridge_state_has_jsonb_annotation(self):
        """Verify ModelBridgeState has correct JSONB annotations."""
        from omninode_bridge.infrastructure.entities.model_bridge_state import (
            ModelBridgeState,
        )

        field_info = ModelBridgeState.model_fields["aggregation_metadata"]
        assert field_info.json_schema_extra == {"db_type": "jsonb"}

    def test_model_metadata_stamp_has_jsonb_annotation(self):
        """Verify ModelMetadataStamp has correct JSONB annotations."""
        from omninode_bridge.infrastructure.entities.model_metadata_stamp import (
            ModelMetadataStamp,
        )

        field_info = ModelMetadataStamp.model_fields["stamp_data"]
        assert field_info.json_schema_extra == {"db_type": "jsonb"}

    def test_model_workflow_execution_has_jsonb_annotation(self):
        """Verify ModelWorkflowExecution has correct JSONB annotations."""
        from omninode_bridge.infrastructure.entities.model_workflow_execution import (
            ModelWorkflowExecution,
        )

        field_info = ModelWorkflowExecution.model_fields["metadata"]
        assert field_info.json_schema_extra == {"db_type": "jsonb"}


class TestErrorMessages:
    """Test suite for validation error messages."""

    def test_error_message_includes_field_name(self):
        """Verify error message includes the problematic field name."""

        with pytest.raises(ValidationError) as exc_info:

            class TestModel(BaseModel):
                problematic_field: dict[str, Any] = Field(default_factory=dict)

                @model_validator(mode="after")
                def _validate(self) -> "TestModel":
                    return validate_jsonb_fields(self)

            TestModel()

        error_msg = str(exc_info.value)
        assert "problematic_field" in error_msg

    def test_error_message_includes_model_name(self):
        """Verify error message includes the model class name."""

        with pytest.raises(ValidationError) as exc_info:

            class ProblematicModel(BaseModel):
                metadata: dict[str, Any] = Field(default_factory=dict)

                @model_validator(mode="after")
                def _validate(self) -> "ProblematicModel":
                    return validate_jsonb_fields(self)

            ProblematicModel()

        error_msg = str(exc_info.value)
        assert "ProblematicModel" in error_msg

    def test_error_message_provides_fix_suggestions(self):
        """Verify error message provides actionable fix suggestions."""

        with pytest.raises(ValidationError) as exc_info:

            class TestModel(BaseModel):
                metadata: dict[str, Any] = Field(default_factory=dict)

                @model_validator(mode="after")
                def _validate(self) -> "TestModel":
                    return validate_jsonb_fields(self)

            TestModel()

        error_msg = str(exc_info.value)
        # Should suggest JsonbField() usage
        assert "JsonbField()" in error_msg or "json_schema_extra" in error_msg


class TestIntegrationScenarios:
    """Integration tests for real-world usage scenarios."""

    def test_entity_model_instantiation(self):
        """Verify entity models can be instantiated correctly."""
        from uuid import uuid4

        from omninode_bridge.infrastructure.entities.model_bridge_state import (
            ModelBridgeState,
        )

        bridge_state = ModelBridgeState(
            bridge_id=uuid4(),
            namespace="test",
            current_fsm_state="PROCESSING",
            aggregation_metadata={
                "total_items": 100,
                "avg_processing_time_ms": 45.2,
            },
        )

        assert bridge_state.aggregation_metadata["total_items"] == 100

    def test_entity_model_serialization(self):
        """Verify entity models serialize correctly with JSONB fields."""
        from datetime import datetime
        from uuid import uuid4

        from omninode_bridge.infrastructure.entities.model_workflow_execution import (
            ModelWorkflowExecution,
        )

        workflow = ModelWorkflowExecution(
            correlation_id=uuid4(),
            workflow_type="test_workflow",
            current_state="PROCESSING",
            namespace="test",
            started_at=datetime.now(UTC),
            metadata={"step": "validation", "retries": 0},
        )

        # Serialize to dict
        data = workflow.model_dump()
        assert data["metadata"]["step"] == "validation"

        # Serialize to JSON
        json_data = workflow.model_dump_json()
        assert "validation" in json_data
