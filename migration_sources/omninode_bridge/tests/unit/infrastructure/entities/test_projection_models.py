#!/usr/bin/env python3
"""Unit tests for ModelWorkflowProjection and ModelProjectionWatermark.

Tests cover:
- Model instantiation with all fields
- Model validation
- Default values
- JSON serialization/deserialization
- Field validation
- Model configuration
- Boundary conditions (version, offset limits)

Pure Reducer Refactor - Wave 1, Workstream 1B
"""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from omninode_bridge.infrastructure.entities.model_projection_watermark import (
    ModelProjectionWatermark,
)
from omninode_bridge.infrastructure.entities.model_workflow_projection import (
    ModelWorkflowProjection,
)


class TestModelWorkflowProjection:
    """Test suite for ModelWorkflowProjection."""

    def test_model_instantiation_with_all_fields(self):
        """Test model instantiation with all fields."""
        workflow_key = "wf_test_123"
        version = 5
        tag = "PROCESSING"
        last_action = "StampContent"
        namespace = "production"
        updated_at = datetime.now()
        indices = {"priority": "high", "team": "platform"}
        extras = {"last_error": None, "retry_count": 0}

        projection = ModelWorkflowProjection(
            workflow_key=workflow_key,
            version=version,
            tag=tag,
            last_action=last_action,
            namespace=namespace,
            updated_at=updated_at,
            indices=indices,
            extras=extras,
        )

        assert projection.workflow_key == workflow_key
        assert projection.version == version
        assert projection.tag == tag
        assert projection.last_action == last_action
        assert projection.namespace == namespace
        assert projection.updated_at == updated_at
        assert projection.indices == indices
        assert projection.extras == extras

    def test_model_instantiation_with_required_fields_only(self):
        """Test model instantiation with only required fields."""
        workflow_key = "wf_test_123"
        version = 1
        tag = "PENDING"
        namespace = "test"

        projection = ModelWorkflowProjection(
            workflow_key=workflow_key,
            version=version,
            tag=tag,
            namespace=namespace,
        )

        assert projection.workflow_key == workflow_key
        assert projection.version == version
        assert projection.tag == tag
        assert projection.namespace == namespace
        assert projection.last_action is None  # Optional field
        assert projection.indices is None  # Optional field
        assert projection.extras is None  # Optional field
        assert isinstance(projection.updated_at, datetime)  # Default factory

    def test_model_validation_missing_required_fields(self):
        """Test model validation with missing required fields."""
        # Missing all required fields
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowProjection()

        errors = exc_info.value.errors()
        assert len(errors) >= 4  # workflow_key, version, tag, namespace
        assert any("workflow_key" in str(error) for error in errors)
        assert any("version" in str(error) for error in errors)
        assert any("tag" in str(error) for error in errors)
        assert any("namespace" in str(error) for error in errors)

    def test_model_validation_invalid_workflow_key(self):
        """Test model validation with invalid workflow_key."""
        version = 1
        tag = "PENDING"
        namespace = "test"

        # Empty workflow_key
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowProjection(
                workflow_key="",
                version=version,
                tag=tag,
                namespace=namespace,
            )

        assert "workflow_key" in str(exc_info.value)

        # Too long workflow_key
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowProjection(
                workflow_key="a" * 256,  # 256 characters, max is 255
                version=version,
                tag=tag,
                namespace=namespace,
            )

        assert "workflow_key" in str(exc_info.value)

    def test_model_validation_invalid_version(self):
        """Test model validation with invalid version."""
        workflow_key = "wf_test_123"
        tag = "PENDING"
        namespace = "test"

        # Version must be >= 1
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowProjection(
                workflow_key=workflow_key,
                version=0,
                tag=tag,
                namespace=namespace,
            )

        assert "version" in str(exc_info.value)

        # Negative version
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowProjection(
                workflow_key=workflow_key,
                version=-1,
                tag=tag,
                namespace=namespace,
            )

        assert "version" in str(exc_info.value)

    def test_model_validation_invalid_tag(self):
        """Test model validation with invalid tag."""
        workflow_key = "wf_test_123"
        version = 1
        namespace = "test"

        # Empty tag
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowProjection(
                workflow_key=workflow_key,
                version=version,
                tag="",
                namespace=namespace,
            )

        assert "tag" in str(exc_info.value)

        # Too long tag
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowProjection(
                workflow_key=workflow_key,
                version=version,
                tag="a" * 51,  # 51 characters, max is 50
                namespace=namespace,
            )

        assert "tag" in str(exc_info.value)

    def test_model_validation_invalid_namespace(self):
        """Test model validation with invalid namespace."""
        workflow_key = "wf_test_123"
        version = 1
        tag = "PENDING"

        # Empty namespace
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowProjection(
                workflow_key=workflow_key,
                version=version,
                tag=tag,
                namespace="",
            )

        assert "namespace" in str(exc_info.value)

        # Too long namespace
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowProjection(
                workflow_key=workflow_key,
                version=version,
                tag=tag,
                namespace="a" * 256,  # 256 characters, max is 255
            )

        assert "namespace" in str(exc_info.value)

    def test_model_validation_invalid_last_action(self):
        """Test model validation with invalid last_action."""
        workflow_key = "wf_test_123"
        version = 1
        tag = "PENDING"
        namespace = "test"

        # Too long last_action
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowProjection(
                workflow_key=workflow_key,
                version=version,
                tag=tag,
                namespace=namespace,
                last_action="a" * 101,  # 101 characters, max is 100
            )

        assert "last_action" in str(exc_info.value)

    def test_model_serialization(self):
        """Test model serialization to JSON."""
        workflow_key = "wf_test_123"
        version = 5
        tag = "COMPLETED"
        namespace = "production"

        projection = ModelWorkflowProjection(
            workflow_key=workflow_key,
            version=version,
            tag=tag,
            namespace=namespace,
        )

        # Test model_dump
        data = projection.model_dump()
        assert data["workflow_key"] == workflow_key
        assert data["version"] == version
        assert data["tag"] == tag
        assert data["namespace"] == namespace
        assert data["last_action"] is None
        assert data["indices"] is None
        assert data["extras"] is None
        assert "updated_at" in data

        # Test model_dump_json
        json_str = projection.model_dump_json()
        parsed_data = json.loads(json_str)
        assert parsed_data["workflow_key"] == workflow_key
        assert parsed_data["version"] == version
        assert parsed_data["tag"] == tag
        assert parsed_data["namespace"] == namespace

    def test_model_deserialization(self):
        """Test model deserialization from JSON."""
        data = {
            "workflow_key": "wf_test_123",
            "version": 3,
            "tag": "PROCESSING",
            "last_action": "AggregateData",
            "namespace": "staging",
            "updated_at": datetime.now().isoformat(),
            "indices": {"priority": "low"},
            "extras": {"retry_count": 2},
        }

        # Test model_validate
        projection = ModelWorkflowProjection.model_validate(data)
        assert projection.workflow_key == data["workflow_key"]
        assert projection.version == data["version"]
        assert projection.tag == data["tag"]
        assert projection.last_action == data["last_action"]
        assert projection.namespace == data["namespace"]
        assert projection.indices == data["indices"]
        assert projection.extras == data["extras"]

    def test_model_roundtrip(self):
        """Test model serialization and deserialization roundtrip."""
        original = ModelWorkflowProjection(
            workflow_key="wf_test_123",
            version=7,
            tag="FAILED",
            last_action="RetryWorkflow",
            namespace="production",
            indices={"priority": "critical"},
            extras={"error": "Timeout", "retry_count": 3},
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = ModelWorkflowProjection.model_validate(data)

        # Verify all fields match
        assert restored.workflow_key == original.workflow_key
        assert restored.version == original.version
        assert restored.tag == original.tag
        assert restored.last_action == original.last_action
        assert restored.namespace == original.namespace
        assert restored.indices == original.indices
        assert restored.extras == original.extras

    def test_model_jsonb_validation(self):
        """Test JSONB field validation."""
        projection = ModelWorkflowProjection(
            workflow_key="wf_test_123",
            version=1,
            tag="PENDING",
            namespace="test",
            indices={"custom_index": "value"},
            extras={"metadata": "data"},
        )

        # The _validate_jsonb_fields method should be called during initialization
        # If it's working correctly, no exception should be raised
        assert projection.indices == {"custom_index": "value"}
        assert projection.extras == {"metadata": "data"}

    def test_model_version_boundary_conditions(self):
        """Test version field boundary conditions."""
        workflow_key = "wf_test_123"
        tag = "PENDING"
        namespace = "test"

        # Minimum valid version (1)
        projection = ModelWorkflowProjection(
            workflow_key=workflow_key,
            version=1,
            tag=tag,
            namespace=namespace,
        )
        assert projection.version == 1

        # Large version number (simulating many updates)
        large_version = 2**63 - 1  # Max BIGINT
        projection = ModelWorkflowProjection(
            workflow_key=workflow_key,
            version=large_version,
            tag=tag,
            namespace=namespace,
        )
        assert projection.version == large_version


class TestModelProjectionWatermark:
    """Test suite for ModelProjectionWatermark."""

    def test_model_instantiation_with_all_fields(self):
        """Test model instantiation with all fields."""
        partition_id = "kafka-partition-0"
        offset = 12345
        updated_at = datetime.now()

        watermark = ModelProjectionWatermark(
            partition_id=partition_id,
            offset=offset,
            updated_at=updated_at,
        )

        assert watermark.partition_id == partition_id
        assert watermark.offset == offset
        assert watermark.updated_at == updated_at

    def test_model_instantiation_with_required_fields_only(self):
        """Test model instantiation with only required fields."""
        partition_id = "kafka-partition-0"

        watermark = ModelProjectionWatermark(partition_id=partition_id)

        assert watermark.partition_id == partition_id
        assert watermark.offset == 0  # Default value
        assert isinstance(watermark.updated_at, datetime)  # Default factory

    def test_model_validation_missing_required_fields(self):
        """Test model validation with missing required fields."""
        # Missing partition_id
        with pytest.raises(ValidationError) as exc_info:
            ModelProjectionWatermark()

        errors = exc_info.value.errors()
        assert any("partition_id" in str(error) for error in errors)

    def test_model_validation_invalid_partition_id(self):
        """Test model validation with invalid partition_id."""
        # Empty partition_id
        with pytest.raises(ValidationError) as exc_info:
            ModelProjectionWatermark(partition_id="")

        assert "partition_id" in str(exc_info.value)

        # Too long partition_id
        with pytest.raises(ValidationError) as exc_info:
            ModelProjectionWatermark(
                partition_id="a" * 256
            )  # 256 characters, max is 255

        assert "partition_id" in str(exc_info.value)

    def test_model_validation_invalid_offset(self):
        """Test model validation with invalid offset."""
        partition_id = "kafka-partition-0"

        # Negative offset
        with pytest.raises(ValidationError) as exc_info:
            ModelProjectionWatermark(partition_id=partition_id, offset=-1)

        assert "offset" in str(exc_info.value)

    def test_model_serialization(self):
        """Test model serialization to JSON."""
        partition_id = "kafka-partition-0"
        offset = 12345

        watermark = ModelProjectionWatermark(
            partition_id=partition_id,
            offset=offset,
        )

        # Test model_dump
        data = watermark.model_dump()
        assert data["partition_id"] == partition_id
        assert data["offset"] == offset
        assert "updated_at" in data

        # Test model_dump_json
        json_str = watermark.model_dump_json()
        parsed_data = json.loads(json_str)
        assert parsed_data["partition_id"] == partition_id
        assert parsed_data["offset"] == offset

    def test_model_deserialization(self):
        """Test model deserialization from JSON."""
        data = {
            "partition_id": "kafka-partition-1",
            "offset": 54321,
            "updated_at": datetime.now().isoformat(),
        }

        # Test model_validate
        watermark = ModelProjectionWatermark.model_validate(data)
        assert watermark.partition_id == data["partition_id"]
        assert watermark.offset == data["offset"]

    def test_model_roundtrip(self):
        """Test model serialization and deserialization roundtrip."""
        original = ModelProjectionWatermark(
            partition_id="kafka-partition-2",
            offset=99999,
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = ModelProjectionWatermark.model_validate(data)

        # Verify all fields match
        assert restored.partition_id == original.partition_id
        assert restored.offset == original.offset

    def test_model_offset_boundary_conditions(self):
        """Test offset field boundary conditions (0 to BIGINT max)."""
        partition_id = "kafka-partition-0"

        # Minimum valid offset (0)
        watermark = ModelProjectionWatermark(
            partition_id=partition_id,
            offset=0,
        )
        assert watermark.offset == 0

        # Large offset (simulating high-throughput partition)
        large_offset = 2**63 - 1  # Max BIGINT
        watermark = ModelProjectionWatermark(
            partition_id=partition_id,
            offset=large_offset,
        )
        assert watermark.offset == large_offset

    def test_watermark_lag_calculation(self):
        """Test watermark lag detection pattern."""
        partition_id = "kafka-partition-0"
        offset = 1000

        # Create watermark with explicit timestamp
        old_timestamp = datetime(2025, 10, 21, 12, 0, 0)
        watermark = ModelProjectionWatermark(
            partition_id=partition_id,
            offset=offset,
            updated_at=old_timestamp,
        )

        # Calculate lag (in production, would compare to datetime.now())
        current_time = datetime(2025, 10, 21, 12, 0, 1)  # 1 second later
        lag_ms = (current_time - watermark.updated_at).total_seconds() * 1000

        assert lag_ms == 1000  # 1 second = 1000ms
        assert watermark.offset == offset

    def test_multiple_partitions(self):
        """Test multiple partition watermarks (typical usage pattern)."""
        # Simulate 3 Kafka partitions
        watermarks = [
            ModelProjectionWatermark(
                partition_id=f"kafka-partition-{i}", offset=i * 1000
            )
            for i in range(3)
        ]

        assert len(watermarks) == 3
        assert watermarks[0].partition_id == "kafka-partition-0"
        assert watermarks[0].offset == 0
        assert watermarks[1].partition_id == "kafka-partition-1"
        assert watermarks[1].offset == 1000
        assert watermarks[2].partition_id == "kafka-partition-2"
        assert watermarks[2].offset == 2000
