"""Unit tests for metadata validation utilities."""

import pytest

from omninode_bridge.schemas import (
    BRIDGE_STATE_METADATA_SCHEMA,
    NODE_REGISTRATION_METADATA_SCHEMA,
    WORKFLOW_METADATA_SCHEMA,
)
from omninode_bridge.utils.metadata_validator import (
    validate_bridge_state_metadata,
    validate_metadata,
    validate_node_registration_metadata,
    validate_workflow_metadata,
)


class TestValidateMetadata:
    """Test suite for validate_metadata function."""

    def test_valid_metadata_passes(self):
        """Test that valid metadata passes validation."""
        metadata = {"priority": 5, "tags": ["test"]}
        schema = {
            "type": "object",
            "properties": {
                "priority": {"type": "integer"},
                "tags": {"type": "array"},
            },
        }

        # Should not raise
        validate_metadata(metadata, schema)

    def test_invalid_metadata_type_raises_error(self):
        """Test that invalid type raises ValueError."""
        metadata = {"priority": "high"}  # Should be integer
        schema = {
            "type": "object",
            "properties": {
                "priority": {"type": "integer"},
            },
        }

        with pytest.raises(ValueError) as exc_info:
            validate_metadata(metadata, schema)

        assert "Metadata validation failed" in str(exc_info.value)
        assert "priority" in str(exc_info.value)

    def test_empty_metadata_passes(self):
        """Test that empty metadata dictionary passes validation."""
        metadata = {}
        schema = {"type": "object", "properties": {}}

        # Should not raise
        validate_metadata(metadata, schema)

    def test_additional_properties_allowed(self):
        """Test that additional properties are allowed when specified."""
        metadata = {"known_field": "value", "custom_field": "value"}
        schema = {
            "type": "object",
            "properties": {"known_field": {"type": "string"}},
            "additionalProperties": True,
        }

        # Should not raise
        validate_metadata(metadata, schema)


class TestWorkflowMetadataValidation:
    """Test suite for workflow metadata validation."""

    def test_valid_workflow_metadata(self):
        """Test valid workflow metadata passes validation."""
        metadata = {
            "workflow_type": "metadata_stamping",
            "priority": 5,
            "tags": ["api", "production"],
            "user_id": "user-123",
            "steps_completed": 3,
            "custom_data": {"key": "value"},
        }

        # Should not raise
        validate_workflow_metadata(metadata)

    def test_priority_out_of_range_raises_error(self):
        """Test that priority > 10 raises ValueError."""
        metadata = {"priority": 15}  # Max is 10

        with pytest.raises(ValueError) as exc_info:
            validate_workflow_metadata(metadata)

        assert "Metadata validation failed" in str(exc_info.value)
        assert "priority" in str(exc_info.value).lower()

    def test_priority_minimum_validation(self):
        """Test that priority < 1 raises ValueError."""
        metadata = {"priority": 0}  # Min is 1

        with pytest.raises(ValueError) as exc_info:
            validate_workflow_metadata(metadata)

        assert "Metadata validation failed" in str(exc_info.value)

    def test_tags_must_be_array(self):
        """Test that tags must be an array."""
        metadata = {"tags": "not-an-array"}

        with pytest.raises(ValueError) as exc_info:
            validate_workflow_metadata(metadata)

        assert "Metadata validation failed" in str(exc_info.value)
        assert "tags" in str(exc_info.value).lower()

    def test_custom_fields_allowed(self):
        """Test that custom fields are allowed in workflow metadata."""
        metadata = {
            "custom_field_1": "value",
            "custom_field_2": 123,
            "nested": {"data": "value"},
        }

        # Should not raise (additionalProperties: True)
        validate_workflow_metadata(metadata)

    def test_empty_workflow_metadata(self):
        """Test that empty workflow metadata passes validation."""
        metadata = {}

        # Should not raise
        validate_workflow_metadata(metadata)


class TestNodeRegistrationMetadataValidation:
    """Test suite for node registration metadata validation."""

    def test_valid_node_registration_metadata(self):
        """Test valid node registration metadata passes validation."""
        metadata = {
            "environment": "prod",
            "region": "us-west-2",
            "deployment_id": "deploy-123",
            "version": "1.0.0",
            "author": "OmniNode Team",
            "description": "Test service",
            "tags": ["cryptography", "hashing"],
            "resource_limits": {
                "cpu_cores": 4,
                "memory_mb": 4096,
                "max_connections": 100,
            },
        }

        # Should not raise
        validate_node_registration_metadata(metadata)

    def test_invalid_environment_raises_error(self):
        """Test that invalid environment enum raises ValueError."""
        metadata = {"environment": "invalid"}  # Must be dev/staging/prod

        with pytest.raises(ValueError) as exc_info:
            validate_node_registration_metadata(metadata)

        assert "Metadata validation failed" in str(exc_info.value)
        assert "environment" in str(exc_info.value).lower()

    def test_invalid_version_format_raises_error(self):
        """Test that invalid semver format raises ValueError."""
        metadata = {"version": "1.0"}  # Must be x.y.z

        with pytest.raises(ValueError) as exc_info:
            validate_node_registration_metadata(metadata)

        assert "Metadata validation failed" in str(exc_info.value)

    def test_valid_version_formats(self):
        """Test that valid semver formats pass validation."""
        valid_versions = ["1.0.0", "2.5.10", "0.1.0"]

        for version in valid_versions:
            metadata = {"version": version}
            # Should not raise
            validate_node_registration_metadata(metadata)

    def test_resource_limits_validation(self):
        """Test that resource limits are validated correctly."""
        metadata = {
            "resource_limits": {
                "cpu_cores": 2.5,  # Allows float
                "memory_mb": 2048,  # Integer
                "max_connections": 50,
            }
        }

        # Should not raise
        validate_node_registration_metadata(metadata)

    def test_negative_resource_limits_raises_error(self):
        """Test that negative resource limits raise ValueError."""
        metadata = {
            "resource_limits": {
                "cpu_cores": -1,  # Negative not allowed
            }
        }

        with pytest.raises(ValueError) as exc_info:
            validate_node_registration_metadata(metadata)

        assert "Metadata validation failed" in str(exc_info.value)

    def test_custom_fields_allowed_in_node_metadata(self):
        """Test that custom fields are allowed in node registration metadata."""
        metadata = {
            "custom_annotation": "value",
            "deployment_strategy": "blue-green",
        }

        # Should not raise (additionalProperties: True)
        validate_node_registration_metadata(metadata)


class TestBridgeStateMetadataValidation:
    """Test suite for bridge state metadata validation."""

    def test_valid_bridge_state_metadata(self):
        """Test valid bridge state metadata passes validation."""
        metadata = {
            "aggregation_window": "5s",
            "window_size_ms": 5000,
            "batch_size": 100,
            "last_window_items": 25,
            "state_version": 1,
            "aggregation_type": "NAMESPACE_GROUPING",
        }

        # Should not raise
        validate_bridge_state_metadata(metadata)

    def test_valid_aggregation_types(self):
        """Test that all valid aggregation types pass validation."""
        valid_types = [
            "NAMESPACE_GROUPING",
            "TIME_WINDOW",
            "FILE_TYPE_GROUPING",
            "SIZE_BUCKETS",
            "WORKFLOW_GROUPING",
            "CUSTOM",
        ]

        for agg_type in valid_types:
            metadata = {"aggregation_type": agg_type}
            # Should not raise
            validate_bridge_state_metadata(metadata)

    def test_invalid_aggregation_type_raises_error(self):
        """Test that invalid aggregation type raises ValueError."""
        metadata = {"aggregation_type": "INVALID_TYPE"}

        with pytest.raises(ValueError) as exc_info:
            validate_bridge_state_metadata(metadata)

        assert "Metadata validation failed" in str(exc_info.value)
        assert "aggregation_type" in str(exc_info.value).lower()

    def test_negative_window_size_raises_error(self):
        """Test that negative window size raises ValueError."""
        metadata = {"window_size_ms": -1000}

        with pytest.raises(ValueError) as exc_info:
            validate_bridge_state_metadata(metadata)

        assert "Metadata validation failed" in str(exc_info.value)

    def test_batch_size_minimum_validation(self):
        """Test that batch_size < 1 raises ValueError."""
        metadata = {"batch_size": 0}  # Min is 1

        with pytest.raises(ValueError) as exc_info:
            validate_bridge_state_metadata(metadata)

        assert "Metadata validation failed" in str(exc_info.value)

    def test_state_version_validation(self):
        """Test that state_version is validated correctly."""
        metadata = {"state_version": 10}

        # Should not raise
        validate_bridge_state_metadata(metadata)

    def test_negative_state_version_raises_error(self):
        """Test that negative state version raises ValueError."""
        metadata = {"state_version": -1}

        with pytest.raises(ValueError) as exc_info:
            validate_bridge_state_metadata(metadata)

        assert "Metadata validation failed" in str(exc_info.value)

    def test_custom_fields_allowed_in_bridge_metadata(self):
        """Test that custom fields are allowed in bridge state metadata."""
        metadata = {
            "custom_metric": 123,
            "performance_data": {"avg_latency_ms": 50},
        }

        # Should not raise (additionalProperties: True)
        validate_bridge_state_metadata(metadata)


class TestConvenienceFunctions:
    """Test suite for convenience validation functions."""

    def test_validate_workflow_metadata_convenience(self):
        """Test workflow metadata convenience function."""
        metadata = {"priority": 7, "tags": ["test"]}

        # Should not raise
        validate_workflow_metadata(metadata)

    def test_validate_node_registration_metadata_convenience(self):
        """Test node registration metadata convenience function."""
        metadata = {"environment": "dev", "version": "1.0.0"}

        # Should not raise
        validate_node_registration_metadata(metadata)

    def test_validate_bridge_state_metadata_convenience(self):
        """Test bridge state metadata convenience function."""
        metadata = {"window_size_ms": 1000, "batch_size": 50}

        # Should not raise
        validate_bridge_state_metadata(metadata)


class TestSchemaDefinitions:
    """Test suite for schema definitions themselves."""

    def test_workflow_schema_structure(self):
        """Test that WORKFLOW_METADATA_SCHEMA has correct structure."""
        assert "type" in WORKFLOW_METADATA_SCHEMA
        assert WORKFLOW_METADATA_SCHEMA["type"] == "object"
        assert "properties" in WORKFLOW_METADATA_SCHEMA
        assert "additionalProperties" in WORKFLOW_METADATA_SCHEMA
        assert WORKFLOW_METADATA_SCHEMA["additionalProperties"] is True

    def test_node_registration_schema_structure(self):
        """Test that NODE_REGISTRATION_METADATA_SCHEMA has correct structure."""
        assert "type" in NODE_REGISTRATION_METADATA_SCHEMA
        assert NODE_REGISTRATION_METADATA_SCHEMA["type"] == "object"
        assert "properties" in NODE_REGISTRATION_METADATA_SCHEMA
        assert "additionalProperties" in NODE_REGISTRATION_METADATA_SCHEMA

    def test_bridge_state_schema_structure(self):
        """Test that BRIDGE_STATE_METADATA_SCHEMA has correct structure."""
        assert "type" in BRIDGE_STATE_METADATA_SCHEMA
        assert BRIDGE_STATE_METADATA_SCHEMA["type"] == "object"
        assert "properties" in BRIDGE_STATE_METADATA_SCHEMA
        assert "additionalProperties" in BRIDGE_STATE_METADATA_SCHEMA
