"""Integration tests for metadata validation in entity models."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omninode_bridge.models.entities.model_bridge_state import ModelBridgeState
from omninode_bridge.models.entities.model_workflow_execution import (
    ModelWorkflowExecution,
)
from omninode_bridge.models.node_registration import ModelNodeRegistration


class TestModelWorkflowExecutionMetadataValidation:
    """Test metadata validation in ModelWorkflowExecution."""

    def test_create_with_valid_metadata(self):
        """Test creating ModelWorkflowExecution with valid metadata."""
        execution = ModelWorkflowExecution(
            correlation_id=uuid4(),
            workflow_type="metadata_stamping",
            current_state="PENDING",
            namespace="test_app",
            started_at=datetime.now(UTC),
            metadata={
                "priority": 8,
                "tags": ["api", "production"],
                "user_id": "user-123",
                "steps_completed": 0,
            },
        )

        assert execution.metadata["priority"] == 8
        assert "api" in execution.metadata["tags"]

    def test_create_with_invalid_priority_raises_error(self):
        """Test that invalid priority in metadata raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecution(
                correlation_id=uuid4(),
                workflow_type="metadata_stamping",
                current_state="PENDING",
                namespace="test_app",
                started_at=datetime.now(UTC),
                metadata={"priority": 15},  # Max is 10
            )

        # Check that validation error mentions metadata
        error_msg = str(exc_info.value)
        assert "metadata" in error_msg.lower()

    def test_create_with_invalid_tags_type_raises_error(self):
        """Test that non-array tags raise ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelWorkflowExecution(
                correlation_id=uuid4(),
                workflow_type="metadata_stamping",
                current_state="PENDING",
                namespace="test_app",
                started_at=datetime.now(UTC),
                metadata={"tags": "not-an-array"},
            )

        error_msg = str(exc_info.value)
        assert "metadata" in error_msg.lower()

    def test_create_with_empty_metadata(self):
        """Test creating ModelWorkflowExecution with empty metadata."""
        execution = ModelWorkflowExecution(
            correlation_id=uuid4(),
            workflow_type="metadata_stamping",
            current_state="PENDING",
            namespace="test_app",
            started_at=datetime.now(UTC),
            metadata={},
        )

        assert execution.metadata == {}

    def test_create_with_custom_metadata_fields(self):
        """Test that custom metadata fields are allowed."""
        execution = ModelWorkflowExecution(
            correlation_id=uuid4(),
            workflow_type="metadata_stamping",
            current_state="PENDING",
            namespace="test_app",
            started_at=datetime.now(UTC),
            metadata={
                "custom_field": "value",
                "nested": {"data": "value"},
            },
        )

        assert execution.metadata["custom_field"] == "value"
        assert execution.metadata["nested"]["data"] == "value"


class TestModelBridgeStateMetadataValidation:
    """Test metadata validation in ModelBridgeState."""

    def test_create_with_valid_aggregation_metadata(self):
        """Test creating ModelBridgeState with valid aggregation_metadata."""
        bridge_state = ModelBridgeState(
            bridge_id=uuid4(),
            namespace="test_app",
            current_fsm_state="ACTIVE",
            aggregation_metadata={
                "window_size_ms": 5000,
                "batch_size": 100,
                "aggregation_type": "NAMESPACE_GROUPING",
                "state_version": 1,
            },
        )

        assert bridge_state.aggregation_metadata["window_size_ms"] == 5000
        assert bridge_state.aggregation_metadata["batch_size"] == 100

    def test_create_with_invalid_aggregation_type_raises_error(self):
        """Test that invalid aggregation_type raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeState(
                bridge_id=uuid4(),
                namespace="test_app",
                current_fsm_state="ACTIVE",
                aggregation_metadata={
                    "aggregation_type": "INVALID_TYPE",
                },
            )

        error_msg = str(exc_info.value)
        assert "metadata" in error_msg.lower() or "aggregation" in error_msg.lower()

    def test_create_with_negative_window_size_raises_error(self):
        """Test that negative window_size_ms raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeState(
                bridge_id=uuid4(),
                namespace="test_app",
                current_fsm_state="ACTIVE",
                aggregation_metadata={
                    "window_size_ms": -1000,
                },
            )

        error_msg = str(exc_info.value)
        assert "metadata" in error_msg.lower() or "aggregation" in error_msg.lower()

    def test_create_with_invalid_batch_size_raises_error(self):
        """Test that batch_size < 1 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelBridgeState(
                bridge_id=uuid4(),
                namespace="test_app",
                current_fsm_state="ACTIVE",
                aggregation_metadata={
                    "batch_size": 0,  # Min is 1
                },
            )

        error_msg = str(exc_info.value)
        assert "metadata" in error_msg.lower() or "aggregation" in error_msg.lower()

    def test_create_with_empty_aggregation_metadata(self):
        """Test creating ModelBridgeState with empty aggregation_metadata."""
        bridge_state = ModelBridgeState(
            bridge_id=uuid4(),
            namespace="test_app",
            current_fsm_state="ACTIVE",
            aggregation_metadata={},
        )

        assert bridge_state.aggregation_metadata == {}

    def test_create_with_custom_aggregation_metadata_fields(self):
        """Test that custom aggregation metadata fields are allowed."""
        bridge_state = ModelBridgeState(
            bridge_id=uuid4(),
            namespace="test_app",
            current_fsm_state="ACTIVE",
            aggregation_metadata={
                "custom_metric": 123,
                "performance_data": {"avg_latency_ms": 50},
            },
        )

        assert bridge_state.aggregation_metadata["custom_metric"] == 123


class TestModelNodeRegistrationMetadataValidation:
    """Test metadata validation in ModelNodeRegistration."""

    def test_create_with_valid_metadata(self):
        """Test creating ModelNodeRegistration with valid metadata."""
        registration = ModelNodeRegistration(
            node_id="test-node-1",
            node_type="effect",
            metadata={
                "environment": "prod",
                "region": "us-west-2",
                "version": "1.0.0",
                "author": "OmniNode Team",
                "tags": ["cryptography", "hashing"],
            },
        )

        assert registration.metadata["environment"] == "prod"
        assert registration.metadata["version"] == "1.0.0"

    def test_create_with_invalid_environment_raises_error(self):
        """Test that invalid environment enum raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id="test-node-1",
                node_type="effect",
                metadata={
                    "environment": "invalid",  # Must be dev/staging/prod
                },
            )

        error_msg = str(exc_info.value)
        assert "metadata" in error_msg.lower()

    def test_create_with_invalid_version_format_raises_error(self):
        """Test that invalid semver format raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id="test-node-1",
                node_type="effect",
                metadata={
                    "version": "1.0",  # Must be x.y.z
                },
            )

        error_msg = str(exc_info.value)
        assert "metadata" in error_msg.lower()

    def test_create_with_valid_resource_limits(self):
        """Test creating with valid resource limits."""
        registration = ModelNodeRegistration(
            node_id="test-node-1",
            node_type="effect",
            metadata={
                "resource_limits": {
                    "cpu_cores": 4,
                    "memory_mb": 4096,
                    "max_connections": 100,
                }
            },
        )

        limits = registration.metadata["resource_limits"]
        assert limits["cpu_cores"] == 4
        assert limits["memory_mb"] == 4096

    def test_create_with_negative_resource_limits_raises_error(self):
        """Test that negative resource limits raise ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id="test-node-1",
                node_type="effect",
                metadata={
                    "resource_limits": {
                        "cpu_cores": -1,  # Negative not allowed
                    }
                },
            )

        error_msg = str(exc_info.value)
        assert "metadata" in error_msg.lower()

    def test_create_with_empty_metadata(self):
        """Test creating ModelNodeRegistration with empty metadata."""
        registration = ModelNodeRegistration(
            node_id="test-node-1",
            node_type="effect",
            metadata={},
        )

        assert registration.metadata == {}

    def test_create_with_custom_metadata_fields(self):
        """Test that custom metadata fields are allowed."""
        registration = ModelNodeRegistration(
            node_id="test-node-1",
            node_type="effect",
            metadata={
                "custom_annotation": "value",
                "deployment_strategy": "blue-green",
            },
        )

        assert registration.metadata["custom_annotation"] == "value"


class TestMetadataValidationEdgeCases:
    """Test edge cases and error handling in metadata validation."""

    def test_workflow_metadata_with_all_valid_fields(self):
        """Test workflow metadata with all valid fields populated."""
        execution = ModelWorkflowExecution(
            correlation_id=uuid4(),
            workflow_type="metadata_stamping",
            current_state="COMPLETED",
            namespace="production_app",
            started_at=datetime.now(UTC),
            metadata={
                "workflow_type": "metadata_stamping",
                "priority": 10,  # Max value
                "tags": ["tag1", "tag2", "tag3"],
                "user_id": "admin-user",
                "steps_completed": 5,
                "custom_data": {
                    "nested": {
                        "deeply": {
                            "nested": "value",
                        }
                    }
                },
            },
        )

        assert execution.metadata["priority"] == 10
        assert len(execution.metadata["tags"]) == 3

    def test_bridge_metadata_with_all_aggregation_types(self):
        """Test bridge state with all valid aggregation types."""
        aggregation_types = [
            "NAMESPACE_GROUPING",
            "TIME_WINDOW",
            "FILE_TYPE_GROUPING",
            "SIZE_BUCKETS",
            "WORKFLOW_GROUPING",
            "CUSTOM",
        ]

        for agg_type in aggregation_types:
            bridge_state = ModelBridgeState(
                bridge_id=uuid4(),
                namespace="test_app",
                current_fsm_state="ACTIVE",
                aggregation_metadata={
                    "aggregation_type": agg_type,
                },
            )

            assert bridge_state.aggregation_metadata["aggregation_type"] == agg_type

    def test_node_metadata_with_all_valid_environments(self):
        """Test node registration with all valid environments."""
        environments = ["dev", "staging", "prod"]

        for env in environments:
            registration = ModelNodeRegistration(
                node_id=f"test-node-{env}",
                node_type="effect",
                metadata={"environment": env},
            )

            assert registration.metadata["environment"] == env
