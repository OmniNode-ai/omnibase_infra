# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelNodeIntrospectionEvent.

Tests validate:
- Required field instantiation
- Optional field handling
- Literal node_type validation
- JSON serialization/deserialization roundtrip
- Timestamp auto-generation
- Frozen model immutability
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)

# Fixed test timestamp for deterministic testing (time injection pattern)
TEST_TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


class TestModelNodeIntrospectionEventBasicInstantiation:
    """Tests for basic model instantiation."""

    def test_valid_instantiation_required_fields_only(self) -> None:
        """Test creating event with only required fields."""
        test_node_id = uuid4()
        correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=correlation_id,
            timestamp=TEST_TIMESTAMP,
        )
        assert event.node_id == test_node_id
        assert event.node_type == "effect"
        assert event.node_version == "1.0.0"  # Default value
        assert event.capabilities == ModelNodeCapabilities()
        assert event.endpoints == {}
        assert event.node_role is None
        assert event.metadata == ModelNodeMetadata()
        assert isinstance(event.correlation_id, UUID)
        assert event.correlation_id == correlation_id
        assert event.network_id is None
        assert event.deployment_id is None
        assert event.epoch is None

    def test_valid_instantiation_all_fields(self) -> None:
        """Test creating event with all fields populated."""
        test_node_id = uuid4()
        correlation_id = uuid4()
        timestamp = datetime.now(UTC)
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="compute",
            node_version="2.1.0",
            capabilities={"processing": True, "batch_size": 100},
            endpoints={
                "health": "http://localhost:8080/health",
                "metrics": "http://localhost:8080/metrics",
            },
            node_role="processor",
            metadata={"version": "1.0.0", "environment": "production"},
            correlation_id=correlation_id,
            network_id="network-001",
            deployment_id="deploy-001",
            epoch=1,
            timestamp=timestamp,
        )
        assert event.node_id == test_node_id
        assert event.node_type == "compute"
        assert event.node_version == "2.1.0"
        assert event.capabilities == ModelNodeCapabilities(
            processing=True, batch_size=100
        )
        assert event.endpoints == {
            "health": "http://localhost:8080/health",
            "metrics": "http://localhost:8080/metrics",
        }
        assert event.node_role == "processor"
        assert event.metadata == ModelNodeMetadata(
            version="1.0.0", environment="production"
        )
        assert event.correlation_id == correlation_id
        assert event.network_id == "network-001"
        assert event.deployment_id == "deploy-001"
        assert event.epoch == 1
        assert event.timestamp == timestamp


class TestModelNodeIntrospectionEventNodeVersion:
    """Tests for node_version field."""

    def test_node_version_default_value(self) -> None:
        """Test that node_version defaults to '1.0.0'."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.node_version == "1.0.0"

    def test_node_version_explicit_value(self) -> None:
        """Test that node_version can be set explicitly."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="2.3.4",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.node_version == "2.3.4"

    def test_node_version_with_prerelease(self) -> None:
        """Test that node_version accepts prerelease versions."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="1.0.0-beta.2",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.node_version == "1.0.0-beta.2"

    def test_node_version_with_build_metadata(self) -> None:
        """Test that node_version accepts build metadata."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="1.0.0+build.456",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.node_version == "1.0.0+build.456"

    def test_node_version_serialization_roundtrip(self) -> None:
        """Test that node_version is preserved in JSON serialization."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="3.2.1",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        json_str = event.model_dump_json()
        restored = ModelNodeIntrospectionEvent.model_validate_json(json_str)
        assert restored.node_version == "3.2.1"

    def test_node_version_in_model_dump(self) -> None:
        """Test that node_version appears in model_dump output."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="4.5.6",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        data = event.model_dump()
        assert "node_version" in data
        assert data["node_version"] == "4.5.6"


class TestModelNodeIntrospectionEventNodeTypeValidation:
    """Tests for node_type Literal validation."""

    def test_valid_node_type_effect(self) -> None:
        """Test that 'effect' is a valid node_type."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.node_type == "effect"

    def test_valid_node_type_compute(self) -> None:
        """Test that 'compute' is a valid node_type."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="compute",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.node_type == "compute"

    def test_valid_node_type_reducer(self) -> None:
        """Test that 'reducer' is a valid node_type."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="reducer",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.node_type == "reducer"

    def test_valid_node_type_orchestrator(self) -> None:
        """Test that 'orchestrator' is a valid node_type."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="orchestrator",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.node_type == "orchestrator"

    def test_invalid_node_type_raises_validation_error(self) -> None:
        """Test that invalid node_type raises ValidationError."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="invalid_type",  # type: ignore[arg-type]
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )
        assert "node_type" in str(exc_info.value)

    def test_invalid_node_type_empty_string(self) -> None:
        """Test that empty string node_type raises ValidationError."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError):
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="",  # type: ignore[arg-type]
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )

    def test_invalid_node_type_none(self) -> None:
        """Test that None node_type raises ValidationError."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError):
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type=None,  # type: ignore[arg-type]
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )


class TestModelNodeIntrospectionEventSerialization:
    """Tests for JSON serialization and deserialization."""

    def test_json_serialization_roundtrip_minimal(self) -> None:
        """Test JSON serialization and deserialization with minimal fields."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="reducer",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        json_str = event.model_dump_json()
        restored = ModelNodeIntrospectionEvent.model_validate_json(json_str)
        assert restored.node_id == event.node_id
        assert restored.node_type == event.node_type
        assert restored.capabilities == event.capabilities
        assert restored.endpoints == event.endpoints

    def test_json_serialization_roundtrip_full(self) -> None:
        """Test JSON serialization and deserialization with all fields."""
        test_node_id = uuid4()
        correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="orchestrator",
            capabilities={"routing": True},
            endpoints={"api": "http://localhost:8080/api"},
            node_role="coordinator",
            metadata={"cluster": "primary"},
            correlation_id=correlation_id,
            network_id="network-001",
            deployment_id="deploy-001",
            epoch=5,
            timestamp=TEST_TIMESTAMP,
        )
        json_str = event.model_dump_json()
        restored = ModelNodeIntrospectionEvent.model_validate_json(json_str)

        assert restored.node_id == event.node_id
        assert restored.node_type == event.node_type
        assert restored.capabilities == event.capabilities
        assert restored.endpoints == event.endpoints
        assert restored.node_role == event.node_role
        assert restored.metadata == event.metadata
        assert restored.correlation_id == event.correlation_id
        assert restored.network_id == event.network_id
        assert restored.deployment_id == event.deployment_id
        assert restored.epoch == event.epoch
        # Timestamps should match within reasonable precision
        assert abs((restored.timestamp - event.timestamp).total_seconds()) < 1

    def test_model_dump_dict(self) -> None:
        """Test model_dump produces correct dict structure."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            capabilities=ModelNodeCapabilities(database=True),
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        data = event.model_dump()
        assert isinstance(data, dict)
        assert data["node_id"] == test_node_id
        assert data["node_type"] == "effect"
        # capabilities is now a nested dict from ModelNodeCapabilities
        assert data["capabilities"]["database"] is True
        # Check other default values in the capabilities dict
        assert data["capabilities"]["postgres"] is False

    def test_model_dump_mode_json(self) -> None:
        """Test model_dump with mode='json' for JSON-compatible output."""
        test_node_id = uuid4()
        correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="compute",
            correlation_id=correlation_id,
            timestamp=TEST_TIMESTAMP,
        )
        data = event.model_dump(mode="json")
        # UUID should be serialized as string in JSON mode
        assert data["node_id"] == str(test_node_id)
        assert data["correlation_id"] == str(correlation_id)
        # Datetime should be serialized as ISO string
        assert isinstance(data["timestamp"], str)


class TestModelNodeIntrospectionEventTimestamp:
    """Tests for timestamp field (required, injected by caller)."""

    def test_timestamp_is_required(self) -> None:
        """Test that timestamp is required (time injection pattern).

        Per ONEX time injection pattern, timestamps must be explicitly
        injected by the caller for testability and deterministic behavior.
        """
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="orchestrator",
                correlation_id=uuid4(),
                # timestamp intentionally omitted
            )
        assert "timestamp" in str(exc_info.value)

    def test_timestamp_explicit_value(self) -> None:
        """Test that explicit timestamp is preserved."""
        test_node_id = uuid4()
        explicit_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            timestamp=explicit_time,
            correlation_id=uuid4(),
        )
        assert event.timestamp == explicit_time

    def test_timestamp_is_datetime(self) -> None:
        """Test that timestamp is a datetime object."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="compute",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert isinstance(event.timestamp, datetime)

    def test_naive_timestamp_rejected(self) -> None:
        """Test that naive datetime (without tzinfo) is rejected.

        Timezone-aware timestamps are required to prevent ambiguity in
        distributed systems where events may be processed across time zones.
        """
        from datetime import datetime as dt

        test_node_id = uuid4()
        naive_timestamp = dt(2025, 1, 1, 12, 0, 0)  # No tzinfo

        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                correlation_id=uuid4(),
                timestamp=naive_timestamp,
            )

        error_str = str(exc_info.value).lower()
        assert "timezone-aware" in error_str or "tzinfo" in error_str


class TestModelNodeIntrospectionEventImmutability:
    """Tests for frozen model immutability."""

    def test_frozen_model_cannot_modify_node_id(self) -> None:
        """Test that node_id cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        with pytest.raises(ValidationError):
            event.node_id = uuid4()  # type: ignore[misc]

    def test_frozen_model_cannot_modify_node_type(self) -> None:
        """Test that node_type cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        with pytest.raises(ValidationError):
            event.node_type = "compute"  # type: ignore[misc]

    def test_frozen_model_cannot_modify_node_version(self) -> None:
        """Test that node_version cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="1.0.0",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        with pytest.raises(ValidationError):
            event.node_version = "2.0.0"  # type: ignore[misc]

    def test_frozen_model_cannot_modify_capabilities(self) -> None:
        """Test that capabilities dict reference cannot be reassigned."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            capabilities={"original": True},
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        with pytest.raises(ValidationError):
            # Intentional: assigning dict to ModelNodeCapabilities to test frozen rejection
            event.capabilities = {"modified": True}  # type: ignore[misc, assignment]

    def test_frozen_model_cannot_modify_correlation_id(self) -> None:
        """Test that correlation_id cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        with pytest.raises(ValidationError):
            event.correlation_id = uuid4()  # type: ignore[misc]

    def test_frozen_model_cannot_modify_timestamp(self) -> None:
        """Test that timestamp cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        with pytest.raises(ValidationError):
            event.timestamp = datetime.now(UTC)  # type: ignore[misc]


class TestModelNodeIntrospectionEventEdgeCases:
    """Tests for edge cases and special values."""

    def test_invalid_node_id_empty_string_raises_error(self) -> None:
        """Test that empty string is not allowed for node_id (UUID type)."""
        with pytest.raises(ValidationError):
            ModelNodeIntrospectionEvent(
                node_id="",  # type: ignore[arg-type]
                node_type="effect",
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )

    def test_complex_capabilities_dict(self) -> None:
        """Test capabilities with complex nested values."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="compute",
            capabilities={
                "processing": True,
                "max_batch": 1000,
                "supported_types": ["json", "xml", "csv"],
                "config": {"timeout": 30, "retries": 3},
            },
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.capabilities.processing is True
        assert event.capabilities.max_batch == 1000
        assert event.capabilities.supported_types == ["json", "xml", "csv"]
        assert event.capabilities.config["timeout"] == 30

    def test_unicode_in_fields(self) -> None:
        """Test Unicode characters in string fields."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_role="处理器",
            metadata={"description": "Узел обработки"},
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.node_id == test_node_id
        assert event.node_role == "处理器"
        assert event.metadata.description == "Узел обработки"

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden by model config."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
                extra_field="not_allowed",  # type: ignore[call-arg]
            )
        assert "extra_field" in str(exc_info.value)

    def test_negative_epoch_raises_validation_error(self) -> None:
        """Test that negative epoch values raise ValidationError.

        Epoch represents a registration ordering counter (monotonically increasing),
        so negative values are semantically invalid.
        """
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                epoch=-1,
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )
        assert "epoch" in str(exc_info.value)

    def test_zero_epoch_allowed(self) -> None:
        """Test that zero epoch is allowed (valid for first registration)."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            epoch=0,
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.epoch == 0

    def test_positive_epoch_allowed(self) -> None:
        """Test that positive epoch values are allowed."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            epoch=42,
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.epoch == 42

    def test_large_epoch_allowed(self) -> None:
        """Test that large epoch values are allowed."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            epoch=2**31,
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.epoch == 2**31


class TestModelNodeIntrospectionEventFromAttributes:
    """Tests for from_attributes configuration (ORM mode)."""

    def test_from_dict_like_object(self) -> None:
        """Test creating model from dict-like object."""
        test_node_id = uuid4()
        test_correlation_id = uuid4()

        class DictLike:
            def __init__(self, node_id: UUID, correlation_id: UUID) -> None:
                self.node_id = node_id
                self.node_type = "compute"
                self.node_version = "1.0.0"
                self.capabilities: dict[str, bool] = {}
                self.endpoints: dict[str, str] = {}
                self.node_role = None
                self.metadata: dict[str, str] = {}
                self.correlation_id = correlation_id
                self.network_id = None
                self.deployment_id = None
                self.epoch = None
                self.timestamp = datetime.now(UTC)

        obj = DictLike(test_node_id, test_correlation_id)
        event = ModelNodeIntrospectionEvent.model_validate(obj)
        assert event.node_id == test_node_id
        assert event.node_type == "compute"
        assert event.node_version == "1.0.0"
        assert event.correlation_id == test_correlation_id


class TestModelNodeIntrospectionEventEquality:
    """Tests for model equality comparison."""

    def test_equal_events_are_equal(self) -> None:
        """Test that two events with same values are equal."""
        test_node_id = uuid4()
        correlation_id = uuid4()
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        event1 = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            timestamp=timestamp,
            correlation_id=correlation_id,
        )
        event2 = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            timestamp=timestamp,
            correlation_id=correlation_id,
        )
        assert event1 == event2

    def test_different_node_id_not_equal(self) -> None:
        """Test that events with different node_id are not equal."""
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        correlation_id = uuid4()
        event1 = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            timestamp=timestamp,
            correlation_id=correlation_id,
        )
        event2 = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            timestamp=timestamp,
            correlation_id=correlation_id,
        )
        assert event1 != event2

    def test_different_node_type_not_equal(self) -> None:
        """Test that events with different node_type are not equal."""
        test_node_id = uuid4()
        correlation_id = uuid4()
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        event1 = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            timestamp=timestamp,
            correlation_id=correlation_id,
        )
        event2 = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="compute",
            timestamp=timestamp,
            correlation_id=correlation_id,
        )
        assert event1 != event2

    def test_not_equal_to_non_model(self) -> None:
        """Test that event is not equal to non-model objects."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event != "not a model"
        assert event != 42
        assert event != None


class TestModelNodeIntrospectionEventHashing:
    """Tests for model hashing behavior.

    Note: Even though this model is frozen (immutable), it contains dict fields
    (capabilities, endpoints, metadata) which are unhashable in Python.
    Pydantic's frozen config prevents field reassignment but doesn't make the
    model hashable if it contains unhashable types.
    """

    def test_frozen_model_with_dict_fields_not_hashable(self) -> None:
        """Test that frozen model with dict fields is not hashable.

        Even frozen Pydantic models are not hashable if they contain
        unhashable types like dict. This is because Pydantic uses the
        field values for hashing, and dict is inherently unhashable.
        """
        test_node_id = uuid4()
        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            timestamp=timestamp,
            correlation_id=uuid4(),
        )
        # Model has dict fields (capabilities, endpoints, metadata) so it's not hashable
        with pytest.raises(TypeError):
            hash(event)


class TestModelNodeIntrospectionEventStringRepresentation:
    """Tests for model string representation."""

    def test_str_contains_model_name(self) -> None:
        """Test that __str__ contains the model name."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        str_repr = str(event)
        # Pydantic models include field values in string representation
        assert "node_id" in str_repr or str(test_node_id) in str_repr

    def test_repr_is_valid(self) -> None:
        """Test that __repr__ produces valid representation."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        repr_str = repr(event)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_str_and_repr_contain_node_type(self) -> None:
        """Test that string representations contain node_type."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="orchestrator",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        str_repr = str(event)
        repr_str = repr(event)
        # At least one should contain the node_type value
        assert "orchestrator" in str_repr or "orchestrator" in repr_str


class TestModelNodeIntrospectionEventCopying:
    """Tests for model copying behavior."""

    def test_model_copy_creates_new_instance(self) -> None:
        """Test that model_copy creates a new instance."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        copied = event.model_copy()
        assert copied is not event
        assert copied == event

    def test_model_copy_with_update(self) -> None:
        """Test that model_copy can update fields."""
        test_node_id = uuid4()
        new_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        copied = event.model_copy(update={"node_id": new_node_id})
        assert copied.node_id == new_node_id
        assert copied.node_type == event.node_type
        # Original is unchanged
        assert event.node_id == test_node_id

    def test_model_copy_deep(self) -> None:
        """Test that deep copy creates independent nested objects."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            capabilities={"key": "value"},
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        copied = event.model_copy(deep=True)
        # Both should have same values
        assert copied.capabilities == event.capabilities
        # But dict should be independent (deep copy)
        # Note: For frozen models, we can't modify in place, but the dict
        # reference should still be different
        assert copied.capabilities is not event.capabilities


class TestModelNodeIntrospectionEventEndpointUrlValidation:
    """Tests for endpoints dict URL validation."""

    def test_valid_http_endpoints(self) -> None:
        """Test that valid HTTP URLs in endpoints are accepted."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            endpoints={
                "health": "http://localhost:8080/health",
                "metrics": "http://localhost:8080/metrics",
            },
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.endpoints["health"] == "http://localhost:8080/health"
        assert event.endpoints["metrics"] == "http://localhost:8080/metrics"

    def test_valid_https_endpoints(self) -> None:
        """Test that valid HTTPS URLs in endpoints are accepted."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            endpoints={
                "api": "https://api.example.com:443/v1",
                "health": "https://api.example.com/health",
            },
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.endpoints["api"] == "https://api.example.com:443/v1"
        assert event.endpoints["health"] == "https://api.example.com/health"

    def test_valid_urls_with_path_and_query(self) -> None:
        """Test that URLs with paths and query parameters are accepted."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            endpoints={
                "health": "http://localhost:8080/api/v1/health?timeout=30&verbose=true",
            },
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert (
            event.endpoints["health"]
            == "http://localhost:8080/api/v1/health?timeout=30&verbose=true"
        )

    def test_empty_endpoints_dict_allowed(self) -> None:
        """Test that empty endpoints dict is allowed."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            endpoints={},
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        assert event.endpoints == {}

    def test_invalid_url_missing_scheme(self) -> None:
        """Test that URLs without scheme are rejected."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                endpoints={"health": "localhost:8080/health"},
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )
        error_str = str(exc_info.value)
        assert "endpoints" in error_str
        assert "Invalid URL" in error_str
        assert "health" in error_str

    def test_invalid_url_missing_host(self) -> None:
        """Test that URLs without host are rejected."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                endpoints={"health": "http:///health"},
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )
        error_str = str(exc_info.value)
        assert "endpoints" in error_str
        assert "Invalid URL" in error_str

    def test_invalid_url_plain_string(self) -> None:
        """Test that plain strings are rejected."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                endpoints={"api": "not-a-url"},
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )
        error_str = str(exc_info.value)
        assert "endpoints" in error_str
        assert "Invalid URL" in error_str
        assert "api" in error_str

    def test_invalid_url_empty_string(self) -> None:
        """Test that empty string URLs are rejected."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                endpoints={"health": ""},
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )
        error_str = str(exc_info.value)
        assert "endpoints" in error_str
        assert "Invalid URL" in error_str

    def test_invalid_url_relative_path(self) -> None:
        """Test that relative paths are rejected."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                endpoints={"health": "/health"},
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )
        error_str = str(exc_info.value)
        assert "endpoints" in error_str
        assert "Invalid URL" in error_str

    def test_multiple_endpoints_one_invalid(self) -> None:
        """Test that validation fails if any endpoint URL is invalid."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                endpoints={
                    "health": "http://localhost:8080/health",
                    "metrics": "invalid-url",
                    "api": "http://localhost:8080/api",
                },
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )
        error_str = str(exc_info.value)
        assert "endpoints" in error_str
        assert "Invalid URL" in error_str
        assert "metrics" in error_str

    def test_error_message_contains_endpoint_name(self) -> None:
        """Test that error message includes the invalid endpoint name."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                endpoints={"my_bad_endpoint": "no-scheme"},
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
            )
        error_str = str(exc_info.value)
        assert "my_bad_endpoint" in error_str

    def test_endpoints_serialization_roundtrip(self) -> None:
        """Test that endpoints survive JSON serialization roundtrip."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            endpoints={
                "health": "http://localhost:8080/health",
                "metrics": "https://api.example.com/metrics",
            },
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
        )
        json_str = event.model_dump_json()
        restored = ModelNodeIntrospectionEvent.model_validate_json(json_str)
        assert restored.endpoints == event.endpoints
