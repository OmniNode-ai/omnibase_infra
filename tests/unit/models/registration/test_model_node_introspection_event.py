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

from omnibase_infra.models.registration import ModelNodeIntrospectionEvent


class TestModelNodeIntrospectionEventBasicInstantiation:
    """Tests for basic model instantiation."""

    def test_valid_instantiation_required_fields_only(self) -> None:
        """Test creating event with only required fields."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
        )
        assert event.node_id == test_node_id
        assert event.node_type == "effect"
        assert event.capabilities == {}
        assert event.endpoints == {}
        assert event.node_role is None
        assert event.metadata == {}
        assert event.correlation_id is None
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
        assert event.capabilities == {"processing": True, "batch_size": 100}
        assert event.endpoints == {
            "health": "http://localhost:8080/health",
            "metrics": "http://localhost:8080/metrics",
        }
        assert event.node_role == "processor"
        assert event.metadata == {"version": "1.0.0", "environment": "production"}
        assert event.correlation_id == correlation_id
        assert event.network_id == "network-001"
        assert event.deployment_id == "deploy-001"
        assert event.epoch == 1
        assert event.timestamp == timestamp


class TestModelNodeIntrospectionEventNodeTypeValidation:
    """Tests for node_type Literal validation."""

    def test_valid_node_type_effect(self) -> None:
        """Test that 'effect' is a valid node_type."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(node_id=test_node_id, node_type="effect")
        assert event.node_type == "effect"

    def test_valid_node_type_compute(self) -> None:
        """Test that 'compute' is a valid node_type."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(node_id=test_node_id, node_type="compute")
        assert event.node_type == "compute"

    def test_valid_node_type_reducer(self) -> None:
        """Test that 'reducer' is a valid node_type."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(node_id=test_node_id, node_type="reducer")
        assert event.node_type == "reducer"

    def test_valid_node_type_orchestrator(self) -> None:
        """Test that 'orchestrator' is a valid node_type."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id, node_type="orchestrator"
        )
        assert event.node_type == "orchestrator"

    def test_invalid_node_type_raises_validation_error(self) -> None:
        """Test that invalid node_type raises ValidationError."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="invalid_type",  # type: ignore[arg-type]
            )
        assert "node_type" in str(exc_info.value)

    def test_invalid_node_type_empty_string(self) -> None:
        """Test that empty string node_type raises ValidationError."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError):
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="",  # type: ignore[arg-type]
            )

    def test_invalid_node_type_none(self) -> None:
        """Test that None node_type raises ValidationError."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError):
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type=None,  # type: ignore[arg-type]
            )


class TestModelNodeIntrospectionEventSerialization:
    """Tests for JSON serialization and deserialization."""

    def test_json_serialization_roundtrip_minimal(self) -> None:
        """Test JSON serialization and deserialization with minimal fields."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="reducer",
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
            capabilities={"database": True},
        )
        data = event.model_dump()
        assert isinstance(data, dict)
        assert data["node_id"] == test_node_id
        assert data["node_type"] == "effect"
        assert data["capabilities"] == {"database": True}

    def test_model_dump_mode_json(self) -> None:
        """Test model_dump with mode='json' for JSON-compatible output."""
        test_node_id = uuid4()
        correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="compute",
            correlation_id=correlation_id,
        )
        data = event.model_dump(mode="json")
        # UUID should be serialized as string in JSON mode
        assert data["node_id"] == str(test_node_id)
        assert data["correlation_id"] == str(correlation_id)
        # Datetime should be serialized as ISO string
        assert isinstance(data["timestamp"], str)


class TestModelNodeIntrospectionEventTimestamp:
    """Tests for timestamp auto-generation."""

    def test_timestamp_auto_generation(self) -> None:
        """Test that timestamp is auto-generated when not provided."""
        test_node_id = uuid4()
        before = datetime.now(UTC)
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="orchestrator",
        )
        after = datetime.now(UTC)
        assert event.timestamp is not None
        assert before <= event.timestamp <= after

    def test_timestamp_explicit_value(self) -> None:
        """Test that explicit timestamp is preserved."""
        test_node_id = uuid4()
        explicit_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            timestamp=explicit_time,
        )
        assert event.timestamp == explicit_time

    def test_timestamp_is_datetime(self) -> None:
        """Test that timestamp is a datetime object."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="compute",
        )
        assert isinstance(event.timestamp, datetime)


class TestModelNodeIntrospectionEventImmutability:
    """Tests for frozen model immutability."""

    def test_frozen_model_cannot_modify_node_id(self) -> None:
        """Test that node_id cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
        )
        with pytest.raises(ValidationError):
            event.node_id = uuid4()  # type: ignore[misc]

    def test_frozen_model_cannot_modify_node_type(self) -> None:
        """Test that node_type cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
        )
        with pytest.raises(ValidationError):
            event.node_type = "compute"  # type: ignore[misc]

    def test_frozen_model_cannot_modify_capabilities(self) -> None:
        """Test that capabilities dict reference cannot be reassigned."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            capabilities={"original": True},
        )
        with pytest.raises(ValidationError):
            event.capabilities = {"modified": True}  # type: ignore[misc]

    def test_frozen_model_cannot_modify_correlation_id(self) -> None:
        """Test that correlation_id cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            correlation_id=uuid4(),
        )
        with pytest.raises(ValidationError):
            event.correlation_id = uuid4()  # type: ignore[misc]

    def test_frozen_model_cannot_modify_timestamp(self) -> None:
        """Test that timestamp cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
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
        )
        assert event.capabilities["processing"] is True
        assert event.capabilities["max_batch"] == 1000
        assert event.capabilities["supported_types"] == ["json", "xml", "csv"]
        assert event.capabilities["config"]["timeout"] == 30

    def test_unicode_in_fields(self) -> None:
        """Test Unicode characters in string fields."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_role="处理器",
            metadata={"description": "Узел обработки"},
        )
        assert event.node_id == test_node_id
        assert event.node_role == "处理器"
        assert event.metadata["description"] == "Узел обработки"

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden by model config."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=test_node_id,
                node_type="effect",
                extra_field="not_allowed",  # type: ignore[call-arg]
            )
        assert "extra_field" in str(exc_info.value)

    def test_negative_epoch_allowed(self) -> None:
        """Test that negative epoch values are allowed (no constraint)."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            epoch=-1,
        )
        assert event.epoch == -1

    def test_zero_epoch_allowed(self) -> None:
        """Test that zero epoch is allowed."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            epoch=0,
        )
        assert event.epoch == 0

    def test_large_epoch_allowed(self) -> None:
        """Test that large epoch values are allowed."""
        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            epoch=2**31,
        )
        assert event.epoch == 2**31


class TestModelNodeIntrospectionEventFromAttributes:
    """Tests for from_attributes configuration (ORM mode)."""

    def test_from_dict_like_object(self) -> None:
        """Test creating model from dict-like object."""
        test_node_id = uuid4()

        class DictLike:
            def __init__(self, node_id: UUID) -> None:
                self.node_id = node_id
                self.node_type = "compute"
                self.capabilities: dict[str, bool] = {}
                self.endpoints: dict[str, str] = {}
                self.node_role = None
                self.metadata: dict[str, str] = {}
                self.correlation_id = None
                self.network_id = None
                self.deployment_id = None
                self.epoch = None
                self.timestamp = datetime.now(UTC)

        obj = DictLike(test_node_id)
        event = ModelNodeIntrospectionEvent.model_validate(obj)
        assert event.node_id == test_node_id
        assert event.node_type == "compute"
