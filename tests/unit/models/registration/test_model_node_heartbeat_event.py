# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelNodeHeartbeatEvent.

Tests validate:
- Required field instantiation
- Optional field handling
- Non-negative constraint validation for uptime_seconds and active_operations_count
- JSON serialization/deserialization roundtrip
- Timestamp auto-generation
- Frozen model immutability
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.models.registration import ModelNodeHeartbeatEvent


class TestModelNodeHeartbeatEventBasicInstantiation:
    """Tests for basic model instantiation."""

    def test_valid_instantiation_required_fields_only(self) -> None:
        """Test creating event with only required fields."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=3600.0,
        )
        assert event.node_id == test_node_id
        assert event.node_type == "effect"
        assert event.uptime_seconds == 3600.0
        assert event.active_operations_count == 0  # Default value
        assert event.memory_usage_mb is None
        assert event.cpu_usage_percent is None
        assert event.correlation_id is None

    def test_valid_instantiation_all_fields(self) -> None:
        """Test creating event with all fields populated."""
        test_node_id = uuid4()
        correlation_id = uuid4()
        timestamp = datetime.now(UTC)
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="compute",
            uptime_seconds=7200.5,
            active_operations_count=10,
            memory_usage_mb=512.0,
            cpu_usage_percent=45.5,
            correlation_id=correlation_id,
            timestamp=timestamp,
        )
        assert event.node_id == test_node_id
        assert event.node_type == "compute"
        assert event.uptime_seconds == 7200.5
        assert event.active_operations_count == 10
        assert event.memory_usage_mb == 512.0
        assert event.cpu_usage_percent == 45.5
        assert event.correlation_id == correlation_id
        assert event.timestamp == timestamp

    def test_valid_node_type_string_values(self) -> None:
        """Test that node_type accepts any string (not constrained like introspection)."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
        )
        assert event.node_type == "effect"

        test_node_id2 = uuid4()
        event2 = ModelNodeHeartbeatEvent(
            node_id=test_node_id2,
            node_type="custom_type",  # Heartbeat allows any string
            uptime_seconds=100.0,
        )
        assert event2.node_type == "custom_type"


class TestModelNodeHeartbeatEventUptimeValidation:
    """Tests for uptime_seconds validation (ge=0 constraint)."""

    def test_negative_uptime_seconds_raises_validation_error(self) -> None:
        """Test that negative uptime_seconds raises ValidationError."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeHeartbeatEvent(
                node_id=test_node_id,
                node_type="effect",
                uptime_seconds=-1.0,
            )
        assert "uptime_seconds" in str(exc_info.value)

    def test_negative_uptime_seconds_large_negative(self) -> None:
        """Test that large negative uptime_seconds raises ValidationError."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError):
            ModelNodeHeartbeatEvent(
                node_id=test_node_id,
                node_type="effect",
                uptime_seconds=-99999.0,
            )

    def test_zero_uptime_seconds_allowed(self) -> None:
        """Test that zero uptime_seconds is valid."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=0.0,
        )
        assert event.uptime_seconds == 0.0

    def test_very_small_positive_uptime_allowed(self) -> None:
        """Test that very small positive uptime is valid."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=0.001,
        )
        assert event.uptime_seconds == 0.001

    def test_large_uptime_seconds_allowed(self) -> None:
        """Test that large uptime values are allowed."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=365 * 24 * 3600.0,  # One year in seconds
        )
        assert event.uptime_seconds == 365 * 24 * 3600.0


class TestModelNodeHeartbeatEventActiveOperationsValidation:
    """Tests for active_operations_count validation (ge=0 constraint)."""

    def test_negative_active_operations_count_raises_validation_error(self) -> None:
        """Test that negative active_operations_count raises ValidationError."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeHeartbeatEvent(
                node_id=test_node_id,
                node_type="effect",
                uptime_seconds=100.0,
                active_operations_count=-1,
            )
        assert "active_operations_count" in str(exc_info.value)

    def test_negative_active_operations_large_negative(self) -> None:
        """Test that large negative active_operations_count raises ValidationError."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError):
            ModelNodeHeartbeatEvent(
                node_id=test_node_id,
                node_type="effect",
                uptime_seconds=100.0,
                active_operations_count=-100,
            )

    def test_zero_active_operations_count_allowed(self) -> None:
        """Test that zero active_operations_count is valid."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            active_operations_count=0,
        )
        assert event.active_operations_count == 0

    def test_positive_active_operations_count_allowed(self) -> None:
        """Test that positive active_operations_count is valid."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            active_operations_count=50,
        )
        assert event.active_operations_count == 50

    def test_large_active_operations_count_allowed(self) -> None:
        """Test that large active_operations_count is valid."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            active_operations_count=10000,
        )
        assert event.active_operations_count == 10000


class TestModelNodeHeartbeatEventSerialization:
    """Tests for JSON serialization and deserialization."""

    def test_json_serialization_roundtrip_minimal(self) -> None:
        """Test JSON serialization and deserialization with minimal fields."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="reducer",
            uptime_seconds=1800.0,
        )
        json_str = event.model_dump_json()
        restored = ModelNodeHeartbeatEvent.model_validate_json(json_str)
        assert restored.node_id == event.node_id
        assert restored.node_type == event.node_type
        assert restored.uptime_seconds == event.uptime_seconds
        assert restored.active_operations_count == event.active_operations_count

    def test_json_serialization_roundtrip_full(self) -> None:
        """Test JSON serialization and deserialization with all fields."""
        test_node_id = uuid4()
        correlation_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="orchestrator",
            uptime_seconds=86400.0,
            active_operations_count=25,
            memory_usage_mb=1024.0,
            cpu_usage_percent=75.5,
            correlation_id=correlation_id,
        )
        json_str = event.model_dump_json()
        restored = ModelNodeHeartbeatEvent.model_validate_json(json_str)

        assert restored.node_id == event.node_id
        assert restored.node_type == event.node_type
        assert restored.uptime_seconds == event.uptime_seconds
        assert restored.active_operations_count == event.active_operations_count
        assert restored.memory_usage_mb == event.memory_usage_mb
        assert restored.cpu_usage_percent == event.cpu_usage_percent
        assert restored.correlation_id == event.correlation_id
        # Timestamps should match within reasonable precision
        assert abs((restored.timestamp - event.timestamp).total_seconds()) < 1

    def test_model_dump_dict(self) -> None:
        """Test model_dump produces correct dict structure."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=500.0,
            active_operations_count=3,
        )
        data = event.model_dump()
        assert isinstance(data, dict)
        assert data["node_id"] == test_node_id
        assert data["node_type"] == "effect"
        assert data["uptime_seconds"] == 500.0
        assert data["active_operations_count"] == 3

    def test_model_dump_mode_json(self) -> None:
        """Test model_dump with mode='json' for JSON-compatible output."""
        test_node_id = uuid4()
        correlation_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="compute",
            uptime_seconds=1000.0,
            correlation_id=correlation_id,
        )
        data = event.model_dump(mode="json")
        # UUID should be serialized as string in JSON mode
        assert data["node_id"] == str(test_node_id)
        assert data["correlation_id"] == str(correlation_id)
        # Datetime should be serialized as ISO string
        assert isinstance(data["timestamp"], str)


class TestModelNodeHeartbeatEventTimestamp:
    """Tests for timestamp auto-generation."""

    def test_timestamp_auto_generation(self) -> None:
        """Test that timestamp is auto-generated when not provided."""
        test_node_id = uuid4()
        before = datetime.now(UTC)
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
        )
        after = datetime.now(UTC)
        assert event.timestamp is not None
        assert before <= event.timestamp <= after

    def test_timestamp_explicit_value(self) -> None:
        """Test that explicit timestamp is preserved."""
        test_node_id = uuid4()
        explicit_time = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=200.0,
            timestamp=explicit_time,
        )
        assert event.timestamp == explicit_time

    def test_timestamp_is_datetime(self) -> None:
        """Test that timestamp is a datetime object."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="compute",
            uptime_seconds=300.0,
        )
        assert isinstance(event.timestamp, datetime)


class TestModelNodeHeartbeatEventImmutability:
    """Tests for frozen model immutability."""

    def test_frozen_model_cannot_modify_node_id(self) -> None:
        """Test that node_id cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
        )
        with pytest.raises(ValidationError):
            event.node_id = uuid4()  # type: ignore[misc]

    def test_frozen_model_cannot_modify_node_type(self) -> None:
        """Test that node_type cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
        )
        with pytest.raises(ValidationError):
            event.node_type = "compute"  # type: ignore[misc]

    def test_frozen_model_cannot_modify_uptime_seconds(self) -> None:
        """Test that uptime_seconds cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
        )
        with pytest.raises(ValidationError):
            event.uptime_seconds = 200.0  # type: ignore[misc]

    def test_frozen_model_cannot_modify_active_operations_count(self) -> None:
        """Test that active_operations_count cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            active_operations_count=5,
        )
        with pytest.raises(ValidationError):
            event.active_operations_count = 10  # type: ignore[misc]

    def test_frozen_model_cannot_modify_memory_usage(self) -> None:
        """Test that memory_usage_mb cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            memory_usage_mb=512.0,
        )
        with pytest.raises(ValidationError):
            event.memory_usage_mb = 1024.0  # type: ignore[misc]

    def test_frozen_model_cannot_modify_cpu_usage(self) -> None:
        """Test that cpu_usage_percent cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            cpu_usage_percent=50.0,
        )
        with pytest.raises(ValidationError):
            event.cpu_usage_percent = 75.0  # type: ignore[misc]

    def test_frozen_model_cannot_modify_correlation_id(self) -> None:
        """Test that correlation_id cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            correlation_id=uuid4(),
        )
        with pytest.raises(ValidationError):
            event.correlation_id = uuid4()  # type: ignore[misc]

    def test_frozen_model_cannot_modify_timestamp(self) -> None:
        """Test that timestamp cannot be modified after creation."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
        )
        with pytest.raises(ValidationError):
            event.timestamp = datetime.now(UTC)  # type: ignore[misc]


class TestModelNodeHeartbeatEventResourceMetrics:
    """Tests for optional resource usage metrics."""

    def test_memory_usage_none_by_default(self) -> None:
        """Test that memory_usage_mb is None by default."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
        )
        assert event.memory_usage_mb is None

    def test_cpu_usage_none_by_default(self) -> None:
        """Test that cpu_usage_percent is None by default."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
        )
        assert event.cpu_usage_percent is None

    def test_memory_usage_zero_allowed(self) -> None:
        """Test that zero memory_usage_mb is valid."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            memory_usage_mb=0.0,
        )
        assert event.memory_usage_mb == 0.0

    def test_cpu_usage_zero_allowed(self) -> None:
        """Test that zero cpu_usage_percent is valid."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            cpu_usage_percent=0.0,
        )
        assert event.cpu_usage_percent == 0.0

    def test_cpu_usage_100_percent_allowed(self) -> None:
        """Test that 100% cpu_usage_percent is valid."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            cpu_usage_percent=100.0,
        )
        assert event.cpu_usage_percent == 100.0

    def test_cpu_usage_over_100_allowed(self) -> None:
        """Test that >100% cpu_usage_percent is allowed (no constraint)."""
        test_node_id = uuid4()
        # Some systems can report >100% for multi-core usage
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            cpu_usage_percent=400.0,  # 4 cores at 100%
        )
        assert event.cpu_usage_percent == 400.0

    def test_large_memory_usage_allowed(self) -> None:
        """Test that large memory_usage_mb is allowed."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=100.0,
            memory_usage_mb=1024 * 1024,  # 1 TB
        )
        assert event.memory_usage_mb == 1024 * 1024


class TestModelNodeHeartbeatEventEdgeCases:
    """Tests for edge cases and special values."""

    def test_invalid_node_id_empty_string_raises_error(self) -> None:
        """Test that empty string is not allowed for node_id (UUID type)."""
        with pytest.raises(ValidationError):
            ModelNodeHeartbeatEvent(
                node_id="",  # type: ignore[arg-type]
                node_type="effect",
                uptime_seconds=100.0,
            )

    def test_empty_string_node_type(self) -> None:
        """Test that empty string is allowed for node_type."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="",
            uptime_seconds=100.0,
        )
        assert event.node_type == ""

    def test_unicode_in_node_type(self) -> None:
        """Test Unicode characters in string fields."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="效果节点",
            uptime_seconds=100.0,
        )
        assert event.node_id == test_node_id
        assert event.node_type == "效果节点"

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden by model config."""
        test_node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeHeartbeatEvent(
                node_id=test_node_id,
                node_type="effect",
                uptime_seconds=100.0,
                extra_field="not_allowed",  # type: ignore[call-arg]
            )
        assert "extra_field" in str(exc_info.value)

    def test_float_precision_preserved(self) -> None:
        """Test that float precision is preserved for metrics."""
        test_node_id = uuid4()
        event = ModelNodeHeartbeatEvent(
            node_id=test_node_id,
            node_type="effect",
            uptime_seconds=3600.123456789,
            memory_usage_mb=256.789012345,
            cpu_usage_percent=33.333333333,
        )
        assert event.uptime_seconds == 3600.123456789
        assert event.memory_usage_mb == 256.789012345
        assert event.cpu_usage_percent == 33.333333333


class TestModelNodeHeartbeatEventFromAttributes:
    """Tests for from_attributes configuration (ORM mode)."""

    def test_from_dict_like_object(self) -> None:
        """Test creating model from dict-like object."""
        test_node_id = uuid4()

        class DictLike:
            def __init__(self, node_id: UUID) -> None:
                self.node_id = node_id
                self.node_type = "compute"
                self.uptime_seconds = 1234.5
                self.active_operations_count = 5
                self.memory_usage_mb = None
                self.cpu_usage_percent = None
                self.correlation_id = None
                self.timestamp = datetime.now(UTC)

        obj = DictLike(test_node_id)
        event = ModelNodeHeartbeatEvent.model_validate(obj)
        assert event.node_id == test_node_id
        assert event.node_type == "compute"
        assert event.uptime_seconds == 1234.5
        assert event.active_operations_count == 5
