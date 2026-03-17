# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelServiceHeartbeatEvent.

Tests validate:
- Required field instantiation with valid data
- Frozen model immutability
- Extra fields forbidden
- JSON serialization keys match expected set
- Roundtrip serialization (model_dump -> model_validate)
- Non-negative constraint validation for numeric fields
- Status literal enforcement
- Timezone-aware datetime validation
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from omnibase_infra.models.registration import ModelServiceHeartbeatEvent

pytestmark = pytest.mark.unit

# Fixed test timestamp for deterministic testing (time injection pattern)
TEST_TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def _make_valid_event(**overrides: object) -> ModelServiceHeartbeatEvent:
    """Create a valid ModelServiceHeartbeatEvent with sensible defaults."""
    defaults: dict[str, object] = {
        "service_id": "omninode-runtime-abc123",
        "service_name": "omninode-runtime",
        "status": "healthy",
        "uptime_ms": 3600000,
        "restart_count": 0,
        "memory_usage_mb": 256.5,
        "cpu_percent": 12.3,
        "version": "1.2.3",
        "emitted_at": TEST_TIMESTAMP,
    }
    defaults.update(overrides)
    return ModelServiceHeartbeatEvent(**defaults)  # type: ignore[arg-type]


class TestModelServiceHeartbeatEventInstantiation:
    """Tests for basic model instantiation."""

    def test_valid_instantiation(self) -> None:
        """Test creating event with all required fields."""
        event = _make_valid_event()
        assert event.service_id == "omninode-runtime-abc123"
        assert event.service_name == "omninode-runtime"
        assert event.status == "healthy"
        assert event.uptime_ms == 3600000
        assert event.restart_count == 0
        assert event.memory_usage_mb == 256.5
        assert event.cpu_percent == 12.3
        assert event.version == "1.2.3"
        assert event.emitted_at == TEST_TIMESTAMP

    def test_status_degraded(self) -> None:
        """Test creating event with degraded status."""
        event = _make_valid_event(status="degraded")
        assert event.status == "degraded"

    def test_status_unhealthy(self) -> None:
        """Test creating event with unhealthy status."""
        event = _make_valid_event(status="unhealthy")
        assert event.status == "unhealthy"

    def test_invalid_status_rejected(self) -> None:
        """Test that invalid status values are rejected."""
        with pytest.raises(ValidationError, match="status"):
            _make_valid_event(status="unknown")


class TestModelServiceHeartbeatEventFrozen:
    """Tests for frozen immutability."""

    def test_frozen_service_id(self) -> None:
        """Test that service_id cannot be modified after creation."""
        event = _make_valid_event()
        with pytest.raises(ValidationError):
            event.service_id = "new-id"  # type: ignore[misc]

    def test_frozen_status(self) -> None:
        """Test that status cannot be modified after creation."""
        event = _make_valid_event()
        with pytest.raises(ValidationError):
            event.status = "unhealthy"  # type: ignore[misc]

    def test_frozen_uptime_ms(self) -> None:
        """Test that uptime_ms cannot be modified after creation."""
        event = _make_valid_event()
        with pytest.raises(ValidationError):
            event.uptime_ms = 9999  # type: ignore[misc]


class TestModelServiceHeartbeatEventExtraForbid:
    """Tests for extra fields forbidden."""

    def test_extra_field_rejected(self) -> None:
        """Test that extra fields raise ValidationError."""
        with pytest.raises(ValidationError, match="extra_field"):
            _make_valid_event(extra_field="not allowed")


class TestModelServiceHeartbeatEventSerialization:
    """Tests for JSON serialization."""

    def test_json_keys_match_expected_set(self) -> None:
        """Test that model_dump keys match the expected field set."""
        event = _make_valid_event()
        data = event.model_dump()
        expected_keys = {
            "service_id",
            "service_name",
            "status",
            "uptime_ms",
            "restart_count",
            "memory_usage_mb",
            "cpu_percent",
            "version",
            "emitted_at",
        }
        assert set(data.keys()) == expected_keys

    def test_roundtrip_serialization(self) -> None:
        """Test model_dump -> model_validate roundtrip preserves all data."""
        event = _make_valid_event()
        data = event.model_dump()
        restored = ModelServiceHeartbeatEvent.model_validate(data)
        assert restored == event

    def test_json_roundtrip(self) -> None:
        """Test model_dump_json -> model_validate_json roundtrip."""
        event = _make_valid_event()
        json_str = event.model_dump_json()
        restored = ModelServiceHeartbeatEvent.model_validate_json(json_str)
        assert restored == event


class TestModelServiceHeartbeatEventConstraints:
    """Tests for field constraints."""

    def test_negative_uptime_ms_rejected(self) -> None:
        """Test that negative uptime_ms raises ValidationError."""
        with pytest.raises(ValidationError, match="uptime_ms"):
            _make_valid_event(uptime_ms=-1)

    def test_negative_restart_count_rejected(self) -> None:
        """Test that negative restart_count raises ValidationError."""
        with pytest.raises(ValidationError, match="restart_count"):
            _make_valid_event(restart_count=-1)

    def test_negative_memory_usage_mb_rejected(self) -> None:
        """Test that negative memory_usage_mb raises ValidationError."""
        with pytest.raises(ValidationError, match="memory_usage_mb"):
            _make_valid_event(memory_usage_mb=-0.1)

    def test_negative_cpu_percent_rejected(self) -> None:
        """Test that negative cpu_percent raises ValidationError."""
        with pytest.raises(ValidationError, match="cpu_percent"):
            _make_valid_event(cpu_percent=-1.0)

    def test_zero_values_accepted(self) -> None:
        """Test that zero values are accepted for all numeric fields."""
        event = _make_valid_event(
            uptime_ms=0,
            restart_count=0,
            memory_usage_mb=0.0,
            cpu_percent=0.0,
        )
        assert event.uptime_ms == 0
        assert event.restart_count == 0
        assert event.memory_usage_mb == 0.0
        assert event.cpu_percent == 0.0

    def test_naive_datetime_rejected(self) -> None:
        """Test that naive (non-timezone-aware) datetime is rejected."""
        naive_dt = datetime(2025, 1, 15, 12, 0, 0)
        with pytest.raises(ValidationError, match="emitted_at"):
            _make_valid_event(emitted_at=naive_dt)
