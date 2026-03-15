# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelIntrospectionPerformanceMetrics.

Tests validate:
- Default value instantiation
- Field constraint validation (ge=0)
- Frozen model immutability
- JSON serialization/deserialization roundtrip
- Threshold tracking fields
- captured_at auto-generation
- Schema examples validation

Related:
    - OMN-926: Add performance metrics to introspection events for observability
    - ModelNodeIntrospectionEvent.performance_metrics field
    - MixinNodeIntrospection.get_performance_metrics()
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from omnibase_infra.models.discovery import ModelIntrospectionPerformanceMetrics


class TestModelIntrospectionPerformanceMetricsInstantiation:
    """Tests for basic model instantiation and defaults."""

    def test_default_instantiation(self) -> None:
        """Test that default instantiation produces valid metrics with zero values."""
        metrics = ModelIntrospectionPerformanceMetrics()
        assert metrics.get_capabilities_ms == 0.0
        assert metrics.discover_capabilities_ms == 0.0
        assert metrics.get_endpoints_ms == 0.0
        assert metrics.get_current_state_ms == 0.0
        assert metrics.total_introspection_ms == 0.0
        assert metrics.cache_hit is False
        assert metrics.method_count == 0
        assert metrics.threshold_exceeded is False
        assert metrics.slow_operations == []
        assert metrics.captured_at is not None

    def test_full_instantiation(self) -> None:
        """Test instantiation with all fields explicitly set."""
        captured = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        metrics = ModelIntrospectionPerformanceMetrics(
            get_capabilities_ms=12.5,
            discover_capabilities_ms=8.2,
            get_endpoints_ms=0.5,
            get_current_state_ms=0.1,
            total_introspection_ms=21.3,
            cache_hit=False,
            method_count=15,
            threshold_exceeded=False,
            slow_operations=[],
            captured_at=captured,
        )
        assert metrics.get_capabilities_ms == 12.5
        assert metrics.discover_capabilities_ms == 8.2
        assert metrics.get_endpoints_ms == 0.5
        assert metrics.get_current_state_ms == 0.1
        assert metrics.total_introspection_ms == 21.3
        assert metrics.cache_hit is False
        assert metrics.method_count == 15
        assert metrics.threshold_exceeded is False
        assert metrics.slow_operations == []
        assert metrics.captured_at == captured

    def test_threshold_exceeded_with_slow_operations(self) -> None:
        """Test metrics indicating threshold violations."""
        metrics = ModelIntrospectionPerformanceMetrics(
            get_capabilities_ms=55.0,
            discover_capabilities_ms=45.0,
            get_endpoints_ms=0.2,
            get_current_state_ms=0.1,
            total_introspection_ms=100.3,
            cache_hit=False,
            method_count=42,
            threshold_exceeded=True,
            slow_operations=[
                "get_capabilities",
                "discover_capabilities",
                "total_introspection",
            ],
        )
        assert metrics.threshold_exceeded is True
        assert len(metrics.slow_operations) == 3
        assert "get_capabilities" in metrics.slow_operations

    def test_cache_hit_metrics(self) -> None:
        """Test metrics for a cache hit scenario (minimal timing)."""
        metrics = ModelIntrospectionPerformanceMetrics(
            total_introspection_ms=0.05,
            cache_hit=True,
            threshold_exceeded=False,
        )
        assert metrics.cache_hit is True
        assert metrics.total_introspection_ms == 0.05
        assert metrics.get_capabilities_ms == 0.0  # Not measured on cache hit


class TestModelIntrospectionPerformanceMetricsValidation:
    """Tests for field validation constraints."""

    def test_negative_timing_ms_rejected(self) -> None:
        """Test that negative timing values are rejected (ge=0 constraint)."""
        with pytest.raises(ValidationError):
            ModelIntrospectionPerformanceMetrics(get_capabilities_ms=-1.0)

    def test_negative_discover_capabilities_ms_rejected(self) -> None:
        """Test that negative discover_capabilities_ms is rejected."""
        with pytest.raises(ValidationError):
            ModelIntrospectionPerformanceMetrics(discover_capabilities_ms=-0.001)

    def test_negative_get_endpoints_ms_rejected(self) -> None:
        """Test that negative get_endpoints_ms is rejected."""
        with pytest.raises(ValidationError):
            ModelIntrospectionPerformanceMetrics(get_endpoints_ms=-5.0)

    def test_negative_get_current_state_ms_rejected(self) -> None:
        """Test that negative get_current_state_ms is rejected."""
        with pytest.raises(ValidationError):
            ModelIntrospectionPerformanceMetrics(get_current_state_ms=-0.1)

    def test_negative_total_introspection_ms_rejected(self) -> None:
        """Test that negative total_introspection_ms is rejected."""
        with pytest.raises(ValidationError):
            ModelIntrospectionPerformanceMetrics(total_introspection_ms=-10.0)

    def test_negative_method_count_rejected(self) -> None:
        """Test that negative method_count is rejected (ge=0 constraint)."""
        with pytest.raises(ValidationError):
            ModelIntrospectionPerformanceMetrics(method_count=-1)

    def test_zero_timing_values_allowed(self) -> None:
        """Test that zero timing values are valid."""
        metrics = ModelIntrospectionPerformanceMetrics(
            get_capabilities_ms=0.0,
            discover_capabilities_ms=0.0,
            get_endpoints_ms=0.0,
            get_current_state_ms=0.0,
            total_introspection_ms=0.0,
            method_count=0,
        )
        assert metrics.total_introspection_ms == 0.0
        assert metrics.method_count == 0

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelIntrospectionPerformanceMetrics(
                unknown_field="value",  # type: ignore[call-arg]
            )
        assert "unknown_field" in str(exc_info.value)


class TestModelIntrospectionPerformanceMetricsImmutability:
    """Tests for frozen model immutability."""

    def test_cannot_modify_timing_field(self) -> None:
        """Test that timing fields cannot be modified after creation."""
        metrics = ModelIntrospectionPerformanceMetrics(
            total_introspection_ms=25.0,
        )
        with pytest.raises(ValidationError):
            metrics.total_introspection_ms = 50.0  # type: ignore[misc]

    def test_cannot_modify_cache_hit(self) -> None:
        """Test that cache_hit cannot be modified after creation."""
        metrics = ModelIntrospectionPerformanceMetrics(cache_hit=False)
        with pytest.raises(ValidationError):
            metrics.cache_hit = True  # type: ignore[misc]

    def test_cannot_modify_slow_operations(self) -> None:
        """Test that slow_operations list reference cannot be reassigned."""
        metrics = ModelIntrospectionPerformanceMetrics(
            slow_operations=["get_capabilities"],
        )
        with pytest.raises(ValidationError):
            metrics.slow_operations = []  # type: ignore[misc]


class TestModelIntrospectionPerformanceMetricsSerialization:
    """Tests for JSON serialization and deserialization."""

    def test_json_roundtrip_default_values(self) -> None:
        """Test JSON serialization roundtrip with default values."""
        metrics = ModelIntrospectionPerformanceMetrics()
        json_str = metrics.model_dump_json()
        restored = ModelIntrospectionPerformanceMetrics.model_validate_json(json_str)
        assert restored.get_capabilities_ms == metrics.get_capabilities_ms
        assert restored.cache_hit == metrics.cache_hit
        assert restored.method_count == metrics.method_count

    def test_json_roundtrip_full_values(self) -> None:
        """Test JSON serialization roundtrip with all fields populated."""
        captured = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        metrics = ModelIntrospectionPerformanceMetrics(
            get_capabilities_ms=12.5,
            discover_capabilities_ms=8.2,
            get_endpoints_ms=0.5,
            get_current_state_ms=0.1,
            total_introspection_ms=21.3,
            cache_hit=False,
            method_count=15,
            threshold_exceeded=True,
            slow_operations=["get_capabilities", "total_introspection"],
            captured_at=captured,
        )
        json_str = metrics.model_dump_json()
        restored = ModelIntrospectionPerformanceMetrics.model_validate_json(json_str)
        assert restored.get_capabilities_ms == 12.5
        assert restored.discover_capabilities_ms == 8.2
        assert restored.get_endpoints_ms == 0.5
        assert restored.get_current_state_ms == 0.1
        assert restored.total_introspection_ms == 21.3
        assert restored.cache_hit is False
        assert restored.method_count == 15
        assert restored.threshold_exceeded is True
        assert restored.slow_operations == ["get_capabilities", "total_introspection"]

    def test_model_dump_dict_structure(self) -> None:
        """Test model_dump produces correct dict structure."""
        metrics = ModelIntrospectionPerformanceMetrics(
            get_capabilities_ms=10.0,
            method_count=5,
            threshold_exceeded=True,
            slow_operations=["get_capabilities"],
        )
        data = metrics.model_dump()
        assert isinstance(data, dict)
        assert data["get_capabilities_ms"] == 10.0
        assert data["method_count"] == 5
        assert data["threshold_exceeded"] is True
        assert data["slow_operations"] == ["get_capabilities"]
        assert "captured_at" in data

    def test_model_dump_mode_json(self) -> None:
        """Test model_dump with mode='json' for JSON-compatible output."""
        metrics = ModelIntrospectionPerformanceMetrics(
            total_introspection_ms=25.0,
        )
        data = metrics.model_dump(mode="json")
        # Datetime should be serialized as ISO string in JSON mode
        assert isinstance(data["captured_at"], str)
        # Numeric values remain numeric
        assert isinstance(data["total_introspection_ms"], float)


class TestModelIntrospectionPerformanceMetricsCapturedAt:
    """Tests for captured_at auto-generation."""

    def test_captured_at_auto_generated(self) -> None:
        """Test that captured_at is automatically set to current UTC time."""
        before = datetime.now(UTC)
        metrics = ModelIntrospectionPerformanceMetrics()
        after = datetime.now(UTC)
        assert before <= metrics.captured_at <= after

    def test_captured_at_explicit_value(self) -> None:
        """Test that captured_at can be explicitly set."""
        explicit_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        metrics = ModelIntrospectionPerformanceMetrics(captured_at=explicit_time)
        assert metrics.captured_at == explicit_time


class TestModelIntrospectionPerformanceMetricsEquality:
    """Tests for model equality comparison."""

    def test_equal_metrics_are_equal(self) -> None:
        """Test that two metrics with same values are equal."""
        captured = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        m1 = ModelIntrospectionPerformanceMetrics(
            total_introspection_ms=25.0,
            method_count=10,
            captured_at=captured,
        )
        m2 = ModelIntrospectionPerformanceMetrics(
            total_introspection_ms=25.0,
            method_count=10,
            captured_at=captured,
        )
        assert m1 == m2

    def test_different_values_not_equal(self) -> None:
        """Test that metrics with different values are not equal."""
        captured = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        m1 = ModelIntrospectionPerformanceMetrics(
            total_introspection_ms=25.0,
            captured_at=captured,
        )
        m2 = ModelIntrospectionPerformanceMetrics(
            total_introspection_ms=50.0,
            captured_at=captured,
        )
        assert m1 != m2


class TestModelIntrospectionPerformanceMetricsSchemaExamples:
    """Tests validating the JSON schema examples defined in model_config."""

    def test_schema_example_normal_performance(self) -> None:
        """Test that the 'normal' schema example validates successfully."""
        metrics = ModelIntrospectionPerformanceMetrics(
            get_capabilities_ms=12.5,
            discover_capabilities_ms=8.2,
            get_endpoints_ms=0.5,
            get_current_state_ms=0.1,
            total_introspection_ms=21.3,
            cache_hit=False,
            method_count=15,
            threshold_exceeded=False,
            slow_operations=[],
            captured_at=datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC),
        )
        assert not metrics.threshold_exceeded
        assert metrics.slow_operations == []

    def test_schema_example_degraded_performance(self) -> None:
        """Test that the 'degraded' schema example validates successfully."""
        metrics = ModelIntrospectionPerformanceMetrics(
            get_capabilities_ms=55.0,
            discover_capabilities_ms=45.0,
            get_endpoints_ms=0.2,
            get_current_state_ms=0.1,
            total_introspection_ms=100.3,
            cache_hit=False,
            method_count=42,
            threshold_exceeded=True,
            slow_operations=[
                "get_capabilities",
                "discover_capabilities",
                "total_introspection",
            ],
            captured_at=datetime(2025, 1, 15, 10, 31, 0, tzinfo=UTC),
        )
        assert metrics.threshold_exceeded is True
        assert len(metrics.slow_operations) == 3
