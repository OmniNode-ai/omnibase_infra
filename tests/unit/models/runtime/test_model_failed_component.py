# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelFailedComponent.

This test module provides comprehensive coverage for the ModelFailedComponent model,
which represents a component that failed during shutdown operations.

Tests cover:
    - Model construction and validation
    - Field validation (min_length constraints)
    - __str__ method behavior
    - Model configuration (frozen, extra forbid)
    - Typical usage patterns in shutdown scenarios

.. versionadded:: 0.7.0
    Created as part of PR #92 review to add dedicated model tests.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_failed_component import ModelFailedComponent


class TestModelFailedComponentConstruction:
    """Tests for ModelFailedComponent construction and validation."""

    def test_construct_with_required_fields(self) -> None:
        """Verify model can be constructed with required fields."""
        failed = ModelFailedComponent(
            component_name="EventBusKafka",
            error_message="Connection timeout during shutdown",
        )

        assert failed.component_name == "EventBusKafka"
        assert failed.error_message == "Connection timeout during shutdown"

    def test_construct_with_various_component_names(self) -> None:
        """Verify various valid component names are accepted."""
        test_cases = [
            "EventBusKafka",
            "PostgresAdapter",
            "ConsulHealthChecker",
            "RuntimeHostProcess",
            "ServiceMessageDispatchEngine",
            "a",  # Single character is valid (min_length=1)
        ]

        for component_name in test_cases:
            failed = ModelFailedComponent(
                component_name=component_name,
                error_message="Test error",
            )
            assert failed.component_name == component_name

    def test_reject_empty_component_name(self) -> None:
        """Verify empty component_name is rejected (min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name="",
                error_message="Test error",
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("component_name",)
        assert "String should have at least 1 character" in str(errors[0]["msg"])

    def test_reject_empty_error_message(self) -> None:
        """Verify empty error_message is rejected (min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name="TestComponent",
                error_message="",
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("error_message",)
        assert "String should have at least 1 character" in str(errors[0]["msg"])


class TestModelFailedComponentStr:
    """Tests for ModelFailedComponent.__str__ method."""

    def test_str_format(self) -> None:
        """Verify __str__ returns expected format."""
        failed = ModelFailedComponent(
            component_name="EventBusKafka",
            error_message="Connection timeout",
        )

        expected = "EventBusKafka: Connection timeout"
        assert str(failed) == expected

    def test_str_with_long_error_message(self) -> None:
        """Verify __str__ handles long error messages."""
        long_error = "A" * 500
        failed = ModelFailedComponent(
            component_name="TestComponent",
            error_message=long_error,
        )

        expected = f"TestComponent: {long_error}"
        assert str(failed) == expected

    def test_str_with_special_characters(self) -> None:
        """Verify __str__ handles special characters in error message."""
        failed = ModelFailedComponent(
            component_name="TestComponent",
            error_message='Error: "Failed" with <special> chars & more',
        )

        expected = 'TestComponent: Error: "Failed" with <special> chars & more'
        assert str(failed) == expected


class TestModelFailedComponentConfiguration:
    """Tests for model configuration (frozen, extra forbid, strict)."""

    def test_model_is_frozen(self) -> None:
        """Verify model instances are immutable (frozen=True)."""
        failed = ModelFailedComponent(
            component_name="TestComponent",
            error_message="Test error",
        )

        with pytest.raises(ValidationError):
            failed.component_name = "NewName"  # type: ignore[misc]

    def test_model_forbids_extra_fields(self) -> None:
        """Verify model rejects extra fields (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name="TestComponent",
                error_message="Test error",
                extra_field="not allowed",  # type: ignore[call-arg]
            )

        errors = exc_info.value.errors()
        assert any("extra" in str(e).lower() for e in errors)

    def test_model_uses_strict_mode(self) -> None:
        """Verify model uses strict type checking.

        In strict mode, types must match exactly - no coercion from
        other types like int to str.
        """
        with pytest.raises(ValidationError):
            ModelFailedComponent(
                component_name=123,  # type: ignore[arg-type]
                error_message="Test error",
            )


class TestModelFailedComponentUsagePatterns:
    """Tests demonstrating typical usage patterns."""

    def test_shutdown_failure_collection(self) -> None:
        """Demonstrate collecting multiple shutdown failures."""
        failures: list[ModelFailedComponent] = []

        # Simulate multiple failures during shutdown
        failures.append(
            ModelFailedComponent(
                component_name="EventBusKafka",
                error_message="Connection refused",
            )
        )
        failures.append(
            ModelFailedComponent(
                component_name="PostgresAdapter",
                error_message="Timeout waiting for transactions to complete",
            )
        )
        failures.append(
            ModelFailedComponent(
                component_name="ConsulClient",
                error_message="Failed to deregister service",
            )
        )

        assert len(failures) == 3
        assert all(isinstance(f, ModelFailedComponent) for f in failures)

        # Can be easily formatted for logging
        failure_lines = [str(f) for f in failures]
        assert failure_lines == [
            "EventBusKafka: Connection refused",
            "PostgresAdapter: Timeout waiting for transactions to complete",
            "ConsulClient: Failed to deregister service",
        ]

    def test_equality_comparison(self) -> None:
        """Verify two models with same values are equal."""
        failed1 = ModelFailedComponent(
            component_name="TestComponent",
            error_message="Test error",
        )
        failed2 = ModelFailedComponent(
            component_name="TestComponent",
            error_message="Test error",
        )

        assert failed1 == failed2

    def test_inequality_on_different_values(self) -> None:
        """Verify models with different values are not equal."""
        failed1 = ModelFailedComponent(
            component_name="Component1",
            error_message="Error 1",
        )
        failed2 = ModelFailedComponent(
            component_name="Component2",
            error_message="Error 2",
        )

        assert failed1 != failed2

    def test_hashable_for_sets(self) -> None:
        """Verify model instances are hashable (frozen models are hashable)."""
        failed1 = ModelFailedComponent(
            component_name="TestComponent",
            error_message="Test error",
        )
        failed2 = ModelFailedComponent(
            component_name="TestComponent",
            error_message="Test error",
        )
        failed3 = ModelFailedComponent(
            component_name="OtherComponent",
            error_message="Other error",
        )

        # Can be used in sets
        failure_set = {failed1, failed2, failed3}
        assert len(failure_set) == 2  # failed1 and failed2 are equal

    def test_from_attributes_compatibility(self) -> None:
        """Verify from_attributes config allows ORM-style construction.

        The model has from_attributes=True, enabling construction from
        objects with matching attributes (useful for ORM/pytest-xdist).
        """

        class MockDBRow:
            """Mock database row or object with attributes."""

            def __init__(self) -> None:
                self.component_name = "FromAttributes"
                self.error_message = "Loaded from ORM"

        # model_validate with from_attributes=True
        row = MockDBRow()
        failed = ModelFailedComponent.model_validate(row, from_attributes=True)

        assert failed.component_name == "FromAttributes"
        assert failed.error_message == "Loaded from ORM"


__all__ = [
    "TestModelFailedComponentConstruction",
    "TestModelFailedComponentStr",
    "TestModelFailedComponentConfiguration",
    "TestModelFailedComponentUsagePatterns",
]
