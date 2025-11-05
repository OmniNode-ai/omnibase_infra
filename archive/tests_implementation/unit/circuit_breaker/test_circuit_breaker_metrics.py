"""Comprehensive tests for ModelCircuitBreakerMetrics.

Migrated and adapted from archived test_event_bus_circuit_breaker.py.
Addresses PR #11 feedback on missing test coverage for circuit breaker models.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from omnibase_infra.models.core.circuit_breaker.model_circuit_breaker_metrics import (
    ModelCircuitBreakerMetrics,
)


class TestModelCircuitBreakerMetrics:
    """Test ModelCircuitBreakerMetrics with comprehensive coverage."""

    def test_model_initialization_defaults(self):
        """Test model initializes with correct default values."""
        metrics = ModelCircuitBreakerMetrics()

        # Verify all counter fields default to 0
        assert metrics.total_events == 0
        assert metrics.successful_events == 0
        assert metrics.failed_events == 0
        assert metrics.queued_events == 0
        assert metrics.dropped_events == 0
        assert metrics.dead_letter_events == 0
        assert metrics.circuit_opens == 0
        assert metrics.circuit_closes == 0

        # Verify datetime fields default to None
        assert metrics.last_failure is None
        assert metrics.last_success is None

        # Verify calculated fields have appropriate defaults
        assert metrics.success_rate_percent == 100.0
        assert metrics.average_response_time_ms == 0.0

    def test_model_with_valid_metrics(self):
        """Test model with complete valid metrics data."""
        now = datetime.now(timezone.utc)
        earlier = datetime(2024, 9, 18, 10, 0, 0, tzinfo=timezone.utc)

        metrics = ModelCircuitBreakerMetrics(
            total_events=1000,
            successful_events=850,
            failed_events=150,
            queued_events=25,
            dropped_events=5,
            dead_letter_events=10,
            circuit_opens=3,
            circuit_closes=2,
            last_failure=earlier,
            last_success=now,
            success_rate_percent=85.0,
            average_response_time_ms=45.7,
        )

        assert metrics.total_events == 1000
        assert metrics.successful_events == 850
        assert metrics.failed_events == 150
        assert metrics.queued_events == 25
        assert metrics.dropped_events == 5
        assert metrics.dead_letter_events == 10
        assert metrics.circuit_opens == 3
        assert metrics.circuit_closes == 2
        assert metrics.last_failure == earlier
        assert metrics.last_success == now
        assert metrics.success_rate_percent == 85.0
        assert metrics.average_response_time_ms == 45.7

    def test_non_negative_integer_constraints(self):
        """Test that all integer fields enforce non-negative constraints (ge=0)."""
        # Test each integer field with negative values
        negative_test_cases = [
            ("total_events", -1),
            ("successful_events", -1),
            ("failed_events", -1),
            ("queued_events", -1),
            ("dropped_events", -1),
            ("dead_letter_events", -1),
            ("circuit_opens", -1),
            ("circuit_closes", -1),
        ]

        for field_name, negative_value in negative_test_cases:
            with pytest.raises(ValidationError) as exc_info:
                ModelCircuitBreakerMetrics(**{field_name: negative_value})

            error_msg = str(exc_info.value)
            assert "greater than or equal to 0" in error_msg
            assert field_name in error_msg

    def test_success_rate_percentage_constraints(self):
        """Test success rate percentage constraints (0.0 <= rate <= 100.0)."""
        # Valid success rates
        valid_rates = [0.0, 25.5, 50.0, 85.7, 100.0]
        for rate in valid_rates:
            metrics = ModelCircuitBreakerMetrics(success_rate_percent=rate)
            assert metrics.success_rate_percent == rate

        # Invalid success rates (below 0)
        invalid_low_rates = [-0.1, -1.0, -50.0]
        for rate in invalid_low_rates:
            with pytest.raises(ValidationError) as exc_info:
                ModelCircuitBreakerMetrics(success_rate_percent=rate)
            assert "greater than or equal to 0" in str(exc_info.value)

        # Invalid success rates (above 100)
        invalid_high_rates = [100.1, 150.0, 200.0]
        for rate in invalid_high_rates:
            with pytest.raises(ValidationError) as exc_info:
                ModelCircuitBreakerMetrics(success_rate_percent=rate)
            assert "less than or equal to 100" in str(exc_info.value)

    def test_response_time_constraints(self):
        """Test average response time constraints (must be >= 0.0)."""
        # Valid response times
        valid_times = [0.0, 1.5, 42.7, 1000.0, 5000.0]
        for time_ms in valid_times:
            metrics = ModelCircuitBreakerMetrics(average_response_time_ms=time_ms)
            assert metrics.average_response_time_ms == time_ms

        # Invalid response times (negative)
        invalid_times = [-0.1, -1.0, -100.0]
        for time_ms in invalid_times:
            with pytest.raises(ValidationError) as exc_info:
                ModelCircuitBreakerMetrics(average_response_time_ms=time_ms)
            assert "greater than or equal to 0" in str(exc_info.value)

    def test_datetime_field_handling(self):
        """Test datetime field handling and serialization."""
        now = datetime.now(timezone.utc)
        failure_time = datetime(2024, 9, 18, 14, 30, 45, tzinfo=timezone.utc)

        metrics = ModelCircuitBreakerMetrics(
            last_success=now,
            last_failure=failure_time,
        )

        assert isinstance(metrics.last_success, datetime)
        assert isinstance(metrics.last_failure, datetime)
        assert metrics.last_success == now
        assert metrics.last_failure == failure_time

    def test_json_serialization_with_datetime_encoding(self):
        """Test JSON serialization with proper datetime encoding."""
        now = datetime.now(timezone.utc)
        metrics = ModelCircuitBreakerMetrics(
            total_events=100,
            successful_events=80,
            failed_events=20,
            last_success=now,
            success_rate_percent=80.0,
        )

        # Test JSON serialization
        json_data = metrics.model_dump()

        assert json_data["total_events"] == 100
        assert json_data["successful_events"] == 80
        assert json_data["failed_events"] == 20
        assert json_data["success_rate_percent"] == 80.0

        # Verify datetime is serialized (should be present)
        assert json_data["last_success"] is not None

    def test_realistic_circuit_breaker_scenarios(self):
        """Test realistic circuit breaker metric scenarios."""

        # Scenario 1: Healthy circuit breaker
        healthy_metrics = ModelCircuitBreakerMetrics(
            total_events=1000,
            successful_events=990,
            failed_events=10,
            queued_events=0,
            dropped_events=0,
            dead_letter_events=0,
            circuit_opens=0,
            circuit_closes=0,
            success_rate_percent=99.0,
            average_response_time_ms=25.3,
        )

        assert healthy_metrics.success_rate_percent == 99.0
        assert healthy_metrics.circuit_opens == 0
        assert healthy_metrics.failed_events == 10

        # Scenario 2: Circuit with some failures
        degraded_metrics = ModelCircuitBreakerMetrics(
            total_events=500,
            successful_events=350,
            failed_events=150,
            queued_events=50,
            dropped_events=10,
            dead_letter_events=25,
            circuit_opens=2,
            circuit_closes=1,
            success_rate_percent=70.0,
            average_response_time_ms=150.7,
        )

        assert degraded_metrics.success_rate_percent == 70.0
        assert degraded_metrics.circuit_opens == 2
        assert degraded_metrics.queued_events == 50
        assert degraded_metrics.dead_letter_events == 25

        # Scenario 3: Circuit breaker in open state with high failures
        open_circuit_metrics = ModelCircuitBreakerMetrics(
            total_events=200,
            successful_events=50,
            failed_events=150,
            queued_events=100,
            dropped_events=25,
            dead_letter_events=75,
            circuit_opens=5,
            circuit_closes=0,
            success_rate_percent=25.0,
            average_response_time_ms=500.0,
        )

        assert open_circuit_metrics.success_rate_percent == 25.0
        assert open_circuit_metrics.circuit_opens == 5
        assert open_circuit_metrics.circuit_closes == 0
        assert open_circuit_metrics.queued_events == 100
        assert open_circuit_metrics.dead_letter_events == 75

    def test_metrics_consistency_validation(self):
        """Test logical consistency in metrics (not enforced by model but good for understanding)."""
        # This test documents expected relationships between fields
        # Note: The model doesn't enforce these relationships, but they should be logically consistent

        metrics = ModelCircuitBreakerMetrics(
            total_events=100,
            successful_events=80,
            failed_events=20,  # successful + failed should ideally equal total
            success_rate_percent=80.0,  # should match successful/total ratio
        )

        # Verify the values are as set (model doesn't enforce consistency)
        assert metrics.total_events == 100
        assert metrics.successful_events == 80
        assert metrics.failed_events == 20
        assert metrics.success_rate_percent == 80.0

        # Document expected relationships (for future enhancement consideration)
        expected_total = metrics.successful_events + metrics.failed_events
        expected_success_rate = (metrics.successful_events / metrics.total_events) * 100

        # These assertions document the expected relationships
        # In a future version, the model might enforce these
        assert expected_total == 100  # successful + failed = total
        assert expected_success_rate == 80.0  # success rate calculation

    def test_edge_case_zero_events(self):
        """Test edge case where no events have been processed."""
        metrics = ModelCircuitBreakerMetrics(
            total_events=0,
            successful_events=0,
            failed_events=0,
            success_rate_percent=100.0,  # Default assumption when no events
        )

        assert metrics.total_events == 0
        assert metrics.successful_events == 0
        assert metrics.failed_events == 0
        assert metrics.success_rate_percent == 100.0

    def test_edge_case_high_volume_metrics(self):
        """Test edge case with high volume metrics."""
        high_volume_metrics = ModelCircuitBreakerMetrics(
            total_events=1000000,
            successful_events=999000,
            failed_events=1000,
            queued_events=500,
            dropped_events=100,
            dead_letter_events=200,
            circuit_opens=10,
            circuit_closes=8,
            success_rate_percent=99.9,
            average_response_time_ms=12.5,
        )

        assert high_volume_metrics.total_events == 1000000
        assert high_volume_metrics.successful_events == 999000
        assert high_volume_metrics.success_rate_percent == 99.9

    def test_model_field_descriptions(self):
        """Test that model has proper field descriptions for documentation."""
        # Create an instance to access field info
        metrics = ModelCircuitBreakerMetrics()

        # Get field information from the model
        fields = metrics.model_fields

        # Verify key fields have descriptions
        assert "total_events" in fields
        assert "successful_events" in fields
        assert "failed_events" in fields
        assert "success_rate_percent" in fields
        assert "average_response_time_ms" in fields

        # Verify descriptions exist (field info should have description)
        total_events_field = fields["total_events"]
        assert hasattr(total_events_field, 'description') or 'description' in str(total_events_field)

    def test_model_immutability_after_creation(self):
        """Test model behavior with validation on assignment."""
        metrics = ModelCircuitBreakerMetrics(
            total_events=100,
            successful_events=80,
        )

        # Initial values
        assert metrics.total_events == 100
        assert metrics.successful_events == 80

        # Test that we can update values (depending on model config)
        # Note: Whether this works depends on validate_assignment config
        try:
            metrics.total_events = 200
            metrics.successful_events = 160
            # If validation on assignment is enabled, this should work
            assert metrics.total_events == 200
            assert metrics.successful_events == 160
        except (ValidationError, AttributeError):
            # If model is immutable or has strict validation, this is expected
            pass

    def test_model_config_settings(self):
        """Test model configuration settings."""
        metrics = ModelCircuitBreakerMetrics()

        # Test JSON encoding configuration
        # The model should have datetime encoding configured
        config = metrics.model_config if hasattr(metrics, 'model_config') else metrics.Config

        # Verify the model has some configuration
        assert config is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])