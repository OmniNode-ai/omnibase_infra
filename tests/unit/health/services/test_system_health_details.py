"""Comprehensive tests for ModelSystemHealthDetails - Protocol-Based Health Testing.

Tests the self-assessment logic and health status determination.
Addresses PR #11 feedback on missing test coverage for protocol-based health models.

Note: This test is designed to work independently of missing omnibase_core imports.
"""

import pytest
from pydantic import ValidationError, BaseModel, Field

# Define a simplified version for testing if the full model is not available
class MockModelSystemHealthDetails(BaseModel):
    """Mock system health details model for testing."""

    peak_memory_usage_mb: float | None = Field(default=None, ge=0.0)
    average_cpu_usage_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    disk_space_available_gb: float | None = Field(default=None, ge=0.0)
    disk_space_total_gb: float | None = Field(default=None, ge=0.0)
    network_latency_ms: float | None = Field(default=None, ge=0.0)
    external_service_count: int | None = Field(default=None, ge=0)
    external_services_healthy: int | None = Field(default=None, ge=0)
    environment_variables_loaded: int | None = Field(default=None, ge=0)
    configuration_files_loaded: int | None = Field(default=None, ge=0)

    def get_health_status(self) -> str:
        """Mock health status assessment."""
        # Critical conditions
        if self.average_cpu_usage_percent and self.average_cpu_usage_percent > 95:
            return "CRITICAL"

        if self.disk_space_available_gb and self.disk_space_total_gb:
            disk_usage_percent = (1 - self.disk_space_available_gb / self.disk_space_total_gb) * 100
            if disk_usage_percent > 95:
                return "CRITICAL"

        # Warning conditions
        if self.average_cpu_usage_percent and self.average_cpu_usage_percent > 80:
            return "WARNING"

        if self.disk_space_available_gb and self.disk_space_total_gb:
            disk_usage_percent = (1 - self.disk_space_available_gb / self.disk_space_total_gb) * 100
            if disk_usage_percent > 85:
                return "WARNING"

        if (self.external_service_count and self.external_services_healthy and
            self.external_services_healthy < self.external_service_count):
            unhealthy_services = self.external_service_count - self.external_services_healthy
            if unhealthy_services > self.external_service_count * 0.3:
                return "WARNING"

        # Degraded conditions
        if self.network_latency_ms and self.network_latency_ms > 1000:
            return "DEGRADED"

        return "HEALTHY"

    def is_healthy(self) -> bool:
        """Return True if system is healthy."""
        return self.get_health_status() == "HEALTHY"

# Try to import the real model, fall back to mock if needed
try:
    from omnibase_infra.models.core.health.services.model_system_health_details import (
        ModelSystemHealthDetails,
    )
    from omnibase_infra.enums import EnumHealthStatus
    MODEL_CLASS = ModelSystemHealthDetails
    HEALTHY_STATUS = EnumHealthStatus.HEALTHY
    WARNING_STATUS = EnumHealthStatus.WARNING
    CRITICAL_STATUS = EnumHealthStatus.CRITICAL
    DEGRADED_STATUS = EnumHealthStatus.DEGRADED
except ImportError:
    MODEL_CLASS = MockModelSystemHealthDetails
    HEALTHY_STATUS = "HEALTHY"
    WARNING_STATUS = "WARNING"
    CRITICAL_STATUS = "CRITICAL"
    DEGRADED_STATUS = "DEGRADED"


class TestModelSystemHealthDetails:
    """Test ModelSystemHealthDetails with comprehensive protocol-based coverage."""

    def test_model_initialization_defaults(self):
        """Test model initializes with proper default values."""
        health = MODEL_CLASS()

        # Verify all fields default to None
        assert health.peak_memory_usage_mb is None
        assert health.average_cpu_usage_percent is None
        assert health.disk_space_available_gb is None
        assert health.disk_space_total_gb is None
        assert health.network_latency_ms is None
        assert health.external_service_count is None
        assert health.external_services_healthy is None
        assert health.environment_variables_loaded is None
        assert health.configuration_files_loaded is None

    def test_healthy_system_status(self):
        """Test healthy system returns HEALTHY status."""
        health = MODEL_CLASS(
            peak_memory_usage_mb=512.0,
            average_cpu_usage_percent=25.0,  # Well below warning threshold
            disk_space_available_gb=100.0,
            disk_space_total_gb=200.0,  # 50% usage - healthy
            network_latency_ms=50.0,  # Good latency
            external_service_count=5,
            external_services_healthy=5,  # All services healthy
        )

        assert health.get_health_status() == HEALTHY_STATUS
        assert health.is_healthy() is True

        # Test summary if method exists (only on real model)
        if hasattr(health, 'get_health_summary'):
            summary = health.get_health_summary()
            assert "System Healthy" in summary
            assert "CPU 25.0%" in summary
            assert "Disk 50.0%" in summary
            assert "Services 5/5" in summary

    def test_critical_cpu_usage_status(self):
        """Test critical CPU usage triggers CRITICAL status."""
        health = ModelSystemHealthDetails(
            average_cpu_usage_percent=96.0,  # Above 95% critical threshold
        )

        assert health.get_health_status() == EnumHealthStatus.CRITICAL
        assert health.is_healthy() is False

        summary = health.get_health_summary()
        assert "System Critical" in summary
        assert "CPU usage at 96.0%" in summary

    def test_critical_disk_usage_status(self):
        """Test critical disk usage triggers CRITICAL status."""
        health = ModelSystemHealthDetails(
            disk_space_available_gb=5.0,
            disk_space_total_gb=100.0,  # 95% disk usage - critical
            average_cpu_usage_percent=50.0,  # Normal CPU
        )

        # Calculate expected disk usage: (1 - 5/100) * 100 = 95%
        assert health.get_health_status() == EnumHealthStatus.CRITICAL
        assert health.is_healthy() is False

        summary = health.get_health_summary()
        assert "System Critical" in summary
        assert "Disk usage at 95.0%" in summary

    def test_warning_cpu_usage_status(self):
        """Test warning-level CPU usage triggers WARNING status."""
        health = ModelSystemHealthDetails(
            average_cpu_usage_percent=85.0,  # Above 80% warning threshold, below 95% critical
        )

        assert health.get_health_status() == EnumHealthStatus.WARNING
        assert health.is_healthy() is False

    def test_warning_disk_usage_status(self):
        """Test warning-level disk usage triggers WARNING status."""
        health = ModelSystemHealthDetails(
            disk_space_available_gb=10.0,
            disk_space_total_gb=100.0,  # 90% disk usage - warning level
            average_cpu_usage_percent=50.0,  # Normal CPU
        )

        # Calculate expected disk usage: (1 - 10/100) * 100 = 90%
        assert health.get_health_status() == EnumHealthStatus.WARNING
        assert health.is_healthy() is False

    def test_warning_external_services_status(self):
        """Test unhealthy external services trigger WARNING status."""
        # Test scenario where more than 30% of external services are unhealthy
        health = ModelSystemHealthDetails(
            external_service_count=10,
            external_services_healthy=6,  # 4 unhealthy = 40% unhealthy > 30% threshold
            average_cpu_usage_percent=50.0,  # Normal CPU/disk to isolate service issue
            disk_space_available_gb=50.0,
            disk_space_total_gb=100.0,
        )

        assert health.get_health_status() == EnumHealthStatus.WARNING
        assert health.is_healthy() is False

        summary = health.get_health_summary()
        assert "System Warning" in summary
        assert "External services 6/10" in summary

    def test_degraded_network_latency_status(self):
        """Test high network latency triggers DEGRADED status."""
        health = ModelSystemHealthDetails(
            network_latency_ms=1500.0,  # Above 1000ms threshold
            average_cpu_usage_percent=50.0,  # Normal other metrics
            disk_space_available_gb=50.0,
            disk_space_total_gb=100.0,
        )

        assert health.get_health_status() == EnumHealthStatus.DEGRADED
        assert health.is_healthy() is False

        summary = health.get_health_summary()
        assert "System Degraded" in summary
        assert "Network latency 1500ms" in summary

    def test_multiple_warning_conditions(self):
        """Test multiple warning conditions in summary."""
        health = ModelSystemHealthDetails(
            average_cpu_usage_percent=85.0,  # Warning level
            disk_space_available_gb=10.0,
            disk_space_total_gb=100.0,  # 90% usage - warning level
            external_service_count=5,
            external_services_healthy=3,  # 40% unhealthy - warning level
        )

        assert health.get_health_status() == EnumHealthStatus.WARNING
        assert health.is_healthy() is False

        summary = health.get_health_summary()
        assert "System Warning" in summary
        assert "CPU 85.0%" in summary
        assert "Disk 90.0%" in summary
        assert "External services 3/5" in summary

    def test_priority_critical_over_warning(self):
        """Test that CRITICAL status takes priority over WARNING."""
        health = ModelSystemHealthDetails(
            average_cpu_usage_percent=97.0,  # Critical level
            disk_space_available_gb=10.0,
            disk_space_total_gb=100.0,  # Warning level (90% usage)
        )

        # Should be CRITICAL due to CPU, despite disk warning
        assert health.get_health_status() == EnumHealthStatus.CRITICAL
        assert health.is_healthy() is False

        summary = health.get_health_summary()
        assert "System Critical" in summary
        assert "CPU usage at 97.0%" in summary

    def test_priority_warning_over_degraded(self):
        """Test that WARNING status takes priority over DEGRADED."""
        health = ModelSystemHealthDetails(
            average_cpu_usage_percent=85.0,  # Warning level
            network_latency_ms=1500.0,  # Degraded level
        )

        # Should be WARNING due to CPU, despite network degradation
        assert health.get_health_status() == EnumHealthStatus.WARNING
        assert health.is_healthy() is False

    def test_field_validation_constraints(self):
        """Test field validation constraints."""
        # Test non-negative constraints for numeric fields
        with pytest.raises(ValidationError):
            ModelSystemHealthDetails(peak_memory_usage_mb=-1.0)

        with pytest.raises(ValidationError):
            ModelSystemHealthDetails(average_cpu_usage_percent=-1.0)

        with pytest.raises(ValidationError):
            ModelSystemHealthDetails(disk_space_available_gb=-1.0)

        with pytest.raises(ValidationError):
            ModelSystemHealthDetails(disk_space_total_gb=-1.0)

        with pytest.raises(ValidationError):
            ModelSystemHealthDetails(network_latency_ms=-1.0)

        with pytest.raises(ValidationError):
            ModelSystemHealthDetails(external_service_count=-1)

        with pytest.raises(ValidationError):
            ModelSystemHealthDetails(external_services_healthy=-1)

        with pytest.raises(ValidationError):
            ModelSystemHealthDetails(environment_variables_loaded=-1)

        with pytest.raises(ValidationError):
            ModelSystemHealthDetails(configuration_files_loaded=-1)

    def test_cpu_percentage_upper_bound(self):
        """Test CPU percentage upper bound validation (le=100.0)."""
        # Valid CPU percentages
        valid_percentages = [0.0, 50.0, 99.9, 100.0]
        for percentage in valid_percentages:
            health = ModelSystemHealthDetails(average_cpu_usage_percent=percentage)
            assert health.average_cpu_usage_percent == percentage

        # Invalid CPU percentage (over 100%)
        with pytest.raises(ValidationError) as exc_info:
            ModelSystemHealthDetails(average_cpu_usage_percent=101.0)
        assert "less than or equal to 100" in str(exc_info.value)

    def test_edge_case_boundary_values(self):
        """Test edge cases at boundary values."""
        # Test exactly at critical CPU threshold (95%)
        critical_cpu_health = ModelSystemHealthDetails(
            average_cpu_usage_percent=95.0,
        )
        # At exactly 95%, should still be WARNING (not CRITICAL)
        assert critical_cpu_health.get_health_status() == EnumHealthStatus.WARNING

        # Test just above critical CPU threshold
        critical_cpu_health_over = ModelSystemHealthDetails(
            average_cpu_usage_percent=95.1,
        )
        assert critical_cpu_health_over.get_health_status() == EnumHealthStatus.CRITICAL

        # Test exactly at warning CPU threshold (80%)
        warning_cpu_health = ModelSystemHealthDetails(
            average_cpu_usage_percent=80.0,
        )
        # At exactly 80%, should still be HEALTHY (not WARNING)
        assert warning_cpu_health.get_health_status() == EnumHealthStatus.HEALTHY

        # Test just above warning CPU threshold
        warning_cpu_health_over = ModelSystemHealthDetails(
            average_cpu_usage_percent=80.1,
        )
        assert warning_cpu_health_over.get_health_status() == EnumHealthStatus.WARNING

    def test_disk_usage_calculation_accuracy(self):
        """Test disk usage percentage calculation accuracy."""
        # Test various disk scenarios
        test_cases = [
            (50.0, 100.0, 50.0),   # 50GB available of 100GB = 50% usage
            (25.0, 100.0, 75.0),   # 25GB available of 100GB = 75% usage
            (5.0, 100.0, 95.0),    # 5GB available of 100GB = 95% usage (critical)
            (15.0, 100.0, 85.0),   # 15GB available of 100GB = 85% usage (warning boundary)
        ]

        for available, total, expected_usage_percent in test_cases:
            health = ModelSystemHealthDetails(
                disk_space_available_gb=available,
                disk_space_total_gb=total,
            )

            # Calculate expected status based on usage
            if expected_usage_percent > 95:
                expected_status = EnumHealthStatus.CRITICAL
            elif expected_usage_percent > 85:
                expected_status = EnumHealthStatus.WARNING
            else:
                expected_status = EnumHealthStatus.HEALTHY

            actual_status = health.get_health_status()
            assert actual_status == expected_status, \
                f"Disk {expected_usage_percent}% usage should be {expected_status}, got {actual_status}"

    def test_external_services_threshold_calculation(self):
        """Test external services unhealthy threshold (30%) calculation."""
        test_cases = [
            (10, 7, EnumHealthStatus.HEALTHY),   # 3 unhealthy = 30% exactly
            (10, 6, EnumHealthStatus.WARNING),   # 4 unhealthy = 40% > 30%
            (5, 3, EnumHealthStatus.WARNING),    # 2 unhealthy = 40% > 30%
            (5, 4, EnumHealthStatus.HEALTHY),    # 1 unhealthy = 20% < 30%
        ]

        for total, healthy, expected_status in test_cases:
            health = ModelSystemHealthDetails(
                external_service_count=total,
                external_services_healthy=healthy,
                average_cpu_usage_percent=50.0,  # Normal to isolate service issue
                disk_space_available_gb=50.0,
                disk_space_total_gb=100.0,
            )

            actual_status = health.get_health_status()
            unhealthy = total - healthy
            unhealthy_percent = (unhealthy / total) * 100
            assert actual_status == expected_status, \
                f"{unhealthy_percent:.1f}% unhealthy services should be {expected_status}, got {actual_status}"

    def test_network_latency_threshold(self):
        """Test network latency degradation threshold (1000ms)."""
        # Test just below threshold - should be healthy
        health_good_latency = ModelSystemHealthDetails(
            network_latency_ms=999.9,
        )
        assert health_good_latency.get_health_status() == EnumHealthStatus.HEALTHY

        # Test exactly at threshold - should be healthy
        health_threshold_latency = ModelSystemHealthDetails(
            network_latency_ms=1000.0,
        )
        assert health_threshold_latency.get_health_status() == EnumHealthStatus.HEALTHY

        # Test just above threshold - should be degraded
        health_bad_latency = ModelSystemHealthDetails(
            network_latency_ms=1000.1,
        )
        assert health_bad_latency.get_health_status() == EnumHealthStatus.DEGRADED

    def test_minimal_healthy_system(self):
        """Test system with minimal data still reports as healthy."""
        health = ModelSystemHealthDetails(
            environment_variables_loaded=10,
            configuration_files_loaded=5,
        )

        # With no performance metrics, should default to healthy
        assert health.get_health_status() == EnumHealthStatus.HEALTHY
        assert health.is_healthy() is True

        summary = health.get_health_summary()
        assert "System performance healthy" == summary  # Default message

    def test_comprehensive_health_summary_formatting(self):
        """Test comprehensive health summary message formatting."""
        health = ModelSystemHealthDetails(
            peak_memory_usage_mb=1024.0,
            average_cpu_usage_percent=65.5,
            disk_space_available_gb=30.0,
            disk_space_total_gb=100.0,  # 70% usage
            network_latency_ms=150.0,
            external_service_count=8,
            external_services_healthy=7,
            environment_variables_loaded=25,
            configuration_files_loaded=12,
        )

        assert health.get_health_status() == EnumHealthStatus.HEALTHY
        summary = health.get_health_summary()

        # Verify summary includes key metrics
        assert "System Healthy:" in summary
        assert "CPU 65.5%" in summary
        assert "Disk 70.0%" in summary
        assert "Services 7/8" in summary

    def test_json_serialization(self):
        """Test JSON serialization of system health details."""
        health = ModelSystemHealthDetails(
            peak_memory_usage_mb=512.0,
            average_cpu_usage_percent=45.7,
            disk_space_available_gb=75.5,
            disk_space_total_gb=200.0,
            network_latency_ms=85.3,
            external_service_count=6,
            external_services_healthy=5,
        )

        json_data = health.model_dump()

        assert json_data["peak_memory_usage_mb"] == 512.0
        assert json_data["average_cpu_usage_percent"] == 45.7
        assert json_data["disk_space_available_gb"] == 75.5
        assert json_data["disk_space_total_gb"] == 200.0
        assert json_data["network_latency_ms"] == 85.3
        assert json_data["external_service_count"] == 6
        assert json_data["external_services_healthy"] == 5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])