"""System health details model implementing ProtocolHealthDetails."""

from typing import TYPE_CHECKING

from omnibase_core.models.model_base import ModelBase
from pydantic import Field

if TYPE_CHECKING:
    from omnibase_spi.protocols.types.core_types import HealthStatus

from omnibase_infra.enums import EnumHealthStatus


class ModelSystemHealthDetails(ModelBase):
    """System-level health details with self-assessment capability."""

    # Performance indicators
    peak_memory_usage_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="Peak memory usage in megabytes",
    )

    average_cpu_usage_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Average CPU usage percentage",
    )

    disk_space_available_gb: float | None = Field(
        default=None,
        ge=0.0,
        description="Available disk space in gigabytes",
    )

    disk_space_total_gb: float | None = Field(
        default=None,
        ge=0.0,
        description="Total disk space in gigabytes",
    )

    # Network and connectivity
    network_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Average network latency in milliseconds",
    )

    external_service_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of external services being monitored",
    )

    external_services_healthy: int | None = Field(
        default=None,
        ge=0,
        description="Number of external services reporting healthy",
    )

    # Configuration and environment
    environment_variables_loaded: int | None = Field(
        default=None,
        ge=0,
        description="Number of environment variables loaded",
    )

    configuration_files_loaded: int | None = Field(
        default=None,
        ge=0,
        description="Number of configuration files loaded",
    )

    def get_health_status(self) -> "HealthStatus":
        """Assess system health status based on performance metrics."""
        # Critical conditions
        if self.average_cpu_usage_percent and self.average_cpu_usage_percent > 95:
            return EnumHealthStatus.CRITICAL

        if self.disk_space_available_gb and self.disk_space_total_gb:
            disk_usage_percent = (1 - self.disk_space_available_gb / self.disk_space_total_gb) * 100
            if disk_usage_percent > 95:
                return EnumHealthStatus.CRITICAL

        # Warning conditions
        if self.average_cpu_usage_percent and self.average_cpu_usage_percent > 80:
            return EnumHealthStatus.WARNING

        if self.disk_space_available_gb and self.disk_space_total_gb:
            disk_usage_percent = (1 - self.disk_space_available_gb / self.disk_space_total_gb) * 100
            if disk_usage_percent > 85:
                return EnumHealthStatus.WARNING

        if (self.external_service_count and self.external_services_healthy and
            self.external_services_healthy < self.external_service_count):
            unhealthy_services = self.external_service_count - self.external_services_healthy
            if unhealthy_services > self.external_service_count * 0.3:  # More than 30% unhealthy
                return EnumHealthStatus.WARNING

        # Performance degradation
        if self.network_latency_ms and self.network_latency_ms > 1000:  # 1+ second latency
            return EnumHealthStatus.DEGRADED

        return EnumHealthStatus.HEALTHY

    def is_healthy(self) -> bool:
        """Return True if system is considered healthy."""
        return self.get_health_status() == EnumHealthStatus.HEALTHY

    def get_health_summary(self) -> str:
        """Generate human-readable system health summary."""
        status = self.get_health_status()

        if status == EnumHealthStatus.CRITICAL:
            if self.average_cpu_usage_percent and self.average_cpu_usage_percent > 95:
                return f"System Critical: CPU usage at {self.average_cpu_usage_percent:.1f}%"
            if self.disk_space_available_gb and self.disk_space_total_gb:
                disk_usage_percent = (1 - self.disk_space_available_gb / self.disk_space_total_gb) * 100
                if disk_usage_percent > 95:
                    return f"System Critical: Disk usage at {disk_usage_percent:.1f}%"

        if status == EnumHealthStatus.WARNING:
            warnings = []
            if self.average_cpu_usage_percent and self.average_cpu_usage_percent > 80:
                warnings.append(f"CPU {self.average_cpu_usage_percent:.1f}%")
            if self.disk_space_available_gb and self.disk_space_total_gb:
                disk_usage_percent = (1 - self.disk_space_available_gb / self.disk_space_total_gb) * 100
                if disk_usage_percent > 85:
                    warnings.append(f"Disk {disk_usage_percent:.1f}%")
            if (self.external_service_count and self.external_services_healthy and
                self.external_services_healthy < self.external_service_count):
                warnings.append(f"External services {self.external_services_healthy}/{self.external_service_count}")

            if warnings:
                return f"System Warning: {', '.join(warnings)}"

        if status == EnumHealthStatus.DEGRADED:
            return f"System Degraded: Network latency {self.network_latency_ms:.0f}ms"

        # Healthy status
        status_parts = []
        if self.average_cpu_usage_percent is not None:
            status_parts.append(f"CPU {self.average_cpu_usage_percent:.1f}%")
        if self.disk_space_available_gb and self.disk_space_total_gb:
            disk_usage_percent = (1 - self.disk_space_available_gb / self.disk_space_total_gb) * 100
            status_parts.append(f"Disk {disk_usage_percent:.1f}%")
        if self.external_service_count and self.external_services_healthy:
            status_parts.append(f"Services {self.external_services_healthy}/{self.external_service_count}")

        if status_parts:
            return f"System Healthy: {', '.join(status_parts)}"

        return "System performance healthy"