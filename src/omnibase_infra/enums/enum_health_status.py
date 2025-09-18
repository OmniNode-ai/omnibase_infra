"""Health status enumeration for service health monitoring."""

from enum import Enum


class EnumHealthStatus(str, Enum):
    """Health status enumeration following ONEX health monitoring standards."""

    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"