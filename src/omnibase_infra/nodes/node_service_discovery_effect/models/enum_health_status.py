# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Health Status Enum for Service Discovery.

This module provides EnumHealthStatus, representing the health status
of a service instance in service discovery operations.

Architecture:
    EnumHealthStatus defines the possible health states for services:
    - HEALTHY: Service is responding and passing health checks
    - UNHEALTHY: Service is failing health checks
    - UNKNOWN: Health status cannot be determined

    This enum is backend-agnostic and represents a normalized view
    of health status across Consul, Kubernetes, Etcd, etc.

Related:
    - ModelServiceInfo: Uses this enum for health_status field
    - ModelDiscoveryQuery: Uses this enum for health_filter field
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from enum import Enum


class EnumHealthStatus(str, Enum):
    """Health status of a service instance.

    Represents the possible health states for services discovered
    through service discovery backends.

    Values:
        HEALTHY: Service is responding and passing health checks.
        UNHEALTHY: Service is failing health checks.
        UNKNOWN: Health status cannot be determined.

    Backend Mappings:
        - Consul: passing -> HEALTHY, critical -> UNHEALTHY, warning -> UNHEALTHY
        - Kubernetes: Ready=True -> HEALTHY, Ready=False -> UNHEALTHY
        - Etcd: Based on TTL and key presence

    Example:
        >>> status = EnumHealthStatus.HEALTHY
        >>> status.value
        'healthy'
        >>> status.is_healthy
        True
    """

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

    @property
    def is_healthy(self) -> bool:
        """Return True if the status indicates a healthy service."""
        return self == EnumHealthStatus.HEALTHY

    @property
    def is_unhealthy(self) -> bool:
        """Return True if the status indicates an unhealthy service."""
        return self == EnumHealthStatus.UNHEALTHY

    @property
    def is_unknown(self) -> bool:
        """Return True if the health status is unknown."""
        return self == EnumHealthStatus.UNKNOWN


__all__ = ["EnumHealthStatus"]
