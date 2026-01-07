# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for Service Discovery Effect Node.

This module exports models for the service discovery effect node:

Enums:
    EnumHealthStatus: Health status values (HEALTHY, UNHEALTHY, UNKNOWN).
    EnumServiceDiscoveryOperation: Operation types (REGISTER, DEREGISTER).

Models:
    ModelHealthCheckConfig: Health check configuration for registration.
    ModelServiceRegistration: Input model for service registration.
    ModelServiceInfo: Information about a discovered service.
    ModelDiscoveryQuery: Query parameters for service discovery.
    ModelDiscoveryResult: Result of service discovery query.
    ModelQueryMetadata: Metadata about query execution.
    ModelRegistrationResult: Result of registration/deregistration.
"""

from .enum_health_status import (
    EnumHealthStatus,
)
from .enum_service_discovery_operation import (
    EnumServiceDiscoveryOperation,
)
from .model_discovery_query import (
    ModelDiscoveryQuery,
)
from .model_discovery_result import (
    ModelDiscoveryResult,
)
from .model_health_check_config import (
    ModelHealthCheckConfig,
)
from .model_query_metadata import (
    ModelQueryMetadata,
)
from .model_registration_result import (
    ModelRegistrationResult,
)
from .model_service_discovery_health_check_details import (
    ModelServiceDiscoveryHealthCheckDetails,
)
from .model_service_discovery_health_check_result import (
    ModelServiceDiscoveryHealthCheckResult,
)
from .model_service_info import (
    ModelServiceInfo,
)
from .model_service_registration import (
    ModelServiceRegistration,
)

__all__ = [
    "EnumHealthStatus",
    "EnumServiceDiscoveryOperation",
    "ModelDiscoveryQuery",
    "ModelDiscoveryResult",
    "ModelHealthCheckConfig",
    "ModelQueryMetadata",
    "ModelRegistrationResult",
    "ModelServiceDiscoveryHealthCheckDetails",
    "ModelServiceDiscoveryHealthCheckResult",
    "ModelServiceInfo",
    "ModelServiceRegistration",
]
