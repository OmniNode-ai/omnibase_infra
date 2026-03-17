# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Service Heartbeat Event Model.

ModelRuntimeHeartbeatEvent for periodic service-level heartbeat broadcasts
in the ONEX platform health telemetry system.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelRuntimeHeartbeatEvent(BaseModel):
    """Event model for periodic service-level heartbeat broadcasts.

    Services publish this event periodically to indicate they are alive and
    report current health metrics. Used by monitoring infrastructure to detect
    service failures and track resource usage.

    Attributes:
        service_id: Unique identifier per service instance (e.g. "omninode-runtime-abc123").
        service_name: Human-readable service name (e.g. "omninode-runtime").
        status: Current health status of the service.
        uptime_ms: Service uptime in milliseconds (must be >= 0).
        restart_count: Number of times the service has restarted (must be >= 0).
        memory_usage_mb: Current memory usage in megabytes (must be >= 0.0).
        cpu_percent: Current CPU usage percentage (must be >= 0.0).
        version: Service version string.
        emitted_at: Timezone-aware timestamp when the event was emitted.

    Example:
        >>> from datetime import UTC, datetime
        >>> event = ModelRuntimeHeartbeatEvent(
        ...     service_id="omninode-runtime-abc123",
        ...     service_name="omninode-runtime",
        ...     status="healthy",
        ...     uptime_ms=3600000,
        ...     restart_count=0,
        ...     memory_usage_mb=256.5,
        ...     cpu_percent=12.3,
        ...     version="1.2.3",
        ...     emitted_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Service identity
    service_id: str = Field(..., description="Unique identifier per service instance")
    service_name: str = (
        Field(  # pattern-ok: pinned to omnidash worker-health consumer contract
            ..., description="Human-readable service name"
        )
    )
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Current health status of the service"
    )

    # Health metrics
    uptime_ms: int = Field(..., ge=0, description="Service uptime in milliseconds")
    restart_count: int = Field(
        ..., ge=0, description="Number of times the service has restarted"
    )

    # Resource usage
    memory_usage_mb: float = Field(
        ..., ge=0.0, description="Current memory usage in megabytes"
    )
    cpu_percent: float = Field(..., ge=0.0, description="Current CPU usage percentage")

    # Metadata
    version: str = Field(..., description="Service version string")

    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    emitted_at: datetime = Field(
        ..., description="Timezone-aware timestamp when the event was emitted"
    )

    @field_validator("emitted_at")
    @classmethod
    def validate_emitted_at_timezone_aware(cls, v: datetime) -> datetime:
        """Validate that emitted_at is timezone-aware.

        Delegates to shared utility for consistent validation across all models.
        """
        return validate_timezone_aware_datetime(v)


__all__ = ["ModelRuntimeHeartbeatEvent"]
