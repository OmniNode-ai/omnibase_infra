# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Event Bus Configuration Model.

The Pydantic model for event bus configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType


class ModelEventBusConfig(BaseModel):
    """Event bus configuration model.

    Defines the event bus type and operational parameters.

    Attributes:
        type: Event bus implementation type (EnumEventBusType enum)
        environment: Deployment environment name
        max_history: Maximum event history to retain
        circuit_breaker_threshold: Failure count before circuit breaker trips
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    type: EnumEventBusType = Field(
        default=EnumEventBusType.KAFKA,
        description="Event bus implementation type",
    )
    environment: str = Field(
        default="local",
        description="Deployment environment name",
    )
    max_history: int = Field(
        default=1000,
        ge=0,
        description="Maximum event history to retain",
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Failure count before circuit breaker trips",
    )

    @field_validator("type")
    @classmethod
    def validate_production_safe(cls, v: EnumEventBusType) -> EnumEventBusType:
        """Reject non-production-safe event bus types.

        INMEMORY is forbidden in omnibase_infra. Only the omniclaude skill
        runner may use inmemory via a separate mechanism.
        """
        if not v.is_production_safe:
            msg = (
                f"Event bus type '{v.value}' is not production-safe. "
                f"Use '{EnumEventBusType.KAFKA.value}' or "
                f"'{EnumEventBusType.CLOUD.value}' instead."
            )
            raise ValueError(msg)
        return v


__all__: list[str] = ["ModelEventBusConfig"]
