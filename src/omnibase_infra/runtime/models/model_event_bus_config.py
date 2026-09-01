# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event Bus Configuration Model.

The Pydantic model for event bus configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_infra.runtime.models.enum_event_bus_profile import EnumEventBusProfile


class ModelEventBusConfig(BaseModel):
    """Event bus configuration model.

    Defines the event bus type and operational parameters.

    Attributes:
        type: Event bus implementation type (EnumEventBusType enum)
        profile: Validation profile (OMN-17304). ``lane`` (the default)
            accepts only production-safe transports; ``local`` also accepts
            the in-memory bus, which is a first-class configured value for
            local runtimes and the shipped tier-0 default.
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
    profile: EnumEventBusProfile = Field(
        default=EnumEventBusProfile.LANE,
        description=(
            "Validation profile for the declared transport (OMN-17304): "
            "'lane' rejects non-production-safe transports (fail-closed "
            "default); 'local' accepts every supported transport, including "
            "the in-memory bus."
        ),
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

    @model_validator(mode="after")
    def validate_production_safe(self) -> ModelEventBusConfig:
        """Enforce the profile axis on the declared transport (OMN-17304).

        Lane-profile runtimes (the default — every config that predates the
        axis) reject non-production-safe transports exactly as the previous
        blanket rule did: an in-memory bus in a deployed lane silently strands
        evidence outside the shared projections. Local-profile runtimes accept
        every supported transport — ``inmemory`` is a first-class configured
        value there, and the shipped tier-0 default configuration declares it.
        """
        if (
            self.profile is EnumEventBusProfile.LANE
            and not self.type.is_production_safe
        ):
            msg = (
                f"Event bus type '{self.type.value}' is not production-safe and "
                f"the event_bus profile is '{EnumEventBusProfile.LANE.value}' "
                f"(the fail-closed default). Lane runtimes must use "
                f"'{EnumEventBusType.KAFKA.value}' or "
                f"'{EnumEventBusType.CLOUD.value}'. A local runtime that "
                f"legitimately wants '{self.type.value}' must declare "
                f"event_bus.profile: '{EnumEventBusProfile.LOCAL.value}' "
                f"(OMN-17304)."
            )
            raise ValueError(msg)
        return self


__all__: list[str] = ["ModelEventBusConfig"]
