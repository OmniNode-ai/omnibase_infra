# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul register payload model for registration reducer.

This payload implements ProtocolIntentPayload for use with ModelIntent.
It contains the same data as ModelConsulRegisterIntent but with an
`intent_type` field instead of `kind` to satisfy the protocol.

Related:
    - ModelConsulRegisterIntent: Core intent model (uses `kind` discriminator)
    - ProtocolIntentPayload: Protocol requiring `intent_type` property
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.registration.model_node_event_bus_config import (
    ModelNodeEventBusConfig,
)

# NOTE: ModelIntentPayloadBase was removed in omnibase_core 0.6.2
# Using pydantic.BaseModel directly as the base class


class ModelPayloadConsulRegister(BaseModel):
    """Payload for Consul service registration intents.

    This payload follows the ONEX intent payload pattern for use with ModelIntent.

    Attributes:
        intent_type: Discriminator literal for intent routing. Always "consul.register".
        correlation_id: Correlation ID for distributed tracing.
        service_id: Unique service identifier for Consul registration.
        service_name: Service name for Consul service catalog.
        tags: Service tags for filtering and categorization.
        health_check: Optional health check configuration.
        event_bus_config: Resolved event bus topics for registry storage.
            If None, node is not included in dynamic topic routing lookups.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["consul.register"] = Field(
        default="consul.register",
        description="Discriminator literal for intent routing.",
    )

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )

    service_id: str = Field(
        ...,
        min_length=1,
        description="Unique service identifier for Consul registration.",
    )

    service_name: str = Field(
        ...,
        min_length=1,
        description="Service name for Consul service catalog.",
    )

    tags: list[str] = Field(
        ...,
        description="Service tags for filtering and categorization.",
    )

    health_check: dict[str, str] | None = Field(
        default=None,
        description="Optional health check configuration (HTTP, Interval, Timeout).",
    )

    event_bus_config: ModelNodeEventBusConfig | None = Field(
        default=None,
        description="Resolved event bus topics for registry storage.",
    )


__all__ = [
    "ModelPayloadConsulRegister",
]
