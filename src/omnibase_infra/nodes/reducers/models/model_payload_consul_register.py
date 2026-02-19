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

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
        node_id: ONEX node identifier (string form of UUID). Required when
            event_bus_config is provided so the consul handler can store the
            event bus config under onex/nodes/{node_id}/event_bus/.
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

    node_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "ONEX node identifier (string form of UUID). Required when "
            "event_bus_config is provided so the consul handler can store "
            "the event bus config under onex/nodes/{node_id}/event_bus/."
        ),
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

    address: str | None = Field(
        default=None,
        description="Service address for Consul registration. Extracted from node endpoints.",
    )

    port: int | None = Field(
        default=None,
        description="Service port for Consul registration. Extracted from node endpoints.",
    )

    health_check: dict[str, str] | None = Field(
        default=None,
        description="Optional health check configuration (HTTP, Interval, Timeout).",
    )

    event_bus_config: ModelNodeEventBusConfig | None = Field(
        default=None,
        description="Resolved event bus topics for registry storage.",
    )

    @model_validator(mode="after")
    def validate_node_id_required_with_event_bus_config(
        self,
    ) -> ModelPayloadConsulRegister:
        """Validate that node_id is set when event_bus_config is provided.

        The Consul handler stores event bus config under
        ``onex/nodes/{node_id}/event_bus/``; without a node_id the key path
        cannot be constructed and the store operation would silently fail or
        produce an incorrect path.

        Raises:
            ValueError: If ``event_bus_config`` is not None and ``node_id``
                is None.
        """
        if self.event_bus_config is not None and self.node_id is None:
            raise ValueError(
                "event_bus_config requires node_id to be set"
            )
        return self


__all__ = [
    "ModelPayloadConsulRegister",
]
