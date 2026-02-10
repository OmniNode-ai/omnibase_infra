# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul deregister payload model for registration reducer.

This payload implements ProtocolIntentPayload for use with ModelIntent.
It contains the service_id needed for Consul service deregistration.

Related:
    - ModelPayloadConsulRegister: Register counterpart
    - IntentEffectConsulDeregister: Effect adapter consuming this payload
    - ProtocolIntentPayload: Protocol requiring `intent_type` property
    - OMN-2115: Bus audit layer 1 - generic bus health diagnostics

.. versionadded:: 0.8.0
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadConsulDeregister(BaseModel):
    """Payload for Consul service deregistration intents.

    Only service_id is needed — Consul deregistration is identified by
    service_id alone (confirmed: mixin_consul_service._deregister_service()
    only reads service_id).

    Attributes:
        intent_type: Discriminator literal for intent routing. Always "consul.deregister".
        correlation_id: Correlation ID for distributed tracing.
        service_id: Unique service identifier to deregister from Consul.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["consul.deregister"] = Field(
        default="consul.deregister",
        description="Discriminator literal for intent routing.",
    )

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )

    service_id: str = Field(
        ...,
        min_length=1,
        description="Unique service identifier to deregister from Consul.",
    )


__all__ = [
    "ModelPayloadConsulDeregister",
]
