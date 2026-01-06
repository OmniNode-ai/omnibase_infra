# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul register payload model for registration reducer.

This payload implements ProtocolIntentPayload for use with ModelIntent.
It contains the same data as ModelConsulRegisterIntent but with an
`intent_type` field instead of `kind` to satisfy the protocol.

Related:
    - ModelConsulRegisterIntent: Core intent model (uses `kind` discriminator)
    - ProtocolIntentPayload: Protocol requiring `intent_type` property
    - OMN-1260: Fix JsonValue/JsonType and validation import migration
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from omnibase_core.models.reducer.payloads import ModelIntentPayloadBase
from pydantic import Field


class ModelPayloadConsulRegister(ModelIntentPayloadBase):
    """Payload for Consul service registration intents.

    This payload extends ModelIntentPayloadBase to satisfy ProtocolIntentPayload,
    enabling use with ModelIntent for reducer output.

    Attributes:
        intent_type: Discriminator literal for intent routing. Always "consul.register".
        correlation_id: Correlation ID for distributed tracing.
        service_id: Unique service identifier for Consul registration.
        service_name: Service name for Consul service catalog.
        tags: Service tags for filtering and categorization.
        health_check: Optional health check configuration.
    """

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


__all__ = [
    "ModelPayloadConsulRegister",
]
