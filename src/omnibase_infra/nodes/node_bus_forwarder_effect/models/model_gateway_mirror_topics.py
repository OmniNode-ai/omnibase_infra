# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mirror topic config model for the gateway forwarder."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    validate_canonical_topic,
)


class ModelGatewayMirrorTopics(BaseModel):
    """Bare contract-declared topics mirrored across the gateway."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inbound: tuple[str, ...] = Field(..., min_length=1)
    outbound: tuple[str, ...] = Field(..., min_length=1)

    @field_validator("inbound", "outbound")
    @classmethod
    def _validate_topics(cls, topics: tuple[str, ...]) -> tuple[str, ...]:
        for topic in topics:
            validate_canonical_topic(topic)
        return topics
