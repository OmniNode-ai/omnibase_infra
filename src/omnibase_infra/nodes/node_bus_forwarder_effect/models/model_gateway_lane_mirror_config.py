# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane-mirror config model for the gateway forwarder (OMN-17034)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    validate_canonical_topic,
)


class ModelGatewayLaneMirrorConfig(BaseModel):
    """Contract-declared lane-to-lane mirror for the hook topic set.

    Lanes are NAMED here, never addressed: the resolved deployment YAML owns
    this deployment's per-lane broker endpoints. That is the same authority
    split ``mirror_topics`` already uses, and it is what keeps a lane rename
    or a broker move out of this contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_lane: str = Field(..., min_length=1)
    mirror_lanes: tuple[str, ...] = Field(..., min_length=1)
    topics: tuple[str, ...] = Field(..., min_length=1)
    max_messages_per_poll: int = Field(default=50, ge=1, le=500)
    poll_timeout_ms: int = Field(default=1_000, ge=1)

    @field_validator("topics")
    @classmethod
    def _validate_topics(cls, topics: tuple[str, ...]) -> tuple[str, ...]:
        for topic in topics:
            validate_canonical_topic(topic)
        return topics

    @model_validator(mode="after")
    def _validate_lane_set(self) -> ModelGatewayLaneMirrorConfig:
        if self.source_lane in self.mirror_lanes:
            raise ValueError(
                "source_lane must not appear in mirror_lanes: a lane mirroring "
                "to itself republishes every record onto the topic it just "
                "consumed, which is an unbounded loop the durable envelope "
                "marker cannot break (each republish is a new source record "
                "carrying the same already-marked envelope id, so the marker "
                "suppresses the mirror publish but never the consumption)"
            )
        if len(set(self.mirror_lanes)) != len(self.mirror_lanes):
            raise ValueError("mirror_lanes must not repeat a lane")
        if len(set(self.topics)) != len(self.topics):
            raise ValueError("lane_mirror topics must not repeat a topic")
        return self


__all__ = ["ModelGatewayLaneMirrorConfig"]
