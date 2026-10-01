# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Match a consumer group to the contract identity a dispatch needs (OMN-20235).

A topic suffix also admits per-run groups and other nodes consuming that
topic. Ownership follows the contract's package and node name, independent
of the environment and deployed version.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.event_bus.util_consumer_group import (
    INSTANCE_SCOPE_INFIX,
    TOPIC_SCOPE_INFIX,
    normalize_kafka_identifier,
)
from omnibase_infra.enums.enum_consumer_group_purpose import EnumConsumerGroupPurpose


class ModelConsumerGroupOwner(BaseModel):
    """Contract identity whose consuming groups may answer a liveness question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service: str = Field(
        min_length=1, description="Distribution shipping the contract."
    )
    node: str = Field(min_length=1, description="Node name declared by the contract.")

    def matches(self, group_id: str) -> bool:
        """Match the base identity, ignoring environment, version and scopes."""
        base = group_id.split(INSTANCE_SCOPE_INFIX, 1)[0].split(TOPIC_SCOPE_INFIX, 1)[0]
        marker = (
            f".{normalize_kafka_identifier(self.service)}"
            f".{normalize_kafka_identifier(self.node)}"
            f".{EnumConsumerGroupPurpose.CONSUME.value}."
        )
        return marker in "." + base
