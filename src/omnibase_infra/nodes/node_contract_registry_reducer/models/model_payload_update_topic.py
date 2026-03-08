# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Update topic intent payload model.

Related:
    - OMN-1653: Contract Registry Reducer implementation
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadUpdateTopic(BaseModel):
    """Payload for PostgreSQL update topic intents.

    Used when a contract-registered event is processed to update
    the topic routing table with producer/consumer information.

    Note: contract_id is a derived natural key (node_name:major.minor.patch),
    not a UUID. This is intentional per the contract registry design.

    Attributes:
        intent_type: Routing discriminator. Always "postgres.update_topic".
        correlation_id: Correlation ID for distributed tracing.
        topic_suffix: Topic suffix (e.g., onex.evt.platform.contract-registered.v1).
        direction: Whether this contract publishes or subscribes to the topic.
        contract_id: Contract that uses this topic.
        node_name: Contract node name.
        event_type: Event type name (optional).
        last_seen_at: When topic usage was last seen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["postgres.update_topic"] = Field(
        default="postgres.update_topic",
        description="Routing discriminator for intent dispatch.",
    )

    correlation_id: UUID = Field(
        ..., description="Correlation ID for distributed tracing."
    )

    topic_suffix: str = Field(..., description="Topic suffix.")

    direction: Literal["publish", "subscribe"] = Field(
        ..., description="Topic direction."
    )

    # ONEX_EXCLUDE: pattern_validator - contract_id is a derived natural key (name:version), not UUID
    contract_id: str = Field(..., description="Contract using this topic.")

    # ONEX_EXCLUDE: pattern_validator - node_name is the contract name, not an entity reference
    node_name: str = Field(..., description="Contract node name.")

    event_type: str | None = Field(default=None, description="Event type name.")

    last_seen_at: datetime = Field(..., description="Last seen timestamp.")


__all__ = ["ModelPayloadUpdateTopic"]
