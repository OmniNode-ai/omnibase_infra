# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer restart command model for Layer 1 health pipeline.

Published to the consumer restart command topic by the triage node
when graduated response escalates to automated restart.

Related Tickets:
    - OMN-5511: Create consumer health event models and enums
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelConsumerRestartCommand(BaseModel):
    """Command to restart a specific consumer instance.

    Published to the consumer restart command topic by the triage node
    when graduated response escalates to automated restart.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    command_id: UUID = Field(
        default_factory=uuid4, description="Unique command identifier."
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID linking to the triggering health event.",
    )
    consumer_identity: str = Field(..., description="Target consumer to restart.")
    consumer_group: str = Field(
        ..., description="Consumer group of the target consumer."
    )
    topic: str = Field(..., description="Topic the consumer is subscribed to.")
    reason: str = Field(
        ..., description="Human-readable reason for the restart command."
    )
    fingerprint: str = Field(
        ..., description="Fingerprint of the incident triggering the restart."
    )
    intent_type: Literal["consumer_health.restart"] = Field(
        default="consumer_health.restart",
        description="Intent type for routing.",
    )
    issued_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the command was issued.",
    )


__all__ = ["ModelConsumerRestartCommand"]
