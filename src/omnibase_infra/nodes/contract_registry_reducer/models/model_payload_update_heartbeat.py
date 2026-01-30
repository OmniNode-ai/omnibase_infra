# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Update heartbeat intent payload model.

Related:
    - OMN-1653: Contract Registry Reducer implementation
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadUpdateHeartbeat(BaseModel):
    """Payload for PostgreSQL update heartbeat intents.

    Used when a node-heartbeat event is processed to update the
    last_seen_at timestamp for a contract.

    Note: contract_id is a derived natural key (node_name:major.minor.patch),
    not a UUID. This is intentional per the contract registry design.

    Attributes:
        intent_type: Routing discriminator. Always "postgres.update_heartbeat".
        correlation_id: Correlation ID for distributed tracing.
        contract_id: Contract to update.
        node_name: Contract node name.
        source_node_id: String representation of heartbeating node's UUID (optional).
        last_seen_at: New heartbeat timestamp.
        uptime_seconds: Node uptime in seconds (optional).
        sequence_number: Heartbeat sequence number (optional).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["postgres.update_heartbeat"] = Field(
        default="postgres.update_heartbeat",
        description="Routing discriminator for intent dispatch.",
    )

    correlation_id: UUID = Field(
        ..., description="Correlation ID for distributed tracing."
    )

    # ONEX_EXCLUDE: pattern_validator - contract_id is a derived natural key (name:version), not UUID
    contract_id: str = Field(..., description="Contract to update.")

    # ONEX_EXCLUDE: pattern_validator - node_name is the contract name, not an entity reference
    node_name: str = Field(..., description="Contract node name.")

    source_node_id: str | None = Field(
        default=None,
        description="Source node ID as string (UUID string representation, optional).",
    )

    last_seen_at: datetime = Field(..., description="Heartbeat timestamp.")

    uptime_seconds: float | None = Field(
        default=None, ge=0, description="Node uptime in seconds."
    )

    sequence_number: int | None = Field(
        default=None, ge=0, description="Heartbeat sequence number."
    )


__all__ = ["ModelPayloadUpdateHeartbeat"]
