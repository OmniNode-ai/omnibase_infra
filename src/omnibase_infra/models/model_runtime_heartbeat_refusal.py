# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A typed refusal to start a runtime that cannot heartbeat (OMN-24496)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_runtime_heartbeat_refusal_reason import (
    EnumRuntimeHeartbeatRefusalReason,
)


class ModelRuntimeHeartbeatRefusal(BaseModel):
    """One refusal to start a runtime whose heartbeat could not run.

    Every runtime heartbeats: when a runtime cannot build its introspection
    service, has an empty name, or cannot start its heartbeat task, it refuses
    to start and carries one of these refusals on the typed error.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_label: str = Field(..., description="Label of the runtime that was refused")
    reason: EnumRuntimeHeartbeatRefusalReason = Field(
        ..., description="Why the runtime refused to start"
    )
    detail: str = Field(
        ..., description="Human-readable failure detail behind the refusal"
    )
    correlation_id: UUID = Field(
        ..., description="Correlation ID of the refused startup attempt"
    )
    count: int = Field(
        ...,
        ge=1,
        description=(
            "How many refusals of this reason this process has made; the "
            "counter value after this one"
        ),
    )


__all__ = ["ModelRuntimeHeartbeatRefusal"]
