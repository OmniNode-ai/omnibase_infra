# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-message replay result for the DLQ replay node (OMN-12619).

Relocated from ``scripts/dlq_replay.py:564``.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.dlq.models.enum_replay_status import EnumReplayStatus


class ModelReplayResult(BaseModel):
    """Result of a single message replay decision/attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Original message correlation ID.")
    original_topic: str = Field(..., description="Topic the message targets on replay.")
    status: EnumReplayStatus = Field(..., description="Outcome of this message.")
    message: str = Field(..., description="Human-readable explanation of the outcome.")
    replay_correlation_id: UUID | None = Field(
        default=None,
        description="New correlation ID minted for this replay/quarantine.",
    )


__all__ = ["ModelReplayResult"]
