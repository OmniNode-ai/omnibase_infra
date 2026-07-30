# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection version and authoritative event-offset proof."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionReplayEvidence(BaseModel):
    """Bind one projection identity/version to source and target offsets."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    projection_id: UUID
    projection_label: str = Field(..., min_length=1)
    projection_version: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    partition: int = Field(..., ge=0)
    source_offset: int = Field(..., ge=0)
    target_offset: int = Field(..., ge=0)


__all__ = ["ModelProjectionReplayEvidence"]
