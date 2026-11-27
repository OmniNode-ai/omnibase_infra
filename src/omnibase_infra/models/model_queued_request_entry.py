# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One queued rebuild request as the queue decision sees it (OMN-26718)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelQueuedRequestEntry(BaseModel):
    """The facts of one queued request that decide whether it folds and who it outranks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    runtime_lane: str = Field(min_length=1)
    requested_by: str = Field(min_length=1)
    git_ref: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    build_source: str = Field(min_length=1)
    services: tuple[str, ...] = ()
    rollback_declared: bool = False
    pinned_image: bool = False
