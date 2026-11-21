# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to close one event-loop reading window (OMN-26249)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeLoopWindowRequest(BaseModel):
    """What the member supplies when a reading window closes.

    The sampler knows the loop gap and the tick lag; the member knows its own
    identity and how many dispatches its event bus abandoned.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Identifies the reading produced.")
    member_id: str = Field(..., min_length=1, description="Identity of the member.")
    host: str = Field(..., min_length=1, description="Host the member runs on.")
    window_end: datetime = Field(..., description="When the window closes (UTC).")
    abandoned_dispatches: int = Field(
        ...,
        ge=0,
        description="Dispatches abandoned at their deadline that have not returned.",
    )


__all__: list[str] = ["ModelRuntimeLoopWindowRequest"]
