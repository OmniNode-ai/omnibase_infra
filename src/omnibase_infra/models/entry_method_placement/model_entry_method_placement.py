# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Output model for the entry-method placement decision (OMN-24492)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_entry_method_placement import (
    EnumEntryMethodPlacement,
)


class ModelEntryMethodPlacement(BaseModel):
    """The decided placement for one handler entry method."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    placement: EnumEntryMethodPlacement = Field(
        ...,
        description=(
            "ON_LOOP for an ``async def`` entry method; WORKER_THREAD for a "
            "plain ``def`` one that would otherwise freeze the loop."
        ),
    )


__all__ = ["ModelEntryMethodPlacement"]
