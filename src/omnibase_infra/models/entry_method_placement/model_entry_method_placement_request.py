# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request model for the entry-method placement decision (OMN-24492)."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field


class ModelEntryMethodPlacementRequest(BaseModel):
    """Everything the placement decision depends on: the entry method itself."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    entry: Callable[..., object] = Field(
        ...,
        description=(
            "The handler's dispatch entry method (``handle`` / ``handle_async`` "
            "or the wiring-time stand-in). Must be callable."
        ),
    )


__all__ = ["ModelEntryMethodPlacementRequest"]
