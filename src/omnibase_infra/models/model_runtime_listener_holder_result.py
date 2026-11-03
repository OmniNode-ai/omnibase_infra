# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The outcome of a listener holder operation (OMN-22379)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeListenerHolderResult(BaseModel):
    """What the holder did, and which listeners it now holds."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ok: bool = Field(description="False when the operation was refused or failed.")
    outcome: str = Field(
        description="bound, recovered, remapped, dropped, inert:<why> or refused:<why>."
    )
    pools: tuple[str, ...] = Field(default=(), description="Pools now held.")
    ports: tuple[int, ...] = Field(default=(), description="Bound ports, per pool.")


__all__: list[str] = ["ModelRuntimeListenerHolderResult"]
