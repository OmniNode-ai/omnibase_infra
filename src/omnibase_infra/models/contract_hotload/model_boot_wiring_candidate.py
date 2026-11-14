# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One contract whose handlers may be left unwired at boot (OMN-21318)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelBootWiringCandidate"]


class ModelBootWiringCandidate(BaseModel):
    """A deferrable contract: its name and every topic it subscribes."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract_name: str = Field(..., min_length=1, description="Contract name.")
    subscribe_topics: tuple[str, ...] = Field(
        ..., min_length=1, description="Every topic the contract subscribes."
    )
