# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Identity of one runtime container's boot streak — OMN-18550."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBootIdentity(BaseModel):
    """What a boot-failure streak is counted for, and the bound it is judged against."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    state_dir: str = Field(
        ..., description="Directory that holds the boot-failure ledger."
    )
    service: str = Field(..., description="Runtime identity the streak is counted for.")
    container_ref: str = Field(..., description="Container hostname.")
    runtime_source_hash: str = Field(
        ..., description="RUNTIME_SOURCE_HASH of the image."
    )
    max_consecutive_failures: int = Field(
        ..., ge=1, description="Declared bound on consecutive failing exits."
    )


__all__: list[str] = ["ModelBootIdentity"]
