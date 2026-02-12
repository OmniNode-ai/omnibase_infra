# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Phase payload for the implement phase.

Records branch name, commit SHA, and files changed during implementation.

Ticket: OMN-2143
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelPhasePayloadImplement(BaseModel):
    """Payload captured after the implement phase completes."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    phase: Literal["implement"] = Field(
        default="implement",
        description="Discriminator field for phase payload union.",
    )
    branch_name: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Git branch name created for this ticket.",
    )
    commit_sha: str = Field(
        ...,
        min_length=7,
        max_length=40,
        pattern=r"^[0-9a-f]+$",
        description="HEAD commit SHA after implementation.",
    )
    files_changed: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Relative paths of files changed during implementation.",
    )


__all__: list[str] = ["ModelPhasePayloadImplement"]
