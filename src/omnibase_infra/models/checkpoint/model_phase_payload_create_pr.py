# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Phase payload for the create_pr phase.

Records PR URL, PR number, and head SHA after PR creation.

Ticket: OMN-2143
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelPhasePayloadCreatePr(BaseModel):
    """Payload captured after the create_pr phase completes."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    phase: Literal["create_pr"] = Field(
        default="create_pr",
        description="Discriminator field for phase payload union.",
    )
    pr_url: str = Field(
        ...,
        min_length=1,
        description="Full URL of the created pull request.",
    )
    pr_number: int = Field(
        ...,
        ge=1,
        description="Pull request number on the remote.",
    )
    head_sha: str = Field(
        ...,
        min_length=7,
        max_length=40,
        pattern=r"^[0-9a-f]+$",
        description="HEAD SHA pushed to the remote branch.",
    )


__all__: list[str] = ["ModelPhasePayloadCreatePr"]
