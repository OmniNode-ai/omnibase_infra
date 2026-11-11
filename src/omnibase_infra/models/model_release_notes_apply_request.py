# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request model for placing a rendered release block (OMN-17427)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelReleaseNotesApplyRequest(BaseModel):
    """A rendered release block and the checkout it is written into."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    clone_path: str = Field(
        ..., description="Checkout of the repository being cut; files are written here."
    )
    entry_file: str = Field(
        ...,
        description="File holding the block `release_train.py changelog` rendered.",
    )
    repo_name: str | None = Field(
        default=None,
        description=(
            "Repository name used in the release-cut contract item's cwd; the "
            "clone directory name when absent."
        ),
    )
