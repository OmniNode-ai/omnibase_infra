# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request model for placing a rendered release block (OMN-17427)."""

from __future__ import annotations

from typing import Literal

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
    repository_visibility: Literal["public", "private", "internal"] = Field(
        ..., description="Target repository visibility from its hosting service."
    )
    hygiene_vocabulary_file: str | None = Field(
        default=None, description="Path to the public repository hygiene vocabulary."
    )
    repo_name: str | None = Field(
        default=None,
        description=(
            "Repository name used in the release-cut contract item's cwd; the "
            "clone directory name when absent."
        ),
    )
