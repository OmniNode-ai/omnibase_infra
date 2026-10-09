# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One repository in the accepted build's provenance tuple."""

from pydantic import BaseModel, ConfigDict, Field


class ModelLabSourceRef(BaseModel):
    """A fully resolved ref; abbreviated commits cannot identify desired state."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    repo: str = Field(pattern=r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$")
    ref: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
