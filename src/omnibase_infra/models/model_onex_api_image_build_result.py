# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""onex-api image build: what the build found and did (OMN-23521)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelOnexApiImageBuildResult(BaseModel):
    """The outcome of one onex-api image build.

    ``built`` is the only pass. ``image`` is the tagged reference when built and
    empty when refused. ``source_revision`` is the commit the image was built
    from, or ``None`` when that is not a full commit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    built: bool
    image: str = Field(description="The tagged image; empty when not built.")
    source_revision: str | None = Field(
        default=None, description="The commit the image was built from."
    )
    reason: str = Field(min_length=1, description="What the build found and did.")
