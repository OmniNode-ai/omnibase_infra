# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""onex-api image build: what is asked of the docker build (OMN-23521)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelOnexApiImageBuildRequest(BaseModel):
    """One docker build of the onex-api image from an archived source tree."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha: str = Field(description="The merged deploy sha the job is for.")
    stamp: str = Field(description="The job stamp the image tag carries.")
    capture_sha: str = Field(
        description="The sha the captured runtime image belongs to."
    )
    runtime_image: str = Field(
        description="The captured runtime image; empty when none was captured."
    )
    capture_error: str | None = Field(
        default=None, description="Why the capture failed, when it did."
    )
    source_tree: Path = Field(description="The archived source tree the build runs in.")
    source_revision: str = Field(
        description="The 40-char commit the tree was archived from."
    )
    image_name: str = Field(min_length=1, description="The image name to tag.")
    dockerfile: str = Field(description="The Dockerfile to build from.")
    context: str = Field(description="The build context path.")
    runtime_image_build_arg: str = Field(
        description="The --build-arg name carrying the runtime image."
    )
    revision_label: str = Field(description="The label key carrying the revision.")
    source_label: str = Field(description="The label key carrying the source repo.")
    source_repo: str = Field(description="The source repository the tree came from.")
    timeout_seconds: int = Field(gt=0, description="Bound on the docker build.")
