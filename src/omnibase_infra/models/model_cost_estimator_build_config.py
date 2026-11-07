# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build recipe declared by the estimator delivery effect contract."""

from pydantic import BaseModel, ConfigDict, Field


class ModelCostEstimatorBuildConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(min_length=1)
    repository_url: str = Field(min_length=1)
    branch: str
    image: str
    clone_timeout_seconds: int = Field(gt=0)
    read_timeout_seconds: int = Field(gt=0)
    build_timeout_seconds: int = Field(gt=0)


class ModelCostEstimatorWebFollow(BaseModel):
    """The web source the dev deploy agent's idle follower tracks (OMN-22808).

    Built from the build declaration: the repository, url and branch are the
    build's own settings; ``container`` is the contract's ``follow`` block.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(min_length=1)
    repository_url: str = Field(min_length=1)
    branch: str = Field(min_length=1)
    container: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    revision_label: str = Field(min_length=1)
    read_timeout_seconds: int = Field(gt=0)
