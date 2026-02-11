# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Composite environment snapshot for RRH validation.

Composes three sub-models into a single snapshot that flows into the
validation compute node.
"""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from omnibase_infra.models.rrh.model_rrh_repo_state import ModelRRHRepoState
from omnibase_infra.models.rrh.model_rrh_runtime_target import ModelRRHRuntimeTarget
from omnibase_infra.models.rrh.model_rrh_toolchain_versions import (
    ModelRRHToolchainVersions,
)


class ModelRRHEnvironmentData(BaseModel):
    """Complete environment snapshot for RRH validation.

    Composed from the three sub-models collected by the emit effect node.
    Flows into ``node_rrh_validate_compute`` as its primary input data.

    Attributes:
        repo_state: Git repository state.
        runtime_target: Deployment target context.
        toolchain: Build-tool versions.
        collected_at: Timezone-aware timestamp of data collection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    repo_state: ModelRRHRepoState = Field(..., description="Git repo state.")
    runtime_target: ModelRRHRuntimeTarget = Field(..., description="Deployment target.")
    toolchain: ModelRRHToolchainVersions = Field(..., description="Tool versions.")
    collected_at: AwareDatetime = Field(
        ..., description="Timezone-aware collection timestamp."
    )


__all__: list[str] = ["ModelRRHEnvironmentData"]
