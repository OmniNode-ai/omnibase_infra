# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The running container, filtered to declared env keys."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_runner_inspect_mount import (
    ModelRunnerInspectMount,
)


class ModelRunnerRunningConfig(BaseModel):
    """The running container, already filtered to the declared env keys on the host."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    mounts: tuple[ModelRunnerInspectMount, ...] = ()
    tmpfs: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
