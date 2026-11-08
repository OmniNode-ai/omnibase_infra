# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One operation on the dev lane's record of branch builds (OMN-22996)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.models.model_dev_lane_branch_build import ModelDevLaneBranchBuild


class ModelDevLaneReturnToDevRequest(BaseModel):
    """Record a build, or settle the records against their pull requests.

    ``record`` carries the build to write. ``settle`` carries the instant to judge,
    the directory of the agent's job records (the builds it declared are read from
    there), each repository's clone, the commit the lane runs for it and the dev
    ref merges are read from. ``hold`` is why the agent may not start a rebuild now (a
    job in flight, a freeze, the load gate); empty when it may. ``git_timeout_seconds`` bounds every git call; it has
    no default (rule 8).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["record", "settle"]
    state_dir: Path
    git_timeout_seconds: int = Field(gt=0)
    build: ModelDevLaneBranchBuild | None = None
    jobs_dir: Path | None = None
    now: datetime | None = None
    clones: dict[str, Path] = Field(default_factory=dict)
    running_refs: dict[str, str] = Field(default_factory=dict)
    dev_ref: str = ""
    hold: str = ""

    @model_validator(mode="after")
    def _operation_inputs(self) -> ModelDevLaneReturnToDevRequest:
        if self.operation == "record" and self.build is None:
            raise ValueError("record needs a build")
        if self.operation == "settle":
            if self.now is None or self.now.tzinfo is None:
                raise ValueError("settle needs the instant to judge, with a timezone")
            if not self.dev_ref:
                raise ValueError("settle needs the dev ref to read merges from")
        return self
