# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Local installer request to retire a whole declared runner pool on one registry host."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerPoolRetireRequest(BaseModel):
    """OMN-26105: ``prefix`` is the exact pool prefix; a runner is ``<prefix>-<digits>`` and nothing else."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["runner.pool.retire"] = "runner.pool.retire"
    host: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
    prefix: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    fleet_config: Path
    only: str = ""
    limit: int = Field(default=0, ge=0)
    dry_run: bool = False
