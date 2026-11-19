# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Local installer request for a secondary runner pool image roll."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerImageRollRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["runner.image.roll"] = "runner.image.roll"
    host: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
    prefix: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    repo_root: Path
    fleet_config: Path
    token_path: Path
    only: str = ""
    limit: int = Field(default=0, ge=0)
    dry_run: bool = False
    skip_build: bool = False
    # OMN-26098: replaces the overlay projection path the lab host table is copied from (see ModelRunnerLabHostTablePolicy).
    lab_host_table_projection: str = ""
