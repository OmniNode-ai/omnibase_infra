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
    # OMN-24139: a lab lane's GitHub shim refuses the raw org registry read, so the orchestrator supplies the registry
    # as a file (the output of `gh api --paginate --slurp /orgs/<org>/actions/runners?per_page=100`). Empty keeps the
    # direct `gh` read. A snapshot older than the age limit is refused, never used.
    registry_snapshot: str = ""
    registry_snapshot_max_age_seconds: int = Field(default=3600, ge=1)
