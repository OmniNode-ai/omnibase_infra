# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request for runner TMPDIR preparation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelDeployRunnersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["runner.tmp.prepare"] = "runner.tmp.prepare"
    host: str
    host_dir: str
    compose_file: str
    prefix: str
    count: int = Field(ge=1)
    dry_run: bool = False
    secondary: bool = True
    # OMN-24299: where the fleet declaration and the canonical clones are; both
    # set => the host's action-archive cache is filled and synced after the dirs.
    fleet_config: str | None = None
    clone_root: str | None = None
    # OMN-26098: replaces the overlay projection path the lab host table is copied from (absolute, or relative to the
    # runner host user's home). deploy-runners.sh fills it from RUNNER_LAB_HOST_TABLE_PROJECTION; unset keeps the
    # declared default of ModelRunnerLabHostTablePolicy.
    lab_host_table_projection: str | None = None
