# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Inputs collected by the existing pre-PR teardown CLI."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class ModelPreprTeardownRequest(BaseModel):
    """One tenant-offboard operation or teardown-report read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["revoke_tenant", "inspect_report", "write_report"]
    compose_project: str = Field(pattern=r"^omnibase-infra-prepr-[1-9][0-9]*$")
    slot_token: str = Field(pattern=r"^prepr[1-9][0-9]*$")
    staging_root: Path
    repo_root: Path
    report_path: Path | None = None
    returncode: int = 0
    plan_only: bool = False
    planned: dict[str, object] = Field(default_factory=dict)
    residue: dict[str, StrictInt] = Field(default_factory=dict)
    controls: dict[str, StrictInt] = Field(default_factory=dict)
    errors: tuple[str, ...] = ()
    authentication: dict[str, str | int | bool] | None = None
