# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One branch install or revert of an omnimarket pull request head on a host."""

import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class ModelVenvBranchInstallRequest(BaseModel):
    """Install the head of one omnimarket pull request into a host's dispatch venv,
    or put the revision the install replaced back.

    The head sha is the one the pull request watcher read; the caller never types
    it. An install names the units that run from the venv: the venv is shared by
    every unit and lane on the host, so naming them is the record of who the
    install moves until it is reverted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    operation: Literal["venv.branch_install"] = "venv.branch_install"
    command: Literal["install", "revert"]
    host: str = Field(min_length=1, description="host the dispatch venv belongs to")
    venv: Path = Field(description="dispatch venv root")
    state_dir: Path = Field(description="where receipts and stage folders are kept")
    pr_number: int = Field(default=0, ge=0, description="omnimarket pull request")
    head_sha: str = Field(default="", description="full head sha of that pull request")
    units: tuple[str, ...] = Field(
        default=(), description="units that run from the venv and are moved by it"
    )

    @model_validator(mode="after")
    def _install_names_what_it_installs(self) -> Self:
        if self.command == "install":
            if self.pr_number < 1:
                raise ValueError("an install names an omnimarket pull request number")
            if not _FULL_SHA.match(self.head_sha):
                raise ValueError("an install names the full 40-character head sha")
            if not self.units or not all(u.strip() for u in self.units):
                raise ValueError("an install names the units it moves")
        return self
