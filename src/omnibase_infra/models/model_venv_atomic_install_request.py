# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One step of the atomic in-place install of a venv's moved packages."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelVenvAtomicInstallRequest(BaseModel):
    """Stage, commit, roll back or finalize the packages a venv pass moves."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    operation: Literal["venv.atomic_install"] = "venv.atomic_install"
    command: Literal[
        "supported",
        "exchange",
        "stage",
        "commit",
        "rollback",
        "finalize",
        "verify",
        "installed_commit",
    ]
    python_bin: Path | None = None
    stage_dir: Path | None = None
    requirements: tuple[str, ...] = ()
    no_deps: bool = False
    label: str = "install"
    paths: tuple[Path, ...] = Field(
        default=(), description="the two directories an exchange swaps"
    )
