# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The record of one branch install: what moved in, and what it replaced."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelVenvBranchInstallReceipt(BaseModel):
    """Written once per install and updated once, by the revert."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    host: str
    repository: Literal["omnimarket"] = "omnimarket"
    pr_number: int = Field(ge=1)
    head_sha: str = Field(min_length=40, max_length=40)
    replaced_revision: str = Field(
        min_length=40,
        max_length=40,
        description="the revision the venv held before the install; the revert target",
    )
    units: tuple[str, ...]
    venv: str
    stage_dir: str = Field(
        description="holds the journal and the replaced files the revert restores"
    )
    installed_at: str
    state: Literal["active", "reverted"] = "active"
    reverted_at: str = ""

    @property
    def installed(self) -> str:
        """The ``<host>:<full sha>`` the install is recorded as."""
        return f"{self.host}:{self.head_sha}"
