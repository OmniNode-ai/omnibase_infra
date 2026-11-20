# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one branch install or revert."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelVenvBranchInstallResult(BaseModel):
    """The verdict, the install record line and the path of the receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["installed", "reverted", "noop", "refused", "failed"]
    exit_code: int
    installed: str = ""
    receipt_path: str = ""
    replaced_revision: str = ""
    output: str = ""
