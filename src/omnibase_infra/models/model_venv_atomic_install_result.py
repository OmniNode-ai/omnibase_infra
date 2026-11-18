# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one atomic in-place install step."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelVenvAtomicInstallResult(BaseModel):
    """The step's exit code, its verdict and the output the tool wrote."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["ok", "refused", "declined", "failed"]
    exit_code: int
    output: str = ""
