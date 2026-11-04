# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One final stdout document and all sample diagnostics."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelCensusOneshotResult(BaseModel):
    """One final stdout document and all sample diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    exit_code: int
    stdout: str = ""
    stderr: str = ""
