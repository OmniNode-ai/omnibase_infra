# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Safe evidence returned to the pre-PR cleanup and lease adapters."""

from pydantic import BaseModel, ConfigDict, Field


class ModelPreprTeardownResult(BaseModel):
    """A cleanup prerequisite or report verdict, without raw command output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool
    errors: tuple[str, ...] = ()
    authentication: dict[str, str | int | bool] | None = None
    diagnostics: dict[str, object] = Field(default_factory=dict)
    exit_code: int = 0
