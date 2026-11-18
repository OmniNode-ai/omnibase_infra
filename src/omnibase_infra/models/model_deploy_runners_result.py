# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Terminal result for runner TMPDIR preparation."""

from pydantic import BaseModel, ConfigDict


class ModelDeployRunnersResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    prepared: tuple[str, ...] = ()
    error_message: str | None = None
    # OMN-24299: the action-archive cache fill and sync never fail the deploy (an
    # absent archive only means a download from GitHub); each problem is named here.
    warnings: tuple[str, ...] = ()
