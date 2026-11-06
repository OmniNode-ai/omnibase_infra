# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Terminal result for runner TMPDIR preparation."""

from pydantic import BaseModel, ConfigDict


class ModelDeployRunnersResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    prepared: tuple[str, ...] = ()
    error_message: str | None = None
