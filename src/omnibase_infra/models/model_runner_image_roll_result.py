# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Observed result of a secondary runner pool image roll."""

from pydantic import BaseModel, ConfigDict


class ModelRunnerImageRollResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    image: str = ""
    planned: tuple[str, ...] = ()
    recreated: tuple[str, ...] = ()
    error_message: str | None = None
