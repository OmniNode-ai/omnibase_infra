# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Observed result of a runner pool retire."""

from pydantic import BaseModel, ConfigDict


class ModelRunnerPoolRetireResult(BaseModel):
    """``planned`` is the selection; ``retired`` and ``deregistered`` are what was read back; ``busy`` was left alone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    planned: tuple[str, ...] = ()
    retired: tuple[str, ...] = ()
    deregistered: tuple[str, ...] = ()
    busy: tuple[str, ...] = ()
    error_message: str | None = None
