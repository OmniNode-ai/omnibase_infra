# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one hook mount drift check."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_runner_hook_mount_finding import (
    ModelRunnerHookMountFinding,
)


class ModelRunnerHookMountDriftResult(BaseModel):
    """ok: every executed hook equals its host copy; drift: findings; error: unread."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Literal["ok", "drift", "error"]
    checked: int = 0
    not_checked: tuple[str, ...] = ()
    findings: tuple[ModelRunnerHookMountFinding, ...] = ()
    error_message: str | None = None
