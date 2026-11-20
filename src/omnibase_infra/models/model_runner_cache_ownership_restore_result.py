# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one cache ownership restore pass (OMN-26113)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_runner_cache_restore_finding import (
    ModelRunnerCacheRestoreFinding,
)


class ModelRunnerCacheOwnershipRestoreResult(BaseModel):
    """ok: nothing failed; failed: a restore did not hold; error: the container list was unread.

    ``busy`` names every container left untouched because a ``Runner.Worker`` was running or the process list was
    unreadable: it is fixed on a later pass, never by interrupting the job.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Literal["ok", "failed", "error"]
    checked: int = 0
    already_owned: tuple[str, ...] = ()
    busy: tuple[str, ...] = ()
    no_cache_mounts: tuple[str, ...] = ()
    findings: tuple[ModelRunnerCacheRestoreFinding, ...] = ()
    error_message: str | None = None
