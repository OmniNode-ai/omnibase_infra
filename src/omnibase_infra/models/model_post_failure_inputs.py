# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_failed_dev_job import ModelFailedDevJob


class ModelPostFailureInputs(BaseModel):
    """Everything one decision reads, gathered before it is made."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    now: datetime
    job_active: bool
    failed_job: ModelFailedDevJob | None
    #: Repository -> its dev head now, ``None`` when it could not be read.
    heads: dict[str, str | None]
    #: The guarded probe run that excludes ``now``, or ``None``.
    probe_blocker: str | None
    #: Why the probe schedule could not be read, or ``None``.
    windows_error: str | None
    #: ``<repository>@<head>`` for every head a rebuild was already started for.
    attempted: frozenset[str]
    frozen_reason: str | None = None
