# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the deploy agent decided about building one dev merge sha (OMN-26431)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelDevMergeBuildDecisionRequest(BaseModel):
    """One decision of the deploy agent about one dev merge sha, as the row states it.

    ``runtime_class`` is the AC1 vocabulary of the rebuild trigger: ``NON_RUNTIME`` for a
    sha whose changed paths are all declared not-runtime (no build follows), ``PUBLISHED``
    for a sha the trigger published a rebuild command for, ``DETECTED_NOT_PUBLISHED`` for a
    sha the agent found ahead of the lane on its own (its idle converge) with no command.

    ``result`` is the agent's answer for a runtime sha: ``queued`` (``job`` names the job),
    ``deferred`` (``reason`` names the gate, ``earliest_start`` the earliest start when it
    is computable) or ``refused`` (``reason`` names the verdict). A ``NON_RUNTIME`` sha has
    no result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(min_length=1)
    trigger_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_class: Literal["NON_RUNTIME", "PUBLISHED", "DETECTED_NOT_PUBLISHED"]
    result: Literal["queued", "deferred", "refused", ""] = ""
    reason: str = ""
    job: str = ""
    correlation: str = ""
    earliest_start: datetime | None = None
    decided_at: datetime
