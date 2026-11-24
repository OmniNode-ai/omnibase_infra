# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One dev merge sha and the job records that could carry it (OMN-26316)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_overdue_dev_merge_job import ModelOverdueDevMergeJob


class ModelOverdueDevMergeBuildRequest(BaseModel):
    """Everything the decision reads: the merge, its class, the clock and the dev job records.

    ``runtime_class`` is the rebuild trigger's classification of the merge (``NON_RUNTIME``
    needs no build). ``window_seconds`` is how long a runtime merge may wait for a dev build
    before one is requested (the operator's goal is one hour). ``jobs`` are the agent's job
    records of every lane; the decision keeps the ``dev`` lane's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(min_length=1)
    merge_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    merged_at: datetime
    runtime_class: Literal["RUNTIME", "NON_RUNTIME"]
    now: datetime
    window_seconds: int = Field(gt=0)
    jobs: tuple[ModelOverdueDevMergeJob, ...] = ()
