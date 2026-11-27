# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The decision about one dev merge sha's build: satisfied, wait or request (OMN-26316)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelOverdueDevMergeBuildDecision(BaseModel):
    """``outcome`` is ``SATISFIED`` (no build is owed), ``WAIT`` or ``REQUEST``.

    ``reason`` names why: ``non_runtime`` and ``built_by_job`` for SATISFIED;
    ``within_window``, ``coalesced``, ``job_in_flight`` and ``failed_head`` for WAIT; ``overdue_no_build``
    for REQUEST. ``job_id`` names the job the reason rests on, when one does. ``due_at`` is
    the instant the window ends. ``request_id`` is set only for REQUEST: one value per
    repository and sha, never a function of the clock, so a caller that publishes the request
    twice publishes the same id and the second is a duplicate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Literal["SATISFIED", "WAIT", "REQUEST"]
    reason: Literal[
        "non_runtime",
        "built_by_job",
        "within_window",
        "coalesced",
        "job_in_flight",
        "failed_head",
        "overdue_no_build",
    ]
    merge_sha: str
    due_at: datetime
    job_id: UUID | None = None
    request_id: UUID | None = None
