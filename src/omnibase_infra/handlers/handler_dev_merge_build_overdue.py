# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whether a dev runtime merge has gone a window with no build, and the one request that follows (OMN-26316).

THE FAILURE
-----------
Dev ran on a build of 17:12Z while omnibase_infra#6191 (``a27c0c218``) merged at
17:46:52Z. The deploy agent builds the dev head when the lane is down or its recreate
window admits an idle restore; the Actions trigger's command for the merge was refused
``not_routed_elsewhere``. The next dev job (``2098b18c``) was accepted at 19:00:23Z, 73
minutes after the merge, and nothing asked for a build in between. The operator's goal is
that every runtime merge is on a dev build within an hour and that a build is requested
when none starts.

THE DECISION
------------
A pure judgement over what the caller read (the merge, the clock and the agent's job
records); it reads no host and publishes nothing. In order:

* ``NON_RUNTIME`` merge: ``SATISFIED``, ``non_runtime``.
* A dev job accepted at or after the merge on ``origin/dev`` (or on the sha itself) that
  succeeded and whose built sha the caller proved contains the merge: ``SATISFIED``,
  ``built_by_job``. A job accepted before the merge staged its source before it, so it never
  carries it. A ``superseded`` job carries nothing. A job whose built sha is unread, is not
  among the proven descendants, or was requested by the stability follow carries nothing:
  the doubt fails closed to ``WAIT`` or ``REQUEST``, never to ``SATISFIED``.
* A dev job in flight, whatever it carries: ``WAIT``, ``job_in_flight`` (never a second
  build over a running one).
* A dev job that carries the merge and failed: ``WAIT``, ``failed_head``; the failed-head
  gate owns that head, and asking again would only be refused.
* Inside the window: ``WAIT``, ``within_window``.
* A dev job (not superseded, not the stability follow's) accepted less than the declared
  coalesce window ago: ``WAIT``, ``coalesced`` (the hourly build carries every merge, so a
  build asked for just after one waits for the next). The coalesce window is declared in the
  node contract; ``0`` coalesces nothing.
* A job pinned to a descendant of the merge (the ref an overdue request itself builds) carries
  it like a job on ``origin/dev``, so one build asked for one merge carries the merges before it.
* Otherwise ``REQUEST``, ``overdue_no_build``, with a ``request_id`` that is a function of
  the repository and sha only.

Whether a job's built sha descends from the merge is a git read, which this decision does
not make: the caller reads it (``descendant_shas``). The cost of the doubt is one request,
which the agent dedups. A failed job whose built sha is unread still reads as the failed
head's, because the failed-head gate owns a head the job may have carried.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Final
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_job_status import EnumJobStatus
from omnibase_infra.models.model_overdue_dev_merge_build_decision import (
    ModelOverdueDevMergeBuildDecision,
)
from omnibase_infra.models.model_overdue_dev_merge_build_request import (
    ModelOverdueDevMergeBuildRequest,
)
from omnibase_infra.models.model_overdue_dev_merge_job import ModelOverdueDevMergeJob

_DEV_LANE: Final = "dev"
_DEV_HEAD: Final = "origin/dev"
#: ``requested_by`` of the stability lane's rebuild that follows dev's tag; it never builds a dev merge.
_STABILITY_FOLLOW: Final = "agent/stability-follow-tag"
#: Statuses that end a job. Any other status is a job still on the agent.
_TERMINAL: Final = frozenset(
    {EnumJobStatus.SUCCESS, EnumJobStatus.FAILED, EnumJobStatus.SUPERSEDED}
)
#: Fixed namespace of the request ids: the id depends on the repository and sha alone.
_REQUEST_NAMESPACE: Final = uuid.UUID("5b0f3d6e-0f55-4c58-9d0e-6c1d2b7e4a31")


def request_id_for(repository: str, merge_sha: str) -> UUID:
    """The one request id of a merge sha: the same value every time it is asked for."""
    return uuid.uuid5(_REQUEST_NAMESPACE, f"{repository}:{merge_sha}")


def _may_carry(
    job: ModelOverdueDevMergeJob, request: ModelOverdueDevMergeBuildRequest
) -> bool:
    return (
        job.requested_by != _STABILITY_FOLLOW
        and job.accepted_at >= request.merged_at
        and (
            job.git_ref in (_DEV_HEAD, request.merge_sha)
            or job.git_ref in request.descendant_shas
        )
    )


def _carries(
    job: ModelOverdueDevMergeJob, request: ModelOverdueDevMergeBuildRequest
) -> bool:
    return (
        _may_carry(job, request)
        and job.built_sha != ""
        and job.built_sha in request.descendant_shas
    )


class HandlerDevMergeBuildOverdue:
    """Judge whether a runtime merge is owed a build request."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelOverdueDevMergeBuildRequest
    ) -> ModelOverdueDevMergeBuildDecision:
        due_at = request.merged_at + timedelta(seconds=request.window_seconds)

        def decide(
            outcome: str,
            reason: str,
            job_id: UUID | None = None,
            request_id: UUID | None = None,
        ) -> ModelOverdueDevMergeBuildDecision:
            return ModelOverdueDevMergeBuildDecision.model_validate(
                {
                    "outcome": outcome,
                    "reason": reason,
                    "merge_sha": request.merge_sha,
                    "due_at": due_at,
                    "job_id": job_id,
                    "request_id": request_id,
                }
            )

        if request.runtime_class == "NON_RUNTIME":
            return decide("SATISFIED", "non_runtime")

        dev_jobs = sorted(
            (job for job in request.jobs if job.runtime_lane == _DEV_LANE),
            key=lambda job: (job.accepted_at, job.job_id),
        )
        for job in dev_jobs:
            if job.status == EnumJobStatus.SUCCESS and _carries(job, request):
                return decide("SATISFIED", "built_by_job", job.job_id)
        for job in dev_jobs:
            if job.status not in _TERMINAL:
                return decide("WAIT", "job_in_flight", job.job_id)
        for job in dev_jobs:
            if (
                job.status == EnumJobStatus.FAILED
                and _may_carry(job, request)
                and (job.built_sha == "" or job.built_sha in request.descendant_shas)
            ):
                return decide("WAIT", "failed_head", job.job_id)
        if request.now < due_at:
            return decide("WAIT", "within_window")
        coalesce = timedelta(seconds=request.coalesce_seconds)
        for job in reversed(dev_jobs):
            if (
                job.requested_by != _STABILITY_FOLLOW
                and job.status != EnumJobStatus.SUPERSEDED
                and request.now - job.accepted_at < coalesce
            ):
                return decide("WAIT", "coalesced", job.job_id)
        return decide(
            "REQUEST",
            "overdue_no_build",
            request_id=request_id_for(request.repository, request.merge_sha),
        )
