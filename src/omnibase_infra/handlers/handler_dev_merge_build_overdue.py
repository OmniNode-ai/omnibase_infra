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
  succeeded: ``SATISFIED``, ``built_by_job``. A job accepted before the merge staged its
  source before it, so it never carries it. A ``superseded`` job carries nothing.
* A dev job in flight, whatever it carries: ``WAIT``, ``job_in_flight`` (never a second
  build over a running one).
* A dev job that carries the merge and failed: ``WAIT``, ``failed_head``; the failed-head
  gate owns that head, and asking again would only be refused.
* Inside the window: ``WAIT``, ``within_window``.
* Otherwise ``REQUEST``, ``overdue_no_build``, with a ``request_id`` that is a function of
  the repository and sha only.

A job pinned to a sha other than the merge's is not counted as carrying it: whether that
sha descends from the merge is a git read, which this decision does not make. The cost of
the doubt is one request, which the agent dedups.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Final

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_overdue_dev_merge_build_decision import (
    ModelOverdueDevMergeBuildDecision,
)
from omnibase_infra.models.model_overdue_dev_merge_build_request import (
    ModelOverdueDevMergeBuildRequest,
)
from omnibase_infra.models.model_overdue_dev_merge_job import ModelOverdueDevMergeJob

_DEV_LANE: Final = "dev"
_DEV_HEAD: Final = "origin/dev"
#: Statuses that end a job. Any other status is a job still on the agent.
_TERMINAL: Final = frozenset({"success", "failed", "superseded"})
#: Fixed namespace of the request ids: the id depends on the repository and sha alone.
_REQUEST_NAMESPACE: Final = uuid.UUID("5b0f3d6e-0f55-4c58-9d0e-6c1d2b7e4a31")


def request_id_for(repository: str, merge_sha: str) -> str:
    """The one request id of a merge sha: the same value every time it is asked for."""
    return str(uuid.uuid5(_REQUEST_NAMESPACE, f"{repository}:{merge_sha}"))


def _carries(
    job: ModelOverdueDevMergeJob, request: ModelOverdueDevMergeBuildRequest
) -> bool:
    return job.accepted_at >= request.merged_at and job.git_ref in (
        _DEV_HEAD,
        request.merge_sha,
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
            outcome: str, reason: str, job_id: str = "", request_id: str = ""
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
        carrying = [job for job in dev_jobs if _carries(job, request)]
        for job in carrying:
            if job.status == "success":
                return decide("SATISFIED", "built_by_job", job.job_id)
        for job in dev_jobs:
            if job.status not in _TERMINAL:
                return decide("WAIT", "job_in_flight", job.job_id)
        for job in carrying:
            if job.status == "failed":
                return decide("WAIT", "failed_head", job.job_id)
        if request.now < due_at:
            return decide("WAIT", "within_window")
        return decide(
            "REQUEST",
            "overdue_no_build",
            request_id=request_id_for(request.repository, request.merge_sha),
        )
