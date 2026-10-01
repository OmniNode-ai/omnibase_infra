# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Grade a commit's merge provenance (OMN-19927, plan task S2).

Pure and deterministic, definition B: ``handle(request) -> result``. A queue
landing fast-forwards the base branch to the merge-group commit, so "was this
commit validated by the queue?" is a lookup: a merge-group run of the
workflow, for this exact sha, whose summary job concluded ``success``.

The run's own conclusion is deliberately not read. The queue gates on the
summary job (the one required context), and a merge-group run routinely
concludes ``failure`` on a non-required job while its summary is green:
omnibase_infra#4235's run 36409772437 on 2026-09-28 is the recorded case.

Anything that is not a positive, exact-match success is not VALIDATED. What
cannot be read, or cannot be matched on its key, is UNDECIDABLE, and both
UNVALIDATED and UNDECIDABLE force the full suite.
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_merge_provenance_compute.models.enum_merge_provenance_reason import (
    EnumMergeProvenanceReason,
)
from omnibase_infra.nodes.node_merge_provenance_compute.models.enum_merge_provenance_verdict import (
    EnumMergeProvenanceVerdict,
)
from omnibase_infra.nodes.node_merge_provenance_compute.models.model_merge_provenance_request import (
    ModelMergeProvenanceRequest,
)
from omnibase_infra.nodes.node_merge_provenance_compute.models.model_merge_provenance_result import (
    ModelMergeProvenanceResult,
)

_MERGE_GROUP_EVENT = "merge_group"
_SUCCESS = "success"


class HandlerMergeProvenance:
    """Commit plus merge-group observation, to a provenance verdict."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelMergeProvenanceRequest
    ) -> ModelMergeProvenanceResult:
        """Grade ``request.sha``."""
        obs = request.observation
        if obs.repository != request.repository or obs.sha != request.sha:
            return self._result(
                request,
                EnumMergeProvenanceVerdict.UNDECIDABLE,
                EnumMergeProvenanceReason.OBSERVATION_MISMATCH,
                f"observation is for {obs.repository}@{obs.sha}, "
                f"not {request.repository}@{request.sha}",
            )
        if not obs.read_ok:
            return self._result(
                request,
                EnumMergeProvenanceVerdict.UNDECIDABLE,
                EnumMergeProvenanceReason.READ_FAILED,
                f"merge-group read failed: {obs.read_error}",
            )

        runs = tuple(
            r
            for r in obs.runs
            if r.event == _MERGE_GROUP_EVENT
            and r.head_sha == request.sha
            and r.workflow_path == obs.workflow_path
        )
        run_ids = tuple(sorted(r.run_id for r in runs))
        if not runs:
            return self._result(
                request,
                EnumMergeProvenanceVerdict.UNVALIDATED,
                EnumMergeProvenanceReason.NO_MERGE_GROUP_RUN,
                f"no {_MERGE_GROUP_EVENT} run of {obs.workflow_path} for this sha",
            )

        green = sorted(r.run_id for r in runs if r.summary_job_conclusion == _SUCCESS)
        if green:
            return self._result(
                request,
                EnumMergeProvenanceVerdict.VALIDATED,
                EnumMergeProvenanceReason.MERGE_GROUP_SUMMARY_SUCCESS,
                f"{obs.summary_job!r} succeeded in merge-group run {green[0]}",
                run_ids=run_ids,
                validating_run_id=green[0],
            )

        absent = sorted(r.run_id for r in runs if r.summary_job_status is None)
        if absent:
            return self._result(
                request,
                EnumMergeProvenanceVerdict.UNDECIDABLE,
                EnumMergeProvenanceReason.SUMMARY_JOB_ABSENT,
                f"merge-group runs {absent} have no job named "
                f"{obs.summary_job!r}; the lookup key may be stale",
                run_ids=run_ids,
            )

        seen = ", ".join(
            f"{r.run_id}={r.summary_job_conclusion or r.summary_job_status}"
            for r in sorted(runs, key=lambda r: r.run_id)
        )
        return self._result(
            request,
            EnumMergeProvenanceVerdict.UNVALIDATED,
            EnumMergeProvenanceReason.NO_SUCCESSFUL_SUMMARY,
            f"no merge-group run has a successful {obs.summary_job!r}: {seen}",
            run_ids=run_ids,
        )

    @staticmethod
    def _result(
        request: ModelMergeProvenanceRequest,
        verdict: EnumMergeProvenanceVerdict,
        reason: EnumMergeProvenanceReason,
        detail: str,
        *,
        run_ids: tuple[int, ...] = (),
        validating_run_id: int | None = None,
    ) -> ModelMergeProvenanceResult:
        return ModelMergeProvenanceResult(
            repository=request.repository,
            sha=request.sha,
            verdict=verdict,
            reason=reason,
            reason_detail=detail,
            run_ids_read=run_ids,
            validating_run_id=validating_run_id,
            forces_full_suite=verdict is not EnumMergeProvenanceVerdict.VALIDATED,
        )


__all__: list[str] = ["HandlerMergeProvenance"]
