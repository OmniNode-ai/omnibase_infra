# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Observe the merge-group runs recorded for one commit (OMN-19927).

EFFECT handler: every GitHub read goes through the injected
``ProtocolMergeGroupRunReader``. It grades nothing; the verdict is
``node_merge_provenance_compute``'s. Its one rule is the plan's: a failed read
is an observation with ``read_ok=False``, never an empty list.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_merge_group_run_fact import (
    ModelMergeGroupRunFact,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_merge_provenance_observation import (
    ModelMergeProvenanceObservation,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_merge_provenance_observe_request import (
    ModelMergeProvenanceObserveRequest,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.protocols.protocol_merge_group_run_reader import (
    ProtocolMergeGroupRunReader,
)

_MAX_ERROR_CHARS = 300


class HandlerMergeProvenanceObserve:
    """Reads merge-group runs for a sha and the summary job of each."""

    def __init__(self, reader: ProtocolMergeGroupRunReader) -> None:
        self._reader = reader

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelMergeProvenanceObserveRequest
    ) -> ModelMergeProvenanceObservation:
        """Observe ``request.sha``; a failed read returns ``read_ok=False``."""
        try:
            runs = await asyncio.to_thread(self._read, request)
        except Exception as exc:  # noqa: BLE001 -- every failure is an observation
            return self._observation(
                request,
                read_ok=False,
                read_error=f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS],
            )
        return self._observation(request, read_ok=True, runs=runs)

    def _read(
        self, request: ModelMergeProvenanceObserveRequest
    ) -> tuple[ModelMergeGroupRunFact, ...]:
        facts: list[ModelMergeGroupRunFact] = []
        runs = self._reader.list_merge_group_runs(request.repository, request.sha)
        for run in sorted(runs, key=lambda r: r.run_id):
            if run.workflow_path != request.workflow_path:
                continue
            jobs = self._reader.list_latest_attempt_jobs(request.repository, run.run_id)
            summary = [j for j in jobs if j.name == request.summary_job]
            job = summary[0] if len(summary) == 1 else None
            if len(summary) > 1:
                raise RuntimeError(
                    f"run {run.run_id} has {len(summary)} jobs named "
                    f"{request.summary_job!r} in its latest attempt"
                )
            facts.append(
                ModelMergeGroupRunFact(
                    run_id=run.run_id,
                    run_attempt=run.run_attempt,
                    event=run.event,
                    head_sha=run.head_sha,
                    head_branch=run.head_branch,
                    workflow_path=run.workflow_path,
                    run_status=run.status,
                    run_conclusion=run.conclusion,
                    summary_job_status=job.status if job else None,
                    summary_job_conclusion=job.conclusion if job else None,
                )
            )
        return tuple(facts)

    @staticmethod
    def _observation(
        request: ModelMergeProvenanceObserveRequest,
        *,
        read_ok: bool,
        read_error: str | None = None,
        runs: tuple[ModelMergeGroupRunFact, ...] = (),
    ) -> ModelMergeProvenanceObservation:
        return ModelMergeProvenanceObservation(
            repository=request.repository,
            sha=request.sha,
            workflow_path=request.workflow_path,
            summary_job=request.summary_job,
            read_ok=read_ok,
            read_error=read_error,
            runs=runs,
            observed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )


__all__: list[str] = ["HandlerMergeProvenanceObserve"]
