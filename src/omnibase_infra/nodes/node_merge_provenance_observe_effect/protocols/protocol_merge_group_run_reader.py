# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The read port of the merge-provenance observe effect (OMN-19927)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_workflow_job_fact import (
    ModelWorkflowJobFact,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_workflow_run_fact import (
    ModelWorkflowRunFact,
)


@runtime_checkable
class ProtocolMergeGroupRunReader(Protocol):
    """Reads Actions runs and jobs. Raises on any failed or partial read.

    An implementation must raise rather than return a short list when a read
    fails or a page is missing: the effect turns the exception into
    ``read_ok=False``, and a short list would be graded as a real absence.
    """

    def list_merge_group_runs(
        self, repository: str, head_sha: str
    ) -> list[ModelWorkflowRunFact]:
        """Every workflow run with event ``merge_group`` for ``head_sha``."""
        ...

    def list_latest_attempt_jobs(
        self, repository: str, run_id: int
    ) -> list[ModelWorkflowJobFact]:
        """Every job of the run's latest attempt."""
        ...


__all__: list[str] = ["ProtocolMergeGroupRunReader"]
