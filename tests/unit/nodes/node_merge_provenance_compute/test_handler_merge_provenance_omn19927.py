# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Provenance verdicts for a ``dev`` push (OMN-19927, plan task S2).

The table rows are the real commits of 2026-09-28, read from the GitHub
Actions API on that day:

* ``e263eee0`` -- omnibase_infra#4111 as it reached ``dev``. The merge queue
  failed its merge group six times (the queue's candidate was ``bbc24237``),
  then a landing lane merged it with a direct REST call. No merge-group run
  exists for ``e263eee0``: the verdict is ``UNVALIDATED``.
* ``7b1f6fb6`` -- omnibase_infra#4235, landed by the queue. Its merge-group
  ``CI`` run (36409772437) concluded ``failure`` overall (a non-required
  runtime-boot job) while its ``CI Summary`` job concluded ``success``, which
  is what the queue gates on: the verdict is ``VALIDATED``.
* A read that failed is ``UNDECIDABLE``, never "no runs".
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_merge_provenance_compute.handlers.handler_merge_provenance import (
    HandlerMergeProvenance,
)
from omnibase_infra.nodes.node_merge_provenance_compute.models.enum_merge_provenance_reason import (
    EnumMergeProvenanceReason,
)
from omnibase_infra.nodes.node_merge_provenance_compute.models.enum_merge_provenance_verdict import (
    EnumMergeProvenanceVerdict,
)
from omnibase_infra.nodes.node_merge_provenance_compute.models.model_merge_provenance_request import (
    ModelMergeProvenanceRequest,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_merge_group_run_fact import (
    ModelMergeGroupRunFact,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_merge_provenance_observation import (
    ModelMergeProvenanceObservation,
)

pytestmark = pytest.mark.unit

REPO = "OmniNode-ai/omnibase_infra"
CI = ".github/workflows/ci.yml"
SUMMARY = "CI Summary"

SHA_4111_ON_DEV = "e263eee036a8e338e6b39ea746d7e59bf91602cb"
SHA_4111_QUEUE_CANDIDATE = "bbc24237a7b1c072f377333a6cf4b8b8f337f119"
SHA_4235_QUEUE_LANDED = "7b1f6fb690d65e86c804e011ef80b8485a4d070e"


def _run(
    sha: str,
    run_id: int,
    *,
    run_conclusion: str | None,
    summary_conclusion: str | None,
    summary_status: str | None = "completed",
    event: str = "merge_group",
    workflow_path: str = CI,
    pr: int = 4235,
) -> ModelMergeGroupRunFact:
    return ModelMergeGroupRunFact(
        run_id=run_id,
        run_attempt=1,
        event=event,
        head_sha=sha,
        head_branch=f"gh-readonly-queue/dev/pr-{pr}-{sha[:8]}",
        workflow_path=workflow_path,
        run_status="completed" if run_conclusion else "in_progress",
        run_conclusion=run_conclusion,
        summary_job_status=summary_status,
        summary_job_conclusion=summary_conclusion,
    )


def _observation(
    sha: str,
    runs: tuple[ModelMergeGroupRunFact, ...] = (),
    *,
    read_ok: bool = True,
    read_error: str | None = None,
) -> ModelMergeProvenanceObservation:
    return ModelMergeProvenanceObservation(
        repository=REPO,
        sha=sha,
        workflow_path=CI,
        summary_job=SUMMARY,
        read_ok=read_ok,
        read_error=read_error,
        runs=runs,
        observed_at="2026-09-28T15:40:00Z",
    )


def _verdict(sha: str, observation: ModelMergeProvenanceObservation):  # type: ignore[no-untyped-def]
    return HandlerMergeProvenance().handle(
        ModelMergeProvenanceRequest(repository=REPO, sha=sha, observation=observation)
    )


# The plan's three-row table (S2, first falsifier).
def test_e263eee_out_of_queue_merge_is_unvalidated() -> None:
    result = _verdict(SHA_4111_ON_DEV, _observation(SHA_4111_ON_DEV))
    assert result.verdict is EnumMergeProvenanceVerdict.UNVALIDATED
    assert result.reason is EnumMergeProvenanceReason.NO_MERGE_GROUP_RUN
    assert result.forces_full_suite is True
    assert result.validating_run_id is None


def test_queue_landed_commit_is_validated() -> None:
    run = _run(
        SHA_4235_QUEUE_LANDED,
        36409772437,
        run_conclusion="failure",
        summary_conclusion="success",
    )
    result = _verdict(
        SHA_4235_QUEUE_LANDED, _observation(SHA_4235_QUEUE_LANDED, (run,))
    )
    assert result.verdict is EnumMergeProvenanceVerdict.VALIDATED
    assert result.reason is EnumMergeProvenanceReason.MERGE_GROUP_SUMMARY_SUCCESS
    assert result.validating_run_id == 36409772437
    assert result.run_ids_read == (36409772437,)
    assert result.forces_full_suite is False


def test_failed_read_is_undecidable_never_empty() -> None:
    result = _verdict(
        SHA_4111_ON_DEV,
        _observation(SHA_4111_ON_DEV, read_ok=False, read_error="HTTP 502"),
    )
    assert result.verdict is EnumMergeProvenanceVerdict.UNDECIDABLE
    assert result.reason is EnumMergeProvenanceReason.READ_FAILED
    assert result.forces_full_suite is True
    assert "HTTP 502" in result.reason_detail


# Rows beyond the plan's three, each a way the lookup could lie.
def test_red_merge_group_summary_is_unvalidated() -> None:
    """bbc24237 is #4111's own queue candidate, whose CI Summary failed."""
    run = _run(
        SHA_4111_QUEUE_CANDIDATE,
        36403686267,
        run_conclusion="failure",
        summary_conclusion="failure",
        pr=4111,
    )
    result = _verdict(
        SHA_4111_QUEUE_CANDIDATE, _observation(SHA_4111_QUEUE_CANDIDATE, (run,))
    )
    assert result.verdict is EnumMergeProvenanceVerdict.UNVALIDATED
    assert result.reason is EnumMergeProvenanceReason.NO_SUCCESSFUL_SUMMARY
    assert result.run_ids_read == (36403686267,)


def test_in_progress_summary_is_not_validation() -> None:
    run = _run(
        SHA_4235_QUEUE_LANDED,
        1,
        run_conclusion=None,
        summary_conclusion=None,
        summary_status="in_progress",
    )
    result = _verdict(
        SHA_4235_QUEUE_LANDED, _observation(SHA_4235_QUEUE_LANDED, (run,))
    )
    assert result.verdict is EnumMergeProvenanceVerdict.UNVALIDATED


def test_run_without_the_summary_job_is_undecidable() -> None:
    """A renamed summary job must not read as 'no validation happened'."""
    run = _run(
        SHA_4235_QUEUE_LANDED,
        2,
        run_conclusion="success",
        summary_conclusion=None,
        summary_status=None,
    )
    result = _verdict(
        SHA_4235_QUEUE_LANDED, _observation(SHA_4235_QUEUE_LANDED, (run,))
    )
    assert result.verdict is EnumMergeProvenanceVerdict.UNDECIDABLE
    assert result.reason is EnumMergeProvenanceReason.SUMMARY_JOB_ABSENT


def test_observation_for_another_sha_is_undecidable() -> None:
    result = _verdict(SHA_4111_ON_DEV, _observation(SHA_4235_QUEUE_LANDED))
    assert result.verdict is EnumMergeProvenanceVerdict.UNDECIDABLE
    assert result.reason is EnumMergeProvenanceReason.OBSERVATION_MISMATCH


@pytest.mark.parametrize(
    ("event", "workflow_path", "sha"),
    [
        ("push", CI, SHA_4235_QUEUE_LANDED),
        ("merge_group", ".github/workflows/deploy-gate.yml", SHA_4235_QUEUE_LANDED),
        ("merge_group", CI, SHA_4111_QUEUE_CANDIDATE),
    ],
    ids=["push-event", "other-workflow", "other-sha"],
)
def test_a_green_summary_counts_only_from_this_workflow_event_and_sha(
    event: str, workflow_path: str, sha: str
) -> None:
    run = _run(
        sha,
        3,
        run_conclusion="success",
        summary_conclusion="success",
        event=event,
        workflow_path=workflow_path,
    )
    result = _verdict(
        SHA_4235_QUEUE_LANDED, _observation(SHA_4235_QUEUE_LANDED, (run,))
    )
    assert result.verdict is not EnumMergeProvenanceVerdict.VALIDATED


def test_handler_is_deterministic() -> None:
    obs = _observation(SHA_4111_ON_DEV)
    assert _verdict(SHA_4111_ON_DEV, obs) == _verdict(SHA_4111_ON_DEV, obs)
