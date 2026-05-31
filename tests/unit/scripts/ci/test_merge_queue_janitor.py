# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from scripts.ci.merge_queue_janitor import WorkflowRun, determine_cancellations


def _run(
    run_id: int,
    *,
    ref: str = "gh-readonly-queue/dev/pr-1-head",
    workflow: str = "CI",
    event: str = "merge_group",
    status: str = "in_progress",
    created_at: str | None = None,
) -> WorkflowRun:
    return WorkflowRun(
        run_id=run_id,
        event=event,
        head_branch=ref,
        workflow_name=workflow,
        status=status,
        created_at=created_at or f"2026-05-31T10:{run_id:02d}:00Z",
        updated_at=created_at or f"2026-05-31T10:{run_id:02d}:30Z",
    )


def test_marks_stale_queue_refs_for_cancellation() -> None:
    candidates = determine_cancellations(
        live_queue_refs={"gh-readonly-queue/dev/pr-2-"},
        active_runs=[_run(1, ref="gh-readonly-queue/dev/pr-1-head")],
    )

    assert len(candidates) == 1
    assert candidates[0].run_id == 1
    assert candidates[0].reason == "stale_queue_ref"


def test_preserves_newest_duplicate_for_same_live_ref_and_workflow() -> None:
    candidates = determine_cancellations(
        live_queue_refs={"gh-readonly-queue/dev/pr-1-"},
        active_runs=[
            _run(10, workflow="CI", created_at="2026-05-31T10:10:00Z"),
            _run(11, workflow="CI", created_at="2026-05-31T10:11:00Z"),
            _run(12, workflow="Receipt Gate", created_at="2026-05-31T10:09:00Z"),
        ],
    )

    assert [candidate.run_id for candidate in candidates] == [10]
    assert candidates[0].reason == "older_duplicate_workflow_for_live_ref"
    assert candidates[0].preserved_run_id == 11


def test_ignores_non_merge_group_and_completed_runs() -> None:
    candidates = determine_cancellations(
        live_queue_refs={"gh-readonly-queue/dev/pr-1-"},
        active_runs=[
            _run(1, event="pull_request"),
            _run(2, status="completed"),
            _run(3, workflow="CI"),
        ],
    )

    assert candidates == []
