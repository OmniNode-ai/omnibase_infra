# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The merge-provenance observe effect (OMN-19927, plan task S2).

The effect reads, through an injected reader, the merge-group runs recorded
for one sha and the summary job of each. A failed read is an observation with
``read_ok=False`` -- never an empty run list, which would read as "not
validated" and hide the outage behind a plausible verdict.
"""

from __future__ import annotations

import asyncio

import pytest

from omnibase_infra.nodes.node_merge_provenance_observe_effect.handlers.handler_merge_group_run_read_github import (
    HandlerMergeGroupRunReadGithub,
    MergeGroupReadError,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.handlers.handler_merge_provenance_observe import (
    HandlerMergeProvenanceObserve,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_merge_provenance_observe_request import (
    ModelMergeProvenanceObserveRequest,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_workflow_job_fact import (
    ModelWorkflowJobFact,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_workflow_run_fact import (
    ModelWorkflowRunFact,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.protocols.protocol_merge_group_run_reader import (
    ProtocolMergeGroupRunReader,
)

pytestmark = pytest.mark.unit

REPO = "OmniNode-ai/omnibase_infra"
SHA = "7b1f6fb690d65e86c804e011ef80b8485a4d070e"


class _Reader:
    def __init__(
        self,
        runs: list[ModelWorkflowRunFact],
        jobs: dict[int, list[ModelWorkflowJobFact]],
        fail_on: str | None = None,
    ) -> None:
        self.runs = runs
        self.jobs = jobs
        self.fail_on = fail_on
        self.run_queries: list[tuple[str, str]] = []

    def list_merge_group_runs(
        self, repository: str, head_sha: str
    ) -> list[ModelWorkflowRunFact]:
        self.run_queries.append((repository, head_sha))
        if self.fail_on == "runs":
            raise RuntimeError("GitHub API HTTP 502 on actions/runs")
        return self.runs

    def list_latest_attempt_jobs(
        self, repository: str, run_id: int
    ) -> list[ModelWorkflowJobFact]:
        if self.fail_on == "jobs":
            raise RuntimeError(f"GitHub API HTTP 403 on runs/{run_id}/jobs")
        return self.jobs.get(run_id, [])


def _run(run_id: int, path: str = ".github/workflows/ci.yml") -> ModelWorkflowRunFact:
    return ModelWorkflowRunFact(
        run_id=run_id,
        run_attempt=1,
        event="merge_group",
        head_sha=SHA,
        head_branch="gh-readonly-queue/dev/pr-4235-31258513",
        workflow_path=path,
        status="completed",
        conclusion="failure",
    )


def _job(name: str, conclusion: str | None = "success") -> ModelWorkflowJobFact:
    return ModelWorkflowJobFact(
        name=name,
        status="completed" if conclusion else "in_progress",
        conclusion=conclusion,
    )


def _observe(reader: _Reader):  # type: ignore[no-untyped-def]
    assert isinstance(reader, ProtocolMergeGroupRunReader)
    handler = HandlerMergeProvenanceObserve(reader=reader)
    return asyncio.run(
        handler.handle(ModelMergeProvenanceObserveRequest(repository=REPO, sha=SHA))
    )


def test_reads_the_summary_job_of_each_ci_merge_group_run() -> None:
    reader = _Reader(
        runs=[_run(36409772437), _run(99, path=".github/workflows/deploy-gate.yml")],
        jobs={
            36409772437: [
                _job("Lint"),
                _job("CI Summary", "success"),
                _job("Runtime Boot Smoke (compose)", "failure"),
            ]
        },
    )
    obs = _observe(reader)
    assert obs.read_ok is True
    assert reader.run_queries == [(REPO, SHA)]
    # Only the CI workflow's runs are carried; the other workflow is not a
    # validation of this repository's required context.
    assert [r.run_id for r in obs.runs] == [36409772437]
    assert obs.runs[0].summary_job_conclusion == "success"
    assert obs.runs[0].run_conclusion == "failure"


def test_no_runs_is_an_observed_empty_list() -> None:
    obs = _observe(_Reader(runs=[], jobs={}))
    assert obs.read_ok is True
    assert obs.runs == ()


def test_run_without_summary_job_carries_none() -> None:
    obs = _observe(_Reader(runs=[_run(5)], jobs={5: [_job("Lint")]}))
    assert obs.read_ok is True
    assert obs.runs[0].summary_job_status is None
    assert obs.runs[0].summary_job_conclusion is None


@pytest.mark.parametrize("fail_on", ["runs", "jobs"])
def test_a_failed_read_is_read_ok_false_never_empty(fail_on: str) -> None:
    obs = _observe(_Reader(runs=[_run(5)], jobs={}, fail_on=fail_on))
    assert obs.read_ok is False
    assert obs.read_error
    assert "HTTP" in obs.read_error
    assert obs.runs == ()


# ---------------------------------------------------------------------------
# The GitHub REST binding, driven through an injected fetch
# ---------------------------------------------------------------------------


def _github(pages: dict[str, dict[str, object]]) -> HandlerMergeGroupRunReadGithub:
    def fetch(path: str) -> dict[str, object]:
        for prefix, payload in pages.items():
            if path.startswith(prefix):
                if path.endswith("&page=1"):
                    return payload
                # Every later page is empty, as GitHub returns past the end.
                key = "jobs" if "jobs" in payload else "workflow_runs"
                return {"total_count": payload["total_count"], key: []}
        raise AssertionError(f"unexpected read {path}")

    return HandlerMergeGroupRunReadGithub(token="t", fetch=fetch)


def _run_row(path: str) -> dict[str, object]:
    return {
        "id": 36409772437,
        "run_attempt": 1,
        "event": "merge_group",
        "head_sha": SHA,
        "head_branch": "gh-readonly-queue/dev/pr-4235-31258513",
        "path": path,
        "status": "completed",
        "conclusion": "failure",
    }


@pytest.mark.parametrize(
    "path", [".github/workflows/ci.yml", ".github/workflows/ci.yml@refs/heads/dev"]
)
def test_github_binding_reads_the_run_path_with_or_without_a_ref(path: str) -> None:
    reader = _github(
        {
            f"/repos/{REPO}/actions/runs?": {
                "total_count": 1,
                "workflow_runs": [_run_row(path)],
            },
            f"/repos/{REPO}/actions/runs/36409772437/jobs": {
                "total_count": 1,
                "jobs": [
                    {
                        "name": "CI Summary",
                        "status": "completed",
                        "conclusion": "success",
                    }
                ],
            },
        }
    )
    obs = _observe(reader)  # type: ignore[arg-type]
    assert obs.read_ok is True
    assert [r.summary_job_conclusion for r in obs.runs] == ["success"]


def test_github_binding_refuses_a_short_page() -> None:
    reader = _github(
        {
            f"/repos/{REPO}/actions/runs?": {
                "total_count": 2,
                "workflow_runs": [_run_row(".github/workflows/ci.yml")],
            },
        }
    )
    with pytest.raises(MergeGroupReadError, match="refusing a short list"):
        reader.list_merge_group_runs(REPO, SHA)


def test_github_binding_turns_an_unknown_status_into_a_failed_read() -> None:
    row = _run_row(".github/workflows/ci.yml")
    row["status"] = "teleported"
    reader = _github(
        {f"/repos/{REPO}/actions/runs?": {"total_count": 1, "workflow_runs": [row]}}
    )
    obs = _observe(reader)  # type: ignore[arg-type]
    assert obs.read_ok is False
