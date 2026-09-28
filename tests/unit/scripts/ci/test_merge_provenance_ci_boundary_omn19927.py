# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The CI boundary of merge provenance, and its wiring into ``ci.yml`` (OMN-19927).

``scripts/ci/merge_provenance.py`` is the I/O boundary only: it builds the
observe effect's reader, runs the effect and then the compute node, and writes
the result. Every decision is the compute handler's. The workflow steps carry
no provenance logic: one command per step, plus reading its output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_workflow_job_fact import (
    ModelWorkflowJobFact,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.models.model_workflow_run_fact import (
    ModelWorkflowRunFact,
)
from scripts.ci import merge_provenance

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_YML = REPO_ROOT / ".github/workflows/ci.yml"
REPO = "OmniNode-ai/omnibase_infra"
SHA_E263EEE = "e263eee036a8e338e6b39ea746d7e59bf91602cb"
SHA_QUEUED = "7b1f6fb690d65e86c804e011ef80b8485a4d070e"


class _Reader:
    def __init__(
        self, runs: list[ModelWorkflowRunFact], jobs: list[ModelWorkflowJobFact]
    ) -> None:
        self._runs = runs
        self._jobs = jobs

    def list_merge_group_runs(
        self, repository: str, head_sha: str
    ) -> list[ModelWorkflowRunFact]:
        return [r for r in self._runs if r.head_sha == head_sha]

    def list_latest_attempt_jobs(
        self, repository: str, run_id: int
    ) -> list[ModelWorkflowJobFact]:
        return self._jobs


def _outputs(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


def test_unvalidated_sha_forces_the_full_suite(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    result_json = tmp_path / "provenance.json"
    merge_provenance.evaluate_and_write(
        repository=REPO,
        sha=SHA_E263EEE,
        reader=_Reader(runs=[], jobs=[]),
        github_output=out,
        result_json=result_json,
    )
    outputs = _outputs(out)
    assert outputs["verdict"] == "UNVALIDATED"
    assert outputs["force_full_suite"] == "true"
    assert json.loads(result_json.read_text())["verdict"] == "UNVALIDATED"


def test_validated_sha_keeps_smart_selection(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    run = ModelWorkflowRunFact(
        run_id=36409772437,
        run_attempt=1,
        event="merge_group",
        head_sha=SHA_QUEUED,
        head_branch="gh-readonly-queue/dev/pr-4235-31258513",
        workflow_path=".github/workflows/ci.yml",
        status="completed",
        conclusion="failure",
    )
    merge_provenance.evaluate_and_write(
        repository=REPO,
        sha=SHA_QUEUED,
        reader=_Reader(
            runs=[run],
            jobs=[
                ModelWorkflowJobFact(
                    name="CI Summary", status="completed", conclusion="success"
                )
            ],
        ),
        github_output=out,
        result_json=tmp_path / "provenance.json",
    )
    outputs = _outputs(out)
    assert outputs["verdict"] == "VALIDATED"
    assert outputs["force_full_suite"] == "false"


def test_missing_token_is_undecidable_and_forces_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    out = tmp_path / "gh_output"
    rc = merge_provenance.main(
        [
            "--repository",
            REPO,
            "--sha",
            SHA_E263EEE,
            "--github-output",
            str(out),
            "--result-json",
            str(tmp_path / "provenance.json"),
        ]
    )
    assert rc == 0
    outputs = _outputs(out)
    assert outputs["verdict"] == "UNDECIDABLE"
    assert outputs["force_full_suite"] == "true"


# ---------------------------------------------------------------------------
# ci.yml wiring
# ---------------------------------------------------------------------------


def _detect_changes_job() -> dict[str, object]:
    workflow = yaml.safe_load(CI_YML.read_text())
    job = workflow["jobs"]["detect-changes"]
    assert isinstance(job, dict)
    return job


def _step(job: dict[str, object], step_id: str) -> dict[str, Any]:
    steps = job["steps"]
    assert isinstance(steps, list)
    matches: list[dict[str, Any]] = [s for s in steps if s.get("id") == step_id]
    assert len(matches) == 1, f"expected one step with id {step_id!r}"
    return matches[0]


def test_provenance_step_runs_on_every_non_queue_non_pr_event() -> None:
    step = _step(_detect_changes_job(), "provenance")
    condition = step["if"]
    assert "github.event_name == 'push'" in condition
    assert "github.event_name == 'workflow_dispatch'" in condition
    assert "pull_request" not in condition
    assert "merge_group" not in condition


def test_provenance_step_is_one_call_with_no_logic() -> None:
    step = _step(_detect_changes_job(), "provenance")
    commands = [
        line.strip()
        for line in step["run"].replace("\\\n", " ").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(commands) == 1, commands
    assert commands[0].startswith("uv run python -m scripts.ci.merge_provenance ")
    assert "${{ github.sha }}" in commands[0]
    for token in (" if ", "jq", "grep", "&&", "||"):
        assert token not in f" {commands[0]} "


def test_selection_step_forces_full_on_anything_but_an_explicit_false() -> None:
    """A provenance step that crashed or wrote nothing must still force full."""
    step = _step(_detect_changes_job(), "detect")
    env = step["env"]
    assert "FORCE_FULL" in env
    expr = env["FORCE_FULL"]
    assert "steps.provenance.outputs.force_full_suite != 'false'" in expr
    assert "github.event_name == 'push'" in expr
    assert "github.event_name == 'workflow_dispatch'" in expr
    assert "--force-full-suite unvalidated_push" in step["run"]
    assert '"${force_args[@]}"' in step["run"]


def test_detect_changes_can_read_actions_runs() -> None:
    permissions = _detect_changes_job()["permissions"]
    assert permissions == {"actions": "read", "contents": "read"}
