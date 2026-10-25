# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: PR-head code cannot receive the judge's secrets or write scope."""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
GATE = "hostile-review-thread-gate.yml"
JUDGE = "hostile-review-judge.yml"
PR_HEAD_EVENTS = {"pull_request", "pull_request_review", "pull_request_review_comment"}
SECRETS = ("ONEXBOT_APP_ID", "ONEXBOT_APP_PRIVATE_KEY", "LOCAL_LLM_SHARED_SECRET")
PENDING = "fromJSON(steps.pending.outputs.count || '0') > 0"


def _triggers(workflow: dict[str, Any]) -> set[str]:
    # PyYAML's YAML 1.1 resolver reads the Actions key `on` as True.
    on = workflow.get("on", workflow.get(True, {}))
    return {on} if isinstance(on, str) else set(on)


def pr_head_violations(workflow: dict) -> list[str]:
    """Find secret-bearing steps and effective write scope on PR-head events.

    Inspect whole steps (including action inputs and run bodies) and both
    inherited env mappings: moving a secret out of a step cannot hide it.
    """
    if not PR_HEAD_EVENTS.intersection(_triggers(workflow)):
        return []
    violations: list[str] = []
    for job_id, job in workflow.get("jobs", {}).items():
        permissions = job.get("permissions", workflow.get("permissions", {}))
        if permissions == "write-all":
            violations.append(f"{job_id}: write-all permission")
        elif isinstance(permissions, dict):
            for scope, grant in permissions.items():
                if grant == "write":
                    violations.append(f"{job_id}: {scope}: write")
        inherited = [workflow.get("env", {}), job.get("env", {})]
        for index, step in enumerate(job.get("steps", [])):
            serialized = yaml.safe_dump([*inherited, step])
            for secret in SECRETS:
                if f"secrets.{secret}" in serialized:
                    violations.append(f"{job_id}/{step.get('name', index)}: {secret}")
    return violations


@functools.cache
def _workflows() -> dict[str, dict[str, Any]]:
    """Parse every workflow, so the boundary checks use the repository's YAML."""
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
    }


def _judge_job() -> dict[str, Any]:
    return _workflows()[JUDGE]["jobs"]["hostile-review-judge"]


def test_the_on_reply_gate_holds_no_secret_or_write_scope() -> None:
    gate = _workflows()[GATE]
    assert pr_head_violations(gate) == []
    assert gate["permissions"] == {"contents": "read", "pull-requests": "read"}
    job = gate["jobs"]["hostile-review-thread-gate"]
    assert job["permissions"] == gate["permissions"]
    assert job["timeout-minutes"] == 10
    assert _triggers(gate) == {"pull_request_review", "pull_request_review_comment"}
    text = yaml.safe_dump(gate)
    assert set(re.findall(r"secrets\.([A-Z_]+)", text)) == {"GITHUB_TOKEN"}
    assert "vars." not in text


@pytest.mark.parametrize(
    ("bad_yaml", "expected"),
    [
        (
            """
on: pull_request_review_comment
jobs:
  gate:
    steps:
      - env:
          LOCAL_LLM_SHARED_SECRET: ${{ secrets.LOCAL_LLM_SHARED_SECRET }}
        run: python3 scripts/ci/check_hostile_review_threads.py
""",
            "LOCAL_LLM_SHARED_SECRET",
        ),
        (
            """
on: pull_request_review
jobs:
  gate:
    env:
      APP_KEY: ${{ secrets.ONEXBOT_APP_PRIVATE_KEY }}
    steps:
      - run: python3 scripts/ci/check_hostile_review_threads.py
""",
            "ONEXBOT_APP_PRIVATE_KEY",
        ),
        (
            """
on: [pull_request]
jobs:
  gate:
    permissions:
      pull-requests: write
    steps: []
""",
            "pull-requests: write",
        ),
        (
            """
on: pull_request_review_comment
env:
  APP_ID: ${{ secrets.ONEXBOT_APP_ID }}
jobs:
  gate:
    steps:
      - run: python3 scripts/ci/check_hostile_review_threads.py
""",
            "ONEXBOT_APP_ID",
        ),
    ],
)
def test_positive_controls_detect_secrets_and_write_scope(
    bad_yaml: str, expected: str
) -> None:
    assert any(
        expected in item for item in pr_head_violations(yaml.safe_load(bad_yaml))
    )


def test_the_original_head_shape_violates_the_boundary() -> None:
    # Minimal excerpt of the pre-split head, embedded rather than read from git.
    original = yaml.safe_load("""
on:
  pull_request_review_comment:
    types: [created, edited, deleted]
  pull_request_review:
    types: [submitted, edited, dismissed]
permissions:
  contents: read
  pull-requests: read
jobs:
  hostile-review-thread-gate:
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
      - name: Mint a read-only CI overlay token
        with:
          app-id: ${{ secrets.ONEXBOT_APP_ID }}
          private-key: ${{ secrets.ONEXBOT_APP_PRIVATE_KEY }}
      - name: The review bot judges the rejections
        env:
          LOCAL_LLM_SHARED_SECRET: ${{ secrets.LOCAL_LLM_SHARED_SECRET }}
        run: uv run python -m omniintelligence.review_pairing.cli_review judge-rejections
      - run: python3 scripts/ci/check_hostile_review_threads.py --post-verdicts result.json
""")
    violations = pr_head_violations(original)
    assert len(violations) == 4
    for expected in (*SECRETS, "pull-requests: write"):
        assert any(expected in item for item in violations)


def test_the_judge_runs_only_on_workflow_run_from_the_base_branch() -> None:
    workflow = _workflows()[JUDGE]
    assert _triggers(workflow) == {"workflow_run"}
    on = workflow.get("on", workflow.get(True))
    assert on["workflow_run"] == {
        "workflows": ["Hostile Review Thread Gate (on reply)"],
        "types": ["completed"],
    }
    assert workflow["permissions"] == {}
    job = _judge_job()
    assert job["permissions"] == {"actions": "read", "contents": "read"}
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 30
    assert job["if"].split() == [
        "(github.event.workflow_run.event",
        "==",
        "'pull_request_review_comment'",
        "||",
        "github.event.workflow_run.event",
        "==",
        "'pull_request_review')",
        "&&",
        "github.event.workflow_run.head_repository.full_name",
        "==",
        "github.repository",
    ]
    assert workflow["concurrency"] == {
        "group": "hostile-review-judge-${{ github.event.workflow_run.head_branch }}",
        "cancel-in-progress": False,
    }
    checkouts = []
    for step in job["steps"]:
        with_ = step.get("with", {})
        if step.get("uses", "").startswith("actions/checkout@"):
            checkouts.append(step)
            ref = with_.get("ref")
            assert "workflow_run.head_sha" not in str(ref)
            assert "head_branch" not in str(ref)
            assert "refs/pull/" not in str(ref)
            if with_.get("repository"):
                assert with_["repository"] == (
                    "${{ github.repository_owner }}/${{ steps.ci-overlay.outputs.name }}"
                )
                assert ref is None
            else:
                assert ref == "${{ github.event.repository.default_branch }}"
            assert with_["persist-credentials"] is False
        run = step.get("run", "")
        assert not re.search(r"git\s+(?:checkout|switch|worktree).*refs/pull/", run)
        if "refs/pull/" in yaml.safe_dump([with_, run]):
            assert step["name"] == "Fetch the PR's commits as data"
            assert run == (
                "git fetch --no-tags --filter=blob:none origin "
                '"+refs/pull/${PR_NUMBER}/head:refs/remotes/origin/pr-head"'
            )
    assert len(checkouts) == 2
    assert checkouts[0]["name"] == "Checkout the base branch (trusted code only)"


def test_the_secrets_reach_only_their_steps() -> None:
    workflow = _workflows()[JUDGE]
    job = _judge_job()
    inherited = yaml.safe_dump([workflow.get("env", {}), job.get("env", {})])
    assert all(f"secrets.{secret}" not in inherited for secret in SECRETS)
    holders = {
        secret: [
            step["name"]
            for step in job["steps"]
            if f"secrets.{secret}" in yaml.safe_dump(step)
        ]
        for secret in SECRETS
    }
    mint_steps = [
        "Mint a read-only CI overlay token",
        "Mint the onexbot App token for the verdict replies",
    ]
    assert holders["ONEXBOT_APP_PRIVATE_KEY"] == mint_steps
    assert holders["ONEXBOT_APP_ID"] == mint_steps
    assert holders["LOCAL_LLM_SHARED_SECRET"] == [
        "The review bot judges the rejections"
    ]
    for step in job["steps"]:
        if step["name"] == "The review bot judges the rejections":
            break
        if "scripts/ci/" in step.get("run", ""):
            assert all(
                f"secrets.{secret}" not in yaml.safe_dump(step) for secret in SECRETS
            )
    verdict_token = next(
        step for step in job["steps"] if step.get("id") == "verdict-token"
    )
    assert verdict_token["if"] == "${{ !cancelled() && " + PENDING + " }}"
    assert verdict_token["with"]["permission-pull-requests"] == "write"
    assert (
        verdict_token["with"]["repositories"] == "${{ github.event.repository.name }}"
    )


def test_the_request_crosses_the_boundary_as_validated_data() -> None:
    steps = {step["name"]: step for step in _judge_job()["steps"]}
    download = steps["Download the judge request"]["with"]
    assert download == {
        "name": "hostile-review-judge-request",
        "run-id": "${{ github.event.workflow_run.id }}",
        "github-token": "${{ github.token }}",
        "path": "${{ runner.temp }}/hostile-review-judge-request",
    }
    read = steps["Read the judge request as data"]
    assert read["id"] == "request"
    assert (
        read["env"]["WORKFLOW_RUN_HEAD_SHA"]
        == "${{ github.event.workflow_run.head_sha }}"
    )
    assert read["run"] == (
        "python3 scripts/ci/check_hostile_review_threads.py --read-request "
        '"${{ runner.temp }}/hostile-review-judge-request/request.json"'
    )
    for name in (
        "Fetch the PR's commits as data",
        "Find rejections awaiting the review bot",
    ):
        assert steps[name]["if"] == "fromJSON(steps.request.outputs.count || '0') > 0"
        assert steps[name]["env"]["PR_NUMBER"] == "${{ steps.request.outputs.pr }}"
    assert (
        steps["Find rejections awaiting the review bot"]["env"]["HEAD_SHA"]
        == "${{ steps.request.outputs.head_sha }}"
    )
    assert (
        steps["Find rejections awaiting the review bot"]["env"][
            "HOSTILE_REVIEW_IS_FORK"
        ]
        == "false"
    )
