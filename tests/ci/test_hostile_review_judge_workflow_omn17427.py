# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: how the review bot is wired into the on-reply thread gate.

The judge runs only inside ``hostile-review-thread-gate.yml``, under the
workflow's own token, with the reviewer's own overlay read and model
environment. These tests hold the wiring that the judge's guarantees rest on:

- the overlay and omniintelligence steps are the reviewer job's own steps,
  byte for byte apart from their ``if:``, so the judge cannot drift onto a
  parallel model path;
- nothing is installed and no model is called unless a rejection awaits a
  judgement, so a PR with no thread passes exactly as before;
- the gate step runs whatever the judge did, and reads its status;
- the model key reaches the judge step alone, the write permission is the
  job's alone and is not persisted in the checkout, and no new secret appears.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
PENDING = "fromJSON(steps.pending.outputs.count || '0') > 0"

SHARED_STEPS = (
    "Resolve the CI overlay",
    "Mint a read-only CI overlay token",
    "Fetch the CI overlay",
    "Place the CI overlay outside the workspace",
    "Clone omniintelligence",
    "Clone omnibase_core (omniintelligence dependency)",
    "Clone omnibase_compat",
    "Set up Python 3.12",
    "Install omniintelligence dependencies",
)
JUDGE = "The review bot judges the rejections"
GATE = "Check hostile-reviewer threads and their dispositions"
POST = "Post the review bot's verdicts"
KEY = "${{ secrets.HOSTILE_REVIEW_ANTHROPIC_API_KEY }}"


# The workflows are read on first use, not at import: a missing or malformed
# file fails each test that reads it, by name, instead of the whole module.
@functools.cache
def _workflow(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / name).read_text())
    return loaded


def _on_reply() -> dict[str, Any]:
    return _workflow("hostile-review-thread-gate.yml")


def _reviewer() -> dict[str, Any]:
    return _workflow("hostile-reviewer.yml")


def _job() -> dict[str, Any]:
    job: dict[str, Any] = _on_reply()["jobs"]["hostile-review-thread-gate"]
    return job


def _steps() -> dict[str, dict[str, Any]]:
    return {step["name"]: step for step in _job()["steps"]}


def _reviewer_steps() -> dict[str, dict[str, Any]]:
    return {
        step["name"]: step for step in _reviewer()["jobs"]["hostile-review"]["steps"]
    }


def _without_if(step: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in step.items() if k != "if"}


@pytest.mark.parametrize("name", SHARED_STEPS)
def test_the_judge_reads_models_exactly_as_the_reviewer_does(name: str) -> None:
    assert _without_if(_steps()[name]) == _without_if(_reviewer_steps()[name])


def test_the_judge_job_names_the_same_voters_overlay_as_the_reviewer() -> None:
    voters = _job()["env"]["REVIEW_VOTERS_OVERLAY"]
    assert (
        voters == _reviewer()["jobs"]["hostile-review"]["env"]["REVIEW_VOTERS_OVERLAY"]
    )
    judge = _job()["env"]["REVIEW_JUDGE_OVERLAY"]
    assert judge == voters.rsplit("/", 1)[0] + "/rejection_judge.yaml"


@pytest.mark.parametrize("name", [*SHARED_STEPS, "Install uv", JUDGE])
def test_nothing_is_installed_or_called_without_a_pending_rejection(
    name: str,
) -> None:
    assert str(_steps()[name]["if"]).startswith(PENDING)


def test_the_pending_count_runs_first_and_stdlib_only() -> None:
    names = [step["name"] for step in _job()["steps"]]
    assert names.index("Find rejections awaiting the review bot") == 1
    pending = _steps()["Find rejections awaiting the review bot"]
    assert pending["id"] == "pending"
    assert pending["run"] == (
        'python3 scripts/ci/check_hostile_review_threads.py --pending "$JUDGE_PENDING_PATH"'
    )


def test_the_judge_is_omniintelligences_in_the_reviewers_environment() -> None:
    names = [step["name"] for step in _job()["steps"]]
    assert names.index("Install uv") < names.index("Set up Python 3.12")
    assert names.index("Install omniintelligence dependencies") < names.index(JUDGE)
    judge = _steps()[JUDGE]
    assert judge["working-directory"] == "../omniintelligence"
    assert judge["run"].split() == [
        "uv",
        "run",
        "python",
        "-m",
        "omniintelligence.review_pairing.cli_review",
        "judge-rejections",
        "\\",
        "--pending",
        '"$JUDGE_PENDING_PATH"',
        "\\",
        "--voters-overlay",
        '"$REVIEW_VOTERS_OVERLAY"',
        "\\",
        "--judge-overlay",
        '"$REVIEW_JUDGE_OVERLAY"',
        "\\",
        "--output",
        '"$JUDGE_RESULT_PATH"',
    ]
    # The judge step holds the model key but no GitHub token: it posts nothing.
    assert "GITHUB_TOKEN" not in judge["env"]


def test_the_verdicts_are_posted_whatever_the_judge_did_then_gated() -> None:
    names = [step["name"] for step in _job()["steps"]]
    post = _steps()[POST]
    assert names.index(JUDGE) < names.index(POST) < names.index(GATE)
    assert post["if"] == "${{ !cancelled() && " + PENDING + " }}"
    assert post["run"] == (
        'python3 scripts/ci/check_hostile_review_threads.py --post-verdicts "$JUDGE_RESULT_PATH"'
    )
    assert post["env"]["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert (
        post["env"]["JUDGE_RESULT_PATH"] == _steps()[JUDGE]["env"]["JUDGE_RESULT_PATH"]
    )
    assert (
        _steps()["Find rejections awaiting the review bot"]["env"]["JUDGE_PENDING_PATH"]
        == _steps()[JUDGE]["env"]["JUDGE_PENDING_PATH"]
    )


def test_the_gate_runs_whatever_the_judge_did_and_reads_its_status() -> None:
    gate = _steps()[GATE]
    assert gate["if"] == "${{ !cancelled() }}"
    assert gate["run"] == "python3 scripts/ci/check_hostile_review_threads.py"
    assert (
        gate["env"]["JUDGE_STATUS_PATH"] == _steps()[POST]["env"]["JUDGE_STATUS_PATH"]
    )
    uploaded = _steps()["Upload the disposition record"]["with"]["path"]
    assert "hostile-review-judge-status.json" in uploaded
    assert "hostile-review-judgements.json" in uploaded


def test_the_model_key_reaches_the_judge_step_alone() -> None:
    holders = [
        name
        for name, step in _steps().items()
        if KEY in yaml.safe_dump(step.get("env", {}))
    ]
    assert holders == [JUDGE]


def test_no_new_secret_is_read() -> None:
    reviewer_secrets = set(
        re.findall(
            r"secrets\.([A-Z_]+)", (WORKFLOWS / "hostile-reviewer.yml").read_text()
        )
    )
    judge_secrets = set(
        re.findall(
            r"secrets\.([A-Z_]+)",
            (WORKFLOWS / "hostile-review-thread-gate.yml").read_text(),
        )
    )
    assert judge_secrets <= reviewer_secrets


def test_only_the_job_may_write_and_only_pull_requests() -> None:
    assert _on_reply()["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert _job()["permissions"] == {"contents": "read", "pull-requests": "write"}


def test_the_write_token_is_not_persisted_in_the_checkout() -> None:
    checkout = _steps()["Checkout (gate script and PR history)"]
    assert checkout["with"]["persist-credentials"] is False


def test_the_job_is_hosted_and_keeps_the_required_context_name() -> None:
    assert _job()["runs-on"] == "ubuntu-latest"
    assert _job()["name"] == "Hostile Review Thread Gate"


def test_every_action_is_pinned_by_commit() -> None:
    for step in _job()["steps"]:
        uses = step.get("uses")
        if uses:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", uses), uses
