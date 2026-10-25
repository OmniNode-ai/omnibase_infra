# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: the base-branch judge shares the reviewer's model environment.

The read-only on-reply gate requests judgement; the workflow_run judge uses
trusted code, the reviewer's overlay/install steps and the onexbot App token.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, cast

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
KEY = "${{ secrets.LOCAL_LLM_SHARED_SECRET }}"
PROVIDER_KEY = "HOSTILE_REVIEW_ANTHROPIC_API_KEY"


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
    job: dict[str, Any] = _workflow("hostile-review-judge.yml")["jobs"][
        "hostile-review-judge"
    ]
    return job


def _gate_job() -> dict[str, Any]:
    return _on_reply()["jobs"]["hostile-review-thread-gate"]


def _gate_steps() -> dict[str, dict[str, Any]]:
    return {step["name"]: step for step in _gate_job()["steps"]}


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
    assert names.index("Find the judge request") == 1
    assert names.index("Read the judge request as data") == 3
    assert names.index("Fetch the PR's commits as data") == 4
    assert names.index("Find rejections awaiting the review bot") == 5
    assert list(_gate_steps()).index("Find rejections awaiting the review bot") == 1
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
    # The judge step holds the signing secret but no GitHub token: it posts
    # nothing.
    assert "GITHUB_TOKEN" not in judge["env"]


def test_the_verdicts_are_posted_whatever_the_judge_did_then_gated() -> None:
    names = [step["name"] for step in _job()["steps"]]
    post = _steps()[POST]
    assert names.index(JUDGE) < names.index(POST)
    assert GATE not in names
    assert post["if"] == "${{ !cancelled() && " + PENDING + " }}"
    assert post["run"] == (
        'python3 scripts/ci/check_hostile_review_threads.py --post-verdicts "$JUDGE_RESULT_PATH"'
    )
    assert post["env"]["GITHUB_TOKEN"] == "${{ steps.verdict-token.outputs.token }}"
    assert (
        post["env"]["JUDGE_RESULT_PATH"] == _steps()[JUDGE]["env"]["JUDGE_RESULT_PATH"]
    )
    assert (
        _steps()["Find rejections awaiting the review bot"]["env"]["JUDGE_PENDING_PATH"]
        == _steps()[JUDGE]["env"]["JUDGE_PENDING_PATH"]
    )


def test_the_gate_runs_whatever_the_judge_did_and_reads_its_status() -> None:
    gate = _gate_steps()[GATE]
    assert gate["if"] == "${{ !cancelled() }}"
    assert gate["run"] == "python3 scripts/ci/check_hostile_review_threads.py"
    assert (
        gate["env"]["JUDGE_STATUS_PATH"] == _steps()[POST]["env"]["JUDGE_STATUS_PATH"]
    )
    uploaded = _gate_steps()["Upload the disposition record"]["with"]["path"]
    assert "hostile-review-judge-status.json" in uploaded
    assert "hostile-review-judgements.json" not in uploaded
    judged = _steps()["Upload the judgements"]["with"]["path"]
    assert "hostile-review-judge-status.json" in judged
    assert "hostile-review-judgements.json" in judged


def test_the_model_key_reaches_the_judge_step_alone() -> None:
    holders = [
        name
        for name, step in _steps().items()
        if KEY in yaml.safe_dump(step.get("env", {}))
    ]
    assert holders == [JUDGE]


def test_no_provider_key_reaches_any_step() -> None:
    # OMN-20974, OMN-21350: a Claude judge goes through the review gateway by
    # reference; no Actions secret carries a provider key to any step.
    text = (WORKFLOWS / "hostile-review-judge.yml").read_text()
    assert PROVIDER_KEY not in text
    for step in _steps().values():
        env = step.get("env", {})
        assert "ANTHROPIC_API_KEY" not in env
        assert "REVIEW_ANTHROPIC_API_KEY" not in env


def test_no_new_secret_is_read() -> None:
    reviewer_secrets = set(
        re.findall(
            r"secrets\.([A-Z_]+)", (WORKFLOWS / "hostile-reviewer.yml").read_text()
        )
    )
    judge_secrets = set(
        re.findall(
            r"secrets\.([A-Z_]+)",
            (WORKFLOWS / "hostile-review-judge.yml").read_text(),
        )
    )
    assert judge_secrets <= reviewer_secrets


def test_the_gate_is_read_only_and_the_judge_has_only_read_permissions() -> None:
    assert _on_reply()["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert _gate_job()["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert _workflow("hostile-review-judge.yml")["permissions"] == {}
    assert _job()["permissions"] == {"actions": "read", "contents": "read"}


def test_the_write_token_is_not_persisted_in_the_checkout() -> None:
    checkout = _gate_steps()["Checkout (gate script and PR history)"]
    assert checkout["with"]["persist-credentials"] is False


def test_the_job_is_hosted_and_keeps_the_required_context_name() -> None:
    assert _job()["runs-on"] == "ubuntu-latest"
    assert _job()["name"] == "Hostile Review Judge"
    assert _gate_job()["runs-on"] == "ubuntu-latest"
    assert _gate_job()["name"] == "Hostile Review Thread Gate"


def test_every_action_is_pinned_by_commit() -> None:
    for step in _job()["steps"] + _gate_job()["steps"]:
        uses = step.get("uses")
        if uses:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", uses), uses


def test_missing_request_skips_download_and_read() -> None:
    steps = _steps()
    for name in ("Download the judge request", "Read the judge request as data"):
        assert steps[name]["if"] == "steps.has-request.outputs.present == 'true'"
    find = steps["Find the judge request"]
    assert find["id"] == "has-request"
    assert find["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert find["run"].splitlines()[0] == "set -euo pipefail"
    assert (
        "repos/${{ github.repository }}/actions/runs/${{ github.event.workflow_run.id }}/artifacts"
        in find["run"]
    )


@pytest.mark.parametrize(
    ("artifacts", "expected"),
    [
        ([], "false"),
        ([{"name": "hostile-review-judge-request", "expired": True}], "false"),
        ([{"name": "hostile-review-judge-request-other", "expired": False}], "false"),
        ([{"name": "hostile-review-judge-request", "expired": False}], "true"),
    ],
)
def test_find_request_requires_an_exact_nonexpired_artifact(
    tmp_path: Path, artifacts: list[dict[str, Any]], expected: str
) -> None:
    # gh --paginate --slurp emits an array of pages. Include a different
    # artifact on page one to check that later pages are inspected too.
    pages = [
        {"artifacts": [{"name": "other", "expired": False}]},
        {"artifacts": artifacts},
    ]
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/bin/bash\nprintf '%s\\n' " + shlex.quote(json.dumps(pages)) + "\n"
    )
    gh.chmod(0o755)
    output = tmp_path / "output"
    run = _steps()["Find the judge request"]["run"]
    run = run.replace("${{ github.repository }}", "owner/repo").replace(
        "${{ github.event.workflow_run.id }}", "123"
    )
    result = subprocess.run(
        ["bash", "-c", run],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "GH_TOKEN": "t",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"present={expected}\n"


def test_find_request_api_failure_fails_closed(tmp_path: Path) -> None:
    gh = tmp_path / "gh"
    # Even valid output must not hide a failed API call in the pipeline.
    gh.write_text("#!/bin/bash\nprintf '%s\\n' '[{\"artifacts\": []}]'\nexit 1\n")
    gh.chmod(0o755)
    output = tmp_path / "output"
    run = _steps()["Find the judge request"]["run"]
    run = run.replace("${{ github.repository }}", "owner/repo").replace(
        "${{ github.event.workflow_run.id }}", "123"
    )
    result = subprocess.run(
        ["bash", "-c", run],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "GH_TOKEN": "t",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not output.exists()


def _find_request_output(tmp_path: Path, pages: list[dict[str, Any]]) -> str:
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/bin/bash\nprintf '%s\\n' " + shlex.quote(json.dumps(pages)) + "\n"
    )
    gh.chmod(0o755)
    output = tmp_path / "output"
    run = _steps()["Find the judge request"]["run"]
    run = run.replace("${{ github.repository }}", "owner/repo").replace(
        "${{ github.event.workflow_run.id }}", "40189445988"
    )
    result = subprocess.run(
        ["bash", "-c", run],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "GH_TOKEN": "t",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output.read_text()


@pytest.mark.live_contact(
    "tests/fixtures/omn20910/infra-pr5237-run-40189445988-on-reply-artifacts.json"
)
def test_the_recorded_on_reply_run_carries_a_request_the_judge_finds(
    tmp_path: Path, recorded_response: dict[str, object]
) -> None:
    """#5237's on-reply run 40189445988 uploaded a judge request.

    The step reads the artifact listing GitHub returned for that run and finds
    the request. The same listing without that artifact (an on-reply run that
    uploaded the disposition record alone) makes the judge skip.
    """
    page = cast("dict[str, Any]", recorded_response["response"])
    names = [artifact["name"] for artifact in page["artifacts"]]
    assert "hostile-review-judge-request" in names
    with_request = tmp_path / "with-request"
    with_request.mkdir()
    assert _find_request_output(with_request, [page]) == "present=true\n"

    dispositions_only = {
        "total_count": 1,
        "artifacts": [
            artifact
            for artifact in page["artifacts"]
            if artifact["name"] != "hostile-review-judge-request"
        ],
    }
    without_request = tmp_path / "without-request"
    without_request.mkdir()
    assert _find_request_output(without_request, [dispositions_only]) == (
        "present=false\n"
    )
