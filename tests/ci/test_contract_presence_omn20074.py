# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: refuse human PR title citations without contracts at the head."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import EXPECTED_EXTERNAL_CONTEXTS

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "contract-validation.yml"
HUMAN_PULL_REQUEST = (
    "github.event_name == 'pull_request' && "
    "github.event.pull_request.user.type != 'Bot'"
)


def _workflow() -> dict[str | bool, Any]:
    workflow: dict[str | bool, Any] = yaml.safe_load(
        WORKFLOW_PATH.read_text(encoding="utf-8")
    )
    return workflow


def _run_presence(tmp_path: Path, title: str) -> subprocess.CompletedProcess[str]:
    steps = _workflow()["jobs"]["contract-validation"]["steps"]
    script = next(
        step["run"] for step in steps if step.get("id") == "contract_presence"
    )
    return subprocess.run(
        ["bash", "-c", script],
        env={
            "PATH": os.environ["PATH"],
            "PR_TITLE": title,
            "HEAD_TREE": str(tmp_path / "head"),
        },
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )


def test_presence_steps_run_before_validation_for_human_pull_requests() -> None:
    workflow = _workflow()
    job = workflow["jobs"]["contract-validation"]
    assert "if" not in job
    steps = job["steps"]
    ids = [step.get("id") for step in steps]
    assert (
        ids.index("resolve-branch")
        < ids.index("contract_head")
        < ids.index("contract_presence")
        < ids.index("validate-contract")
    )
    checkout = steps[ids.index("contract_head")]
    presence = steps[ids.index("contract_presence")]
    for step in (checkout, presence):
        assert step["if"] == HUMAN_PULL_REQUEST
    assert checkout["uses"] == (
        "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0"
    )
    assert checkout["with"] == {
        "ref": "${{ github.event.pull_request.head.sha }}",
        "path": ".contract-presence/head",
        "sparse-checkout": "contracts",
        "persist-credentials": False,
        "fetch-depth": 1,
    }
    assert presence["env"] == {
        "PR_TITLE": "${{ github.event.pull_request.title }}",
        "HEAD_TREE": ".contract-presence/head",
    }
    assert presence["shell"] == "bash"
    # PyYAML 1.1 maps the bare on: key to True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    assert triggers["pull_request"]["branches"] == ["main", "dev", "develop"]
    assert triggers["pull_request"]["types"] == [
        "opened",
        "synchronize",
        "reopened",
        "edited",
    ]


def test_presence_refuses_a_cited_ticket_with_no_contract(tmp_path: Path) -> None:
    result = _run_presence(tmp_path, "fix(OMN-123): x")
    assert result.returncode == 1
    assert (
        "::error::pull request cites OMN-123 but carries no contracts/OMN-123.yaml"
        in result.stdout
    )
    assert (
        "::notice::Include contracts/OMN-123.yaml at this PR head; "
        "a later companion merge cannot supply evidence for this head." in result.stdout
    )


def test_presence_admits_a_cited_ticket_whose_contract_is_at_the_head(
    tmp_path: Path,
) -> None:
    contracts = tmp_path / "head" / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "OMN-123.yaml").write_text("", encoding="utf-8")
    result = _run_presence(tmp_path, "fix(OMN-123): x")
    assert result.returncode == 0
    assert (
        "::notice::Contract presence: every cited ticket carries "
        "contracts/<ticket>.yaml at the head." in result.stdout
    )


def test_presence_names_every_missing_ticket(tmp_path: Path) -> None:
    title = "fix(OMN-1, OMN-2): x"
    result = _run_presence(tmp_path, title)
    assert result.returncode == 1
    for ticket in ("OMN-1", "OMN-2"):
        assert f"carries no contracts/{ticket}.yaml" in result.stdout

    contracts = tmp_path / "head" / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "OMN-1.yaml").write_text("", encoding="utf-8")
    result = _run_presence(tmp_path, title)
    assert result.returncode == 1
    assert "carries no contracts/OMN-2.yaml" in result.stdout
    assert "carries no contracts/OMN-1.yaml" not in result.stdout


def test_presence_admits_a_title_that_cites_no_ticket(tmp_path: Path) -> None:
    result = _run_presence(tmp_path, "chore: tidy")
    assert result.returncode == 0
    assert (
        "::notice::Contract presence: the pull request title cites no OMN ticket."
        in result.stdout
    )


def test_presence_reads_the_title_like_the_reusable(tmp_path: Path) -> None:
    result = _run_presence(tmp_path, "fix(omn-123): x")
    assert result.returncode == 0
    assert "title cites no OMN ticket" in result.stdout

    contracts = tmp_path / "head" / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "omn-123.yaml").write_text("", encoding="utf-8")
    result = _run_presence(tmp_path, "fix(OMN-123): x")
    assert result.returncode == 1
    assert "carries no contracts/OMN-123.yaml" in result.stdout


def test_presence_red_reds_ci_summary() -> None:
    assert "contract-validation" in EXPECTED_EXTERNAL_CONTEXTS
