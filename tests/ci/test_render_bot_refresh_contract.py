# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20413: OCC-writer refresh PRs carry a head-bound contract for their ticket.

The Repo Evidence Gate refuses a PR that cites a ticket without
``contracts/<ticket>.yaml`` at its head, refuses a contract carried unchanged
from the merge base, and runs each bound check at the merge base, where it must
fail. The lane census and omnimarket contract-pin producers render that
contract inline in the step that opens their PR. These tests run the rendering
lines of each workflow, as written, against a refreshed and a pre-refresh file.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

PRODUCERS = [
    pytest.param(
        "lane-census-refresh.yml",
        "OMN-18606",
        "deploy/lane-census/census-snapshot.json",
        {"EMITTED": "2026-10-09T12:56:14.434338+00:00"},
        '{"emitted_at": "2026-10-07T18:52:47.150843+00:00"}\n',
        '{"emitted_at": "2026-10-09T12:56:14.434338+00:00"}\n',
        id="lane-census",
    ),
    pytest.param(
        "omnimarket-contract-pin-refresh.yml",
        "OMN-17292",
        ".github/omnimarket-contract-pin.yaml",
        {"TARGET_PIN": "4" * 40},
        "omnimarket_contract_ref: " + "f" * 40 + "\n",
        "omnimarket_contract_ref: " + "4" * 40 + "\n",
        id="omnimarket-pin",
    ),
]


def _step_script(workflow: str, ticket: str) -> str:
    data = yaml.safe_load(
        (REPO / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    )
    for job in data["jobs"].values():
        for step in job.get("steps", []):
            run = step.get("run", "")
            if f"contracts/{ticket}.yaml <<EOF_CONTRACT" in run:
                return str(run)
    raise AssertionError(f"{workflow} renders no contracts/{ticket}.yaml")


def _render_lines(script: str) -> str:
    start = script.index("PATTERN=")
    end = script.index("\nEOF_CONTRACT\n", start) + len("\nEOF_CONTRACT\n")
    return "set -euo pipefail\n" + script[start:end]


@pytest.mark.parametrize(
    ("workflow", "ticket", "file", "env", "base", "head"), PRODUCERS
)
def test_rendered_check_passes_at_head_and_fails_at_base(
    tmp_path: Path,
    workflow: str,
    ticket: str,
    file: str,
    env: dict[str, str],
    base: str,
    head: str,
) -> None:
    target = tmp_path / file
    target.parent.mkdir(parents=True)
    (tmp_path / "contracts").mkdir()
    target.write_text(head, encoding="utf-8")

    lines = _render_lines(_step_script(workflow, ticket))
    subprocess.run(["bash", "-c", lines], cwd=tmp_path, env=env, check=True)
    contract = yaml.safe_load((tmp_path / "contracts" / f"{ticket}.yaml").read_text())
    assert contract["ticket_id"] == ticket
    (item,) = contract["dod_evidence"]
    (check,) = item["checks"]
    assert check["check_type"] == "command"
    assert check["cwd"] == "${OMNI_HOME}/omnibase_infra"
    assert next(iter(env.values())) in check["check_value"]

    argv = shlex.split(check["check_value"])
    assert subprocess.run(argv, cwd=tmp_path, check=False).returncode == 0
    target.write_text(base, encoding="utf-8")
    assert subprocess.run(argv, cwd=tmp_path, check=False).returncode != 0


@pytest.mark.parametrize(
    ("workflow", "ticket", "file", "env", "base", "head"), PRODUCERS
)
def test_render_refuses_a_file_without_the_refreshed_value(
    tmp_path: Path,
    workflow: str,
    ticket: str,
    file: str,
    env: dict[str, str],
    base: str,
    head: str,
) -> None:
    target = tmp_path / file
    target.parent.mkdir(parents=True)
    (tmp_path / "contracts").mkdir()
    target.write_text(base, encoding="utf-8")

    lines = _render_lines(_step_script(workflow, ticket))
    result = subprocess.run(["bash", "-c", lines], cwd=tmp_path, env=env, check=False)
    assert result.returncode != 0
    assert not (tmp_path / "contracts" / f"{ticket}.yaml").exists()


@pytest.mark.parametrize(
    ("workflow", "ticket", "file", "env", "base", "head"), PRODUCERS
)
def test_producer_commits_the_contract(
    workflow: str,
    ticket: str,
    file: str,
    env: dict[str, str],
    base: str,
    head: str,
) -> None:
    script = _step_script(workflow, ticket)
    render = script.index(f"contracts/{ticket}.yaml <<EOF_CONTRACT")
    add = script.index(f"contracts/{ticket}.yaml", script.index("git add", render))
    assert render < add < script.index("git commit", render)


@pytest.mark.live_contact(
    "tests/ci/fixtures/bot_refresh_contract_omn20413_dod_verify_annotations.json"
)
def test_each_recorded_refusal_names_the_contract_its_producer_now_renders(
    recorded_response: dict[str, object],
) -> None:
    recording = recorded_response["response"]
    assert isinstance(recording, dict)
    producers = {
        "omnibase_infra#4772": "lane-census-refresh.yml",
        "omnibase_infra#4816": "omnimarket-contract-pin-refresh.yml",
    }
    assert set(recording) == set(producers)
    for pr, workflow in producers.items():
        messages = [a["message"] for a in recording[pr]["annotations"]]
        refusals = [
            re.fullmatch(
                r"pull request cites (OMN-[0-9]+) but carries no contracts/\1\.yaml", m
            )
            for m in messages
        ]
        (ticket,) = [r.group(1) for r in refusals if r]
        script = _step_script(workflow, ticket)
        assert f"contracts/{ticket}.yaml" in script
        assert f"{ticket})" in script or f"({ticket}" in script
