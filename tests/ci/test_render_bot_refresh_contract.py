# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20068: pin refresh PRs carry test-side evidence the gate can control.

The Repo Evidence Gate refuses a PR that cites a ticket without
``contracts/<ticket>.yaml`` at its head, refuses a contract carried unchanged
from the merge base, and requires a changed test-side path. With that test side
overlaid at the merge base, each bound check must fail. The omnimarket pin
producer writes the target fixture and renders a contract binding the pin
tests with a target-specific evidence item. These tests run the workflow's
rendering lines as written, then run the bound tests with pins at the target
and at the base, keeping the rendered contract and fixture overlaid.

OMN-19998: the producer edits the pin in the contract and nothing else. It used to
re-render the whole file from a template that claims the pin criterion only, while
the live ticket carries AC1 to AC4, so every bot pull request read AC2 to AC4 as
added and dropped the criteria dev holds. The tests refresh a contract that carries
several criteria and a retirement, require the rendered file to equal that contract
with the pin value replaced, and run the old template as a failing control.

OMN-20934: the lane census producer no longer opens its PR here. The census is
private config committed to the CI overlay repository, which runs no Repo
Evidence Gate, so that producer renders no contract;
tests/unit/scripts/test_lane_census_refresh_decision_omn18606.py pins that it
stages the census file alone.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
PIN_TEST = Path("tests/ci/test_omnimarket_contract_pin_advance_omn17292.py")
PIN_FIXTURE = Path("tests/ci/fixtures/omnimarket_contract_pin_advance_omn17292.yaml")
PIN_CHECK = (
    "env -u PYTHONPATH uv run --frozen pytest "
    "tests/ci/test_omnimarket_contract_pin_advance_omn17292.py -q"
)
# TEST_SIDE from omnibase_core receipt-gate.yml.
TEST_SIDE = r"^tests?/|/tests?/|(^|/)conftest\.py$|(^|/)test_[^/]*\.py$|_test\.py$"

SEED = REPO / "tests/ci/fixtures/omn17292_contract_multi_criteria_omn19998.yaml"
SEED_PIN = "f" * 40

# The rendering the workflow ran before OMN-19998, kept as the failing control: it
# writes a whole contract that claims AC1 only.
OLD_TEMPLATE = r"""set -euo pipefail
CHECK="env -u PYTHONPATH uv run --frozen pytest tests/ci/test_omnimarket_contract_pin_advance_omn17292.py -q"
cat > contracts/OMN-17292.yaml <<EOF_CONTRACT
schema_version: "1.0.0"
ticket_id: OMN-17292
title: "The omnimarket contract pin advances to ${TARGET_PIN}"
summary: The automated refresh moves .github/omnimarket-contract-pin.yaml to a value the merge base does not carry.
is_seam_ticket: false
interface_change: false
interfaces_touched: []
requirements:
  - id: req-omn17292
    statement: .github/omnimarket-contract-pin.yaml carries the value this refresh produced.
    acceptance:
      - id: AC1
        statement: "AC1: the omnimarket contract pin, the omnimarket sibling pin and every inline omnimarket checkout ref carry ${TARGET_PIN} -- falsifier: $CHECK"
evidence_requirements:
  - kind: tests
    description: The contract pin, sibling pin and every inline omnimarket checkout ref carry ${TARGET_PIN}.
    command: "$CHECK"
emergency_bypass:
  enabled: false
  justification: ''
  follow_up_ticket_id: ''
dod_evidence:
  - id: dod-omn-17292-ac1
    description: The contract pin, sibling pin and every inline omnimarket checkout ref carry ${TARGET_PIN}.
    source: manual
    checks:
      - check_type: test_passes
        check_value: "$CHECK"
        cwd: \${OMNI_HOME}/omnibase_infra
    binds_ac: [AC1]
EOF_CONTRACT
"""

PRODUCERS = [
    pytest.param(
        "omnimarket-contract-pin-refresh.yml",
        "OMN-17292",
        ".github/omnimarket-contract-pin.yaml",
        {"TARGET_PIN": "4a" * 20},
        "omnimarket_contract_ref: " + "f" * 40 + "\n",
        "omnimarket_contract_ref: " + "4a" * 20 + "\n",
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
            if "<<'PY_CONTRACT'" in run and f"contracts/{ticket}.yaml" in run:
                return str(run)
    raise AssertionError(f"{workflow} edits no contracts/{ticket}.yaml")


def _render_lines(script: str) -> str:
    start = script.index("PATTERN=")
    end = script.index("\nPY_CONTRACT\n", start) + len("\nPY_CONTRACT\n")
    return "set -euo pipefail\n" + script[start:end]


def _labels(contract: dict[str, object]) -> set[str]:
    requirements = contract["requirements"]
    assert isinstance(requirements, list)
    return {
        str(criterion["id"])
        for requirement in requirements
        for criterion in requirement.get("acceptance", [])
    }


def _only_the_pin_changed(seed: str, rendered: str, old: str, new: str) -> bool:
    """The reading that fails the whole-file template: the pin is the only edit."""
    return rendered != seed and rendered == seed.replace(old, new)


def _lay_out(tmp_path: Path, file: str, pin_file_text: str) -> None:
    target = tmp_path / file
    target.parent.mkdir(parents=True)
    (tmp_path / "contracts").mkdir()
    (tmp_path / PIN_FIXTURE).parent.mkdir(parents=True)
    target.write_text(pin_file_text, encoding="utf-8")
    shutil.copyfile(SEED, tmp_path / "contracts" / "OMN-17292.yaml")


def _env(env: dict[str, str]) -> dict[str, str]:
    return {**env, "PATH": os.environ["PATH"]}


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
    _lay_out(tmp_path, file, head)

    lines = _render_lines(_step_script(workflow, ticket))
    subprocess.run(["bash", "-c", lines], cwd=tmp_path, env=_env(env), check=True)
    contract = yaml.safe_load((tmp_path / "contracts" / f"{ticket}.yaml").read_text())
    assert contract["ticket_id"] == ticket
    (item,) = [i for i in contract["dod_evidence"] if i["id"] == "dod-omn-17292-ac1"]
    (check,) = item["checks"]
    assert check["check_type"] == "test_passes"
    assert check["cwd"] == "${OMNI_HOME}/omnibase_infra"
    assert check["check_value"] == PIN_CHECK
    assert item["binds_ac"] == ["AC1"]
    target_sha = env["TARGET_PIN"]
    assert target_sha in item["description"]
    evidence = contract["evidence_requirements"][0]
    assert evidence["kind"] == "tests"
    assert evidence["command"] == PIN_CHECK
    assert target_sha in evidence["description"]
    fixture = yaml.safe_load((tmp_path / PIN_FIXTURE).read_text(encoding="utf-8"))
    assert fixture["omnimarket_contract_ref"] == target_sha
    assert re.search(TEST_SIDE, PIN_FIXTURE.as_posix())

    shutil.copyfile(REPO / PIN_TEST, tmp_path / PIN_TEST)
    sibling_pins = tmp_path / ".github/sibling-pins.yaml"
    inline_workflow = tmp_path / ".github/workflows/x.yml"
    inline_workflow.parent.mkdir()

    def write_other_pins(sha: str) -> None:
        sibling_pins.write_text(f"pins:\n  omnimarket: {sha}\n", encoding="utf-8")
        inline_workflow.write_text(
            "        repository: OmniNode-ai/omnimarket\n"
            f"          ref: {sha} # .github/sibling-pins.yaml\n",
            encoding="utf-8",
        )

    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(tmp_path),
        "-c",
        "/dev/null",
        "-o",
        "markers=unit: Unit tests",
        str(tmp_path / PIN_TEST),
    ]
    write_other_pins(target_sha)
    result = subprocess.run(
        argv, cwd=tmp_path, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    target.write_text(base, encoding="utf-8")
    write_other_pins("f" * 40)
    result = subprocess.run(
        argv, cwd=tmp_path, check=False, capture_output=True, text=True
    )
    assert result.returncode != 0, result.stdout + result.stderr


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
    _lay_out(tmp_path, file, base)

    lines = _render_lines(_step_script(workflow, ticket))
    result = subprocess.run(
        ["bash", "-c", lines], cwd=tmp_path, env=_env(env), check=False
    )
    assert result.returncode != 0
    contract = tmp_path / "contracts" / f"{ticket}.yaml"
    assert contract.read_text(encoding="utf-8") == SEED.read_text(encoding="utf-8")
    assert not (tmp_path / PIN_FIXTURE).exists()


@pytest.mark.parametrize(
    ("workflow", "ticket", "file", "env", "base", "head"), PRODUCERS
)
def test_refresh_edits_only_the_pin(
    tmp_path: Path,
    workflow: str,
    ticket: str,
    file: str,
    env: dict[str, str],
    base: str,
    head: str,
) -> None:
    _lay_out(tmp_path, file, head)
    seed_text = SEED.read_text(encoding="utf-8")

    lines = _render_lines(_step_script(workflow, ticket))
    subprocess.run(["bash", "-c", lines], cwd=tmp_path, env=_env(env), check=True)
    rendered_text = (tmp_path / "contracts" / f"{ticket}.yaml").read_text(
        encoding="utf-8"
    )

    assert _only_the_pin_changed(seed_text, rendered_text, SEED_PIN, env["TARGET_PIN"])
    assert SEED_PIN not in rendered_text
    seed, rendered = yaml.safe_load(seed_text), yaml.safe_load(rendered_text)
    assert _labels(rendered) == _labels(seed) == {"AC1", "AC2", "AC3", "AC4"}
    assert (
        rendered["dod_evidence"][3]["supersedes_ac_binding"]
        == (seed["dod_evidence"][3]["supersedes_ac_binding"])
    )


@pytest.mark.parametrize(
    ("workflow", "ticket", "file", "env", "base", "head"), PRODUCERS
)
def test_control_the_whole_file_template_fails_the_same_reading(
    tmp_path: Path,
    workflow: str,
    ticket: str,
    file: str,
    env: dict[str, str],
    base: str,
    head: str,
) -> None:
    _lay_out(tmp_path, file, head)
    seed_text = SEED.read_text(encoding="utf-8")

    subprocess.run(
        ["bash", "-c", OLD_TEMPLATE],
        cwd=tmp_path,
        env=_env(env),
        check=True,
    )
    rendered_text = (tmp_path / "contracts" / f"{ticket}.yaml").read_text(
        encoding="utf-8"
    )

    assert not _only_the_pin_changed(
        seed_text, rendered_text, SEED_PIN, env["TARGET_PIN"]
    )
    dropped = _labels(yaml.safe_load(seed_text)) - _labels(
        yaml.safe_load(rendered_text)
    )
    assert dropped == {"AC2", "AC3", "AC4"}


@pytest.mark.parametrize(
    ("workflow", "ticket", "file", "env", "base", "head"), PRODUCERS
)
@pytest.mark.parametrize("fault", ["no_title", "pin_already_named"])
def test_render_refuses_a_contract_it_cannot_move_by_its_pin(
    tmp_path: Path,
    workflow: str,
    ticket: str,
    file: str,
    env: dict[str, str],
    base: str,
    head: str,
    fault: str,
) -> None:
    _lay_out(tmp_path, file, head)
    contract = tmp_path / "contracts" / f"{ticket}.yaml"
    text = contract.read_text(encoding="utf-8")
    if fault == "no_title":
        text = text.replace("The omnimarket contract pin advances to", "Pin at")
    else:
        text = text.replace(SEED_PIN, env["TARGET_PIN"])
    contract.write_text(text, encoding="utf-8")

    lines = _render_lines(_step_script(workflow, ticket))
    result = subprocess.run(
        ["bash", "-c", lines], cwd=tmp_path, env=_env(env), check=False
    )
    assert result.returncode != 0
    assert contract.read_text(encoding="utf-8") == text


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
    render = script.index("<<'PY_CONTRACT'")
    add = script.index("git add --", render)
    commit = script.index("git commit", add)
    add_list = script[add : script.index("\nif ", add)]
    assert f"contracts/{ticket}.yaml" in add_list
    assert PIN_FIXTURE.as_posix() in add_list
    assert render < add < commit


@pytest.mark.live_contact(
    "tests/ci/fixtures/bot_refresh_contract_omn20413_dod_verify_annotations.json"
)
def test_each_recorded_refusal_names_the_contract_its_producer_now_renders(
    recorded_response: dict[str, object],
) -> None:
    recording = recorded_response["response"]
    assert isinstance(recording, dict)
    producers = {
        "omnibase_infra#4816": "omnimarket-contract-pin-refresh.yml",
    }
    # The recording also holds the retired lane census producer's refusal
    # (OMN-20934 moved that producer's PR out of this repository).
    assert set(producers) < set(recording)
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
