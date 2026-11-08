# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21916: the lane resolver names the variable the caller actually read.

Observed 2026-11-08: the C15, C16 and C28 proofs (stability overlay,
``vars.LAB_PROOF_LANES_JSON``) all ended in the resolve step with
``no lab lane is declared; set the repository variable LAB_LANES_JSON``. The
resolver hard-coded the dev overlay's variable name in every message, so a
stability caller whose own variable was empty was pointed at a variable that
is set (C11 and C12 read it and are green). These tests pin that each message
carries the name the caller passes, and that the three stability proofs pass it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = REPO_ROOT / ".github" / "actions" / "resolve-lab-lane"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

STABILITY_VAR = "LAB_PROOF_LANES_JSON"
DEV_VAR = "LAB_LANES_JSON"

STABILITY_PROOFS = (
    "chain-canary.yml",
    "chain-canary-c16-receipt-identity.yml",
    "chain-canary-c28-consumer-flow.yml",
)


def _load_resolver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "resolve_lab_lane_omn21916", ACTION_DIR / "resolve_lab_lane.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver = _load_resolver()


def _main_output(
    capsys: pytest.CaptureFixture[str], lanes_raw: str, var_name: str | None
) -> tuple[int, str]:
    environ = {
        "LAB_LANES_JSON": lanes_raw,
        "LANE_REQUIRE": "gateway_url",
        "LANE_MATCH": "lane_role=stability",
    }
    if var_name is not None:
        environ["LAB_LANES_VAR"] = var_name
    code = resolver.main(environ)
    return code, capsys.readouterr().out


@pytest.mark.parametrize("raw", ["", "   "])
def test_an_empty_overlay_names_the_variable_the_caller_passed(
    capsys: pytest.CaptureFixture[str], raw: str
) -> None:
    code, out = _main_output(capsys, raw, STABILITY_VAR)
    assert code == 1
    assert STABILITY_VAR in out
    assert DEV_VAR not in out


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("not json", "is not JSON"),
        ("{}", "must be a non-empty JSON list"),
        ("[1]", "is not an object"),
        (json.dumps([{"name": ""}]), "has no valid name"),
    ],
)
def test_a_malformed_overlay_names_the_variable_the_caller_passed(
    capsys: pytest.CaptureFixture[str], raw: str, fragment: str
) -> None:
    code, out = _main_output(capsys, raw, STABILITY_VAR)
    assert code == 1
    assert fragment in out
    assert STABILITY_VAR in out
    assert DEV_VAR not in out


def test_the_dev_overlay_message_is_unchanged_when_the_caller_names_no_variable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out = _main_output(capsys, "", None)
    assert code == 1
    assert f"set repository variable {DEV_VAR}" in out


def test_the_action_forwards_the_variable_name_to_the_resolver() -> None:
    action = yaml.safe_load((ACTION_DIR / "action.yml").read_text(encoding="utf-8"))
    assert action["inputs"]["lanes-var"]["default"] == DEV_VAR
    step_env = action["runs"]["steps"][0]["env"]
    assert step_env["LAB_LANES_VAR"] == "${{ inputs.lanes-var }}"


@pytest.mark.parametrize("workflow", STABILITY_PROOFS)
def test_each_stability_proof_passes_the_name_of_the_variable_it_reads(
    workflow: str,
) -> None:
    loaded = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    resolve_steps = [
        step
        for job in loaded["jobs"].values()
        for step in job["steps"]
        if step.get("uses") == "./.github/actions/resolve-lab-lane"
    ]
    assert len(resolve_steps) == 1
    inputs = resolve_steps[0]["with"]
    assert inputs["lanes-json"] == "${{ vars.LAB_PROOF_LANES_JSON }}"
    assert inputs["lanes-var"] == STABILITY_VAR
