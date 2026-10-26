# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ticket-contract criterion-label check is wired into one test split (OMN-21279).

omnibase_infra#5370 repeated criterion label AC7 that omnibase_infra#5365 had
already merged; Git merged the two contract files with no text conflict, the PR
runs stayed green, and the merge-queue full suite removed the PR three times.
The change-aware selector does not pick ``tests/scripts/`` for a
``contracts/`` change, so the check only ran in the full suite. The wiring
asserted below runs the check in one split of the ``test-parallel`` job of
``.github/workflows/ci.yml``, so a PR whose contract repeats a criterion
label goes red on its own run.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]
_CI_WORKFLOW = ".github/workflows/ci.yml"
_JOB_NAME = "test-parallel"
_CHECK_FILE = "tests/scripts/ci/test_contract_criterion_labels_distinct_omn17427.py"


@pytest.fixture
def workflow() -> dict[str, Any]:
    text = (_ROOT / _CI_WORKFLOW).read_text(encoding="utf-8")
    return cast("dict[str, Any]", yaml.safe_load(text))


def _steps_naming_the_check(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in workflow["jobs"][_JOB_NAME]["steps"]
        if _CHECK_FILE in str(step.get("run", ""))
    ]


def _assert_criterion_label_check_wiring(workflow: dict[str, Any]) -> None:
    named = _steps_naming_the_check(workflow)
    assert len(named) == 1, (
        f"{_CI_WORKFLOW} job {_JOB_NAME!r} names {_CHECK_FILE!r} in "
        f"{len(named)} steps; expected exactly one"
    )
    (step,) = named
    assert len(step["run"].strip().splitlines()) == 1
    command = shlex.split(step["run"])
    if command[:3] == ["uv", "run", "--frozen"]:
        command = command[3:]
    assert command[:2] == ["pytest", _CHECK_FILE]
    # Only reporting options: selection, collection, early exit and shell
    # commands must never turn this into a partial or non-blocking check.
    assert all(argument in {"-q", "--no-cov"} for argument in command[2:])

    condition = str(step.get("if", "")).strip()
    condition = condition.removeprefix("${{").removesuffix("}}")
    assert re.sub(r"[\s'\"]", "", condition) == "matrix.split==1"

    job = workflow["jobs"][_JOB_NAME]
    for scope in (job, step):
        value = scope.get("continue-on-error", False)
        assert value is False or value == "false"
    assert "shell" not in step
    for scope in (workflow, job):
        assert "shell" not in scope.get("defaults", {}).get("run", {})


def test_exactly_one_step_runs_the_criterion_label_check_blocking_in_split_1(
    workflow: dict[str, Any],
) -> None:
    _assert_criterion_label_check_wiring(workflow)


@pytest.mark.parametrize(
    ("target", "changes"),
    [
        pytest.param("step", {"run": f"echo {_CHECK_FILE}"}, id="echo-only"),
        *[
            pytest.param("step", {"run": f"pytest {_CHECK_FILE} {flag}"}, id=flag)
            for flag in (
                "--co",
                "--collect-only",
                "-k 'not contract'",
                f"--deselect={_CHECK_FILE}",
                f"--ignore={_CHECK_FILE}",
                "-m 'not unit'",
                "-x",
                "--maxfail=1",
                "--lf",
            )
        ],
        *[
            pytest.param("step", {"run": command}, id=name)
            for name, command in (
                ("or-true", f"pytest {_CHECK_FILE} || true"),
                ("semicolon", f"pytest {_CHECK_FILE}; true"),
                ("and-chain", f"true && pytest {_CHECK_FILE}"),
                ("set-plus-e", f"set +e\npytest {_CHECK_FILE}\ntrue"),
                ("second-command", f"pytest {_CHECK_FILE}\ntrue"),
                ("newline-between-arguments", f"pytest\n{_CHECK_FILE} -q"),
            )
        ],
        pytest.param("job", {"continue-on-error": True}, id="job-nonblocking"),
        pytest.param("job", {"continue-on-error": "true"}, id="job-string-true"),
        pytest.param("step", {"continue-on-error": True}, id="step-nonblocking"),
        pytest.param("step", {"continue-on-error": "true"}, id="step-string-true"),
        pytest.param("step", {"shell": "bash {0}"}, id="step-shell"),
        pytest.param(
            "job", {"defaults": {"run": {"shell": "bash {0}"}}}, id="job-shell"
        ),
        pytest.param(
            "workflow",
            {"defaults": {"run": {"shell": "bash {0}"}}},
            id="workflow-shell",
        ),
        pytest.param("step", {"if": "matrix.split == 2"}, id="wrong-split"),
        pytest.param("step", {"if": "always()"}, id="wrong-condition"),
    ],
)
def test_broken_wiring_is_rejected(
    workflow: dict[str, Any], target: str, changes: dict[str, Any]
) -> None:
    job = workflow["jobs"][_JOB_NAME]
    (step,) = _steps_naming_the_check(workflow)
    {"workflow": workflow, "job": job, "step": step}[target].update(changes)
    with pytest.raises(AssertionError):
        _assert_criterion_label_check_wiring(workflow)


@pytest.mark.parametrize(
    "condition",
    [
        "matrix.split == 1",
        "${{ matrix.split == 1 }}",
        "matrix.split == '1'",
        '${{ matrix.split == "1" }}',
        "  matrix.split   ==  1  ",
    ],
)
def test_equivalent_split_1_conditions_are_accepted(
    workflow: dict[str, Any], condition: str
) -> None:
    (step,) = _steps_naming_the_check(workflow)
    step["if"] = condition
    _assert_criterion_label_check_wiring(workflow)


@pytest.mark.parametrize("target", ["job", "step"])
@pytest.mark.parametrize("value", [False, "false"])
def test_explicitly_blocking_wiring_is_accepted(
    workflow: dict[str, Any], target: str, value: bool | str
) -> None:
    job = workflow["jobs"][_JOB_NAME]
    (step,) = _steps_naming_the_check(workflow)
    {"job": job, "step": step}[target]["continue-on-error"] = value
    _assert_criterion_label_check_wiring(workflow)


def test_the_named_check_file_exists() -> None:
    assert (_ROOT / _CHECK_FILE).is_file()
