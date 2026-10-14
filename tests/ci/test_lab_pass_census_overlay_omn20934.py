# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20934 -- the dev lab-pass verify job supplies the private lane census.

The receipt's ``lane_sync`` check runs ``scripts/lane-census-check.sh``, which
reads the lane manifest from ``$ONEX_WORKSPACE_CONFIG_ROOT``. The manifest left
this repository at 222eebe0d, and the verify runners carry no OMNI_HOME and no
private clone, so from that merge on every dev lab-pass receipt read::

    lane_sync indeterminate  census unreadable: ValueError: census exit=3 ...
    neither ONEX_WORKSPACE_CONFIG_ROOT nor OMNI_HOME is set

and no dev head could reach PASS. The fix is the step every other census
reader already takes, ``.github/actions/fetch-census-overlay``, placed in the
verify job ahead of the probe. These tests pin that placement, refuse its
absence (negative control on a job without the step), and check that the
probe's census inherits the variable the action exports instead of
overwriting it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.lab_pass_receipt import _run_read_only, check_lane_sync

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runtime-rebuild-trigger.yml"
ACTION = REPO_ROOT / ".github" / "actions" / "fetch-census-overlay" / "action.yml"
VERIFY_JOB = "verify-lane-converged"
PROBE_STEP = "Probe the dev lane"
FETCH_USES = "./.github/actions/fetch-census-overlay"
CONFIG_ROOT_VAR = "ONEX_WORKSPACE_CONFIG_ROOT"


def _steps(workflow: Mapping[str, Any], job: str) -> list[dict[str, Any]]:
    steps = workflow["jobs"][job]["steps"]
    assert isinstance(steps, list)
    return steps


def census_fetch_problem(workflow: Mapping[str, Any], job: str) -> str | None:
    """Why ``job`` would probe the lane without the private census, or None."""
    steps = _steps(workflow, job)
    probe = [i for i, s in enumerate(steps) if s.get("name") == PROBE_STEP]
    if len(probe) != 1:
        return f"{job} has {len(probe)} '{PROBE_STEP}' steps"
    fetch = [i for i, s in enumerate(steps) if s.get("uses") == FETCH_USES]
    if not fetch:
        return f"{job} never fetches the private lane census"
    if fetch[0] > probe[0]:
        return f"{job} fetches the census after '{PROBE_STEP}'"
    step = steps[fetch[0]]
    if step.get("if") != steps[probe[0]].get("if"):
        return (
            f"the census fetch runs under {step.get('if')!r}, the probe under "
            f"{steps[probe[0]].get('if')!r}"
        )
    inputs = step.get("with") or {}
    for key, ref in (
        ("overlay-repo", "vars.OMNI_CI_OVERLAY_REPO"),
        ("app-id", "secrets.ONEXBOT_APP_ID"),
        ("private-key", "secrets.ONEXBOT_APP_PRIVATE_KEY"),
    ):
        if ref not in str(inputs.get(key, "")):
            return f"the census fetch input {key} does not read {ref}"
    if CONFIG_ROOT_VAR in (steps[probe[0]].get("env") or {}):
        return f"'{PROBE_STEP}' overrides {CONFIG_ROOT_VAR} the fetch exported"
    return None


def _workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_the_verify_job_fetches_the_census_before_the_probe() -> None:
    assert census_fetch_problem(_workflow(), VERIFY_JOB) is None


def test_a_verify_job_without_the_fetch_is_refused() -> None:
    """Negative control: the job as it stood at 0ba1a61d3."""
    workflow = _workflow()
    steps = _steps(workflow, VERIFY_JOB)
    workflow["jobs"][VERIFY_JOB]["steps"] = [
        s for s in steps if s.get("uses") != FETCH_USES
    ]
    problem = census_fetch_problem(workflow, VERIFY_JOB)
    assert problem is not None
    assert "never fetches" in problem


def test_a_fetch_after_the_probe_is_refused() -> None:
    workflow = _workflow()
    steps = _steps(workflow, VERIFY_JOB)
    fetch = next(s for s in steps if s.get("uses") == FETCH_USES)
    reordered = [s for s in steps if s is not fetch] + [fetch]
    workflow["jobs"][VERIFY_JOB]["steps"] = reordered
    problem = census_fetch_problem(workflow, VERIFY_JOB)
    assert problem is not None
    assert "after" in problem


def test_the_action_exports_the_variable_the_census_reads() -> None:
    text = ACTION.read_text(encoding="utf-8")
    assert f'echo "{CONFIG_ROOT_VAR}=$dest" >> "$GITHUB_ENV"' in text


class _CapturingRunner:
    """Records the env each call adds; answers docker info, then a clean plan."""

    def __init__(self) -> None:
        self.envs: list[Mapping[str, str] | None] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.envs.append(env)
        if argv[0] == "docker":
            return subprocess.CompletedProcess(list(argv), 0, "lab-host-a\n", "")
        plan = (
            '{"host": "lab-host-a", "lanes_checked": ["dev"], "findings": [], '
            '"has_drift": false}'
        )
        return subprocess.CompletedProcess(list(argv), 0, plan, "")


def test_the_census_call_does_not_override_the_config_root() -> None:
    runner = _CapturingRunner()
    check = check_lane_sync(runner=runner)
    assert check.ok, check.evidence
    census_env = runner.envs[-1]
    assert census_env is not None
    assert "LANE_CENSUS_HOST" in census_env
    assert CONFIG_ROOT_VAR not in census_env


def test_the_read_only_runner_passes_the_exported_root_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CONFIG_ROOT_VAR, str(tmp_path))
    result = _run_read_only(
        [sys.executable, "-c", f"import os; print(os.environ['{CONFIG_ROOT_VAR}'])"],
        timeout=30,
        env={"LANE_CENSUS_HOST": "lab-host-a"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path)
    # Positive control: without the export the child sees nothing.
    monkeypatch.delenv(CONFIG_ROOT_VAR)
    result = _run_read_only(
        [
            sys.executable,
            "-c",
            f"import os; print(os.environ.get('{CONFIG_ROOT_VAR}', 'unset'))",
        ],
        timeout=30,
        env={"LANE_CENSUS_HOST": "lab-host-a"},
    )
    assert result.stdout.strip() == "unset"
    assert os.environ.get(CONFIG_ROOT_VAR) is None


class _ReplayRunner:
    """Answers each call with the next recorded response, in order."""

    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self._responses = list(responses)
        self.envs: list[Mapping[str, str] | None] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.envs.append(env)
        answer = self._responses.pop(0)
        return subprocess.CompletedProcess(
            list(argv),
            int(answer["returncode"]),
            str(answer["stdout"]),
            str(answer["stderr"]),
        )


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn20934_lane_sync_census_unconfigured.json"
)
def test_the_recorded_receipt_failure_is_the_unconfigured_census(
    recorded_response: dict[str, Any],
) -> None:
    """Replay the lane_sync responses recorded in the 0ba1a61d3 receipt.

    The probe must reach the recorded verdict byte for byte, and the recorded
    cause must be the absent config root that the fetch step above supplies.
    """
    recorded = recorded_response["check"]
    runner = _ReplayRunner(recorded_response["responses"])
    check = check_lane_sync(runner=runner)
    assert check.name == recorded["name"]
    assert check.ok is recorded["ok"]
    assert check.outcome.value == recorded["outcome"]
    assert check.evidence == recorded["evidence"]
    assert f"neither {CONFIG_ROOT_VAR} nor OMNI_HOME is set" in check.evidence
    census_env = runner.envs[-1]
    assert census_env is not None
    assert CONFIG_ROOT_VAR not in census_env
