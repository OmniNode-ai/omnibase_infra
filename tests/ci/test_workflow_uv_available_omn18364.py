# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18364: a hosted-runner job may not invoke ``uv`` before installing it.

WHY THIS MODULE EXISTS AND NOT ONLY A REVIEW HABIT. The failure it ratchets had
already happened once in the same job and been written down at length in that
job's own comments — ``lab_pass_receipt.py`` importing pydantic with nothing in
the job installing it, so the gate "has never emitted a passing receipt … it is
fail-always". The next step added to that job invoked ``uv`` ninety lines ahead
of the job's ``astral-sh/setup-uv``, exited 127, and stopped every candidate
reaching staging for seventeen hours. A lesson recorded in prose did not
survive one edit.

WHAT IS ASSERTED, and why each half is needed:

1. The repository is clean. This is the ratchet.
2. A POSITIVE CONTROL — a fixture job that invokes ``uv`` on a hosted runner
   with no installer — is reported. A clean scan that cannot fail is
   indistinguishable from a scan that inspected nothing, which is the same
   confusion the receipt model's evidence rule exists to catch.
3. The three shapes that legitimately make ``uv`` available are accepted, so
   the ratchet does not push authors into a fourth, redundant setup step.
4. ``command -v uv`` is NOT an invocation. ``shared-env-runner-parity.yml``'s
   hosted job probes for uv's ABSENCE as its own positive control, and a
   substring checker would flag it.
5. Self-hosted jobs are out of scope. The fleet provisions a toolchain; ~200
   steps rely on it, and flagging them would get this checker disabled.
6. The step this ticket was opened for is covered — asserted against the live
   workflow, by name, so a rename does not silently drop it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from scripts.ci.check_workflow_uv_available import (
    Violation,
    installs_uv,
    invokes_uv,
    main,
    runs_on_hosted,
    scan_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github/workflows"
CHECKER = REPO_ROOT / "scripts/ci/check_workflow_uv_available.py"

DELIVER = WORKFLOWS / "deliver-dev-candidate-to-staging.yml"
GATE_JOB = "candidate-boot-gate"
DEADLINE_STEP_ID = "rollout-deadline"


def _write(tmp_path: Path, name: str, body: str) -> Path:
    directory = tmp_path / "workflows"
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text(dedent(body).lstrip(), encoding="utf-8")
    return path


def test_the_repository_has_no_hosted_job_invoking_uv_before_installing_it() -> None:
    """The ratchet itself.

    Every violation here is a step that will exit 127 the first time it runs,
    and where that step's outcome feeds a receipt or a required check the
    failure is attributed to the thing being checked rather than to the
    missing interpreter.
    """
    violations = scan_paths(
        sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]), REPO_ROOT
    )
    assert violations == [], "\n".join(
        ["hosted-runner steps invoking `uv` with no installer ahead of them:"]
        + [f"  {violation.render()}" for violation in violations]
    )


def test_the_positive_control_is_reported(tmp_path: Path) -> None:
    """A clean scan is only evidence if the same scan can report a dirty one."""
    path = _write(
        tmp_path,
        "control.yml",
        """
        name: control
        on: push
        jobs:
          gate:
            runs-on: ubuntu-latest
            steps:
              - name: Assert something
                run: |
                  set -euo pipefail
                  uv run --no-project --with pytest python3 -m pytest -q
        """,
    )
    violations = scan_paths([path], REPO_ROOT)
    assert len(violations) == 1
    assert violations[0] == Violation(
        workflow="control.yml", job="gate", index=0, step="Assert something"
    )
    assert "exit 127" in violations[0].render()


@pytest.mark.parametrize(
    "installer",
    [
        pytest.param(
            "      - uses: astral-sh/setup-uv@37802adc94f370d6bfd71619e3f0bf239e1f3b78",
            id="published-action",
        ),
        pytest.param(
            "      - run: curl -LsSf https://astral.sh/uv/install.sh | sh",
            id="install-script",
        ),
        pytest.param(
            "      - run: pip install uv",
            id="pip",
        ),
    ],
)
def test_an_earlier_installer_clears_the_finding(
    tmp_path: Path, installer: str
) -> None:
    """The three shapes already in this repository are all accepted.

    A ratchet that recognises one shape pushes authors into adding a second,
    redundant setup step beside a working one.
    """
    path = _write(
        tmp_path,
        "installed.yml",
        f"""
        name: installed
        on: push
        jobs:
          gate:
            runs-on: ubuntu-latest
            steps:
{installer}
              - name: Assert something
                run: uv run --no-project python3 -c 'print(1)'
        """,
    )
    assert scan_paths([path], REPO_ROOT) == []


def test_an_installer_in_a_later_step_does_not_clear_an_earlier_invocation(
    tmp_path: Path,
) -> None:
    """Order is the whole defect.

    The job that produced this ticket DOES install uv — ninety lines after the
    step that needs it. A checker that only asked "does this job install uv?"
    would have reported it clean.
    """
    path = _write(
        tmp_path,
        "ordering.yml",
        """
        name: ordering
        on: push
        jobs:
          gate:
            runs-on: ubuntu-latest
            steps:
              - name: Assert something
                run: uv run --no-project python3 -c 'print(1)'
              - uses: astral-sh/setup-uv@37802adc94f370d6bfd71619e3f0bf239e1f3b78
              - name: Emit the receipt
                run: uv run --no-project --with pydantic python3 emit.py
        """,
    )
    violations = scan_paths([path], REPO_ROOT)
    assert [violation.step for violation in violations] == ["Assert something"]


def test_probing_for_uv_is_not_invoking_it(tmp_path: Path) -> None:
    """``command -v uv`` is a probe for its ABSENCE, and a real one is live.

    ``shared-env-runner-parity.yml``'s hosted job asserts that uv is not
    pre-provisioned, as the positive control for its own cold-bootstrap path.
    A substring checker flags that step and the only way to quiet it is to
    weaken the control.
    """
    path = _write(
        tmp_path,
        "probe.yml",
        """
        name: probe
        on: push
        jobs:
          gate:
            runs-on: ubuntu-latest
            steps:
              - name: Prove no toolchain is pre-provisioned
                run: |
                  if command -v uv >/dev/null 2>&1; then
                    echo "::error::uv is unexpectedly pre-installed"
                    exit 1
                  fi
        """,
    )
    assert scan_paths([path], REPO_ROOT) == []
    assert not invokes_uv({"run": "if command -v uv >/dev/null 2>&1; then\n"})
    assert not invokes_uv({"run": "uv_version=1\necho ${uv_version}\n"})
    assert invokes_uv({"run": "set -e\nuv run --frozen ruff format\n"})
    assert invokes_uv({"run": "cd x && uv sync\n"})
    assert invokes_uv({"run": "UV_CACHE_DIR=/tmp uvx ruff check\n"})


def test_a_commented_invocation_is_not_an_invocation(tmp_path: Path) -> None:
    """The comment blocks in these workflows quote commands verbatim."""
    assert not invokes_uv({"run": "# uv run --frozen pytest -q\necho hi\n"})


def test_self_hosted_jobs_are_out_of_scope() -> None:
    """The fleet provisions a toolchain; ~200 steps depend on that.

    Fail-open on anything not provably hosted, including the routing
    expressions this repository's jobs resolve their labels from.
    """
    assert runs_on_hosted({"runs-on": "ubuntu-latest"})
    assert runs_on_hosted({"runs-on": ["ubuntu-24.04"]})
    assert runs_on_hosted({"runs-on": {"labels": ["ubuntu-latest"]}})
    assert not runs_on_hosted({"runs-on": ["self-hosted", "omnibase-ci"]})
    assert not runs_on_hosted(
        {"runs-on": "${{ fromJSON(needs.route.outputs.labels) }}"}
    )
    assert not runs_on_hosted({"runs-on": ["ubuntu-latest", "self-hosted"]})
    assert not runs_on_hosted({})


def test_a_local_composite_action_that_installs_uv_is_recognised() -> None:
    """``./.github/actions/setup-python-uv`` is how most jobs here get uv."""
    action = REPO_ROOT / ".github/actions/setup-python-uv"
    if not action.is_dir():
        pytest.skip("the local setup-python-uv composite action is not in this tree")
    assert installs_uv({"uses": "./.github/actions/setup-python-uv"}, REPO_ROOT)


def test_the_rollout_deadline_step_can_run_its_own_interpreter() -> None:
    """The step this ticket was opened for, asserted against the live workflow.

    Named rather than left to the repo-wide sweep so a rename of the step or of
    the job fails here with the name, instead of quietly removing the case.
    """
    document = yaml.safe_load(DELIVER.read_text(encoding="utf-8"))
    job = document["jobs"][GATE_JOB]
    steps = job["steps"]
    index = next(
        i for i, step in enumerate(steps) if step.get("id") == DEADLINE_STEP_ID
    )
    assert invokes_uv(steps[index]), (
        f"the {DEADLINE_STEP_ID!r} step no longer invokes uv; if that is "
        "deliberate this test should be retired rather than relaxed"
    )
    assert any(installs_uv(step, REPO_ROOT) for step in steps[:index]), (
        f"the {DEADLINE_STEP_ID!r} step invokes uv with no installer ahead of "
        f"it in {GATE_JOB!r}. It exits 127, and its outcome is interpolated "
        "into the lab-pass receipt's rollout_deadline_bounded check — so a "
        "missing interpreter is recorded as a verdict on the rollout-wedge "
        "invariant and the delivery to staging is refused."
    )


def test_the_checker_refuses_to_report_clean_on_a_directory_it_cannot_read(
    tmp_path: Path,
) -> None:
    """An empty result is not evidence of absence."""
    assert main(["--workflows-dir", str(tmp_path / "nope")]) == 1


def test_the_checker_exits_non_zero_from_the_command_line(tmp_path: Path) -> None:
    """The pre-commit hook and the CI job both invoke it as a process."""
    _write(
        tmp_path,
        "control.yml",
        """
        name: control
        on: push
        jobs:
          gate:
            runs-on: ubuntu-latest
            steps:
              - run: uv sync
        """,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--workflows-dir",
            str(tmp_path / "workflows"),
            "--repo-root",
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "OMN-18364" in completed.stderr
