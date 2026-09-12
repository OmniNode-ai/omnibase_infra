# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The lab-load probe reports honestly in its conclusion and its artifact (OMN-18253).

This is the plan's named regression fixture for phase 3 and the repair half of
it. The gate half is OMN-18249.

THREE STATES THAT USED TO LOOK THE SAME
    ``probe_lab_saturation_from_fleet`` never raises, so a probe that could not
    reach the runner API returned ``{"ok": false, ...}``, the heredoc printed it,
    and the step exited 0. A crashing interpreter left a ZERO-BYTE record behind
    the shell's redirect. And the job's ``continue-on-error: true`` kept either
    out of the run's conclusion. "The lab is quiet", "I could not measure" and
    "I died before writing" were one green.

WHAT MUST STAY TRUE, and is the reason this is not simply a suppression deletion
    A BUSY LAB IS A MEASUREMENT, NOT AN OUTAGE. The suppression's stated reason
    was that this job failing is a data point rather than an outage, and that
    reason is correct. It survives here as a property: saturation returns
    ``ok: true`` and exits 0. What fails is the probe being unable to answer,
    never the answer being bad news. ``test_a_saturated_lab_is_a_measurement``
    is that control, and a repair that deleted the suppression without it would
    turn every busy hour on the lab host into a red workflow.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "probe_lab_load.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "dev-lane-liveness.yml"
PRE_REPAIR = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn18253"
    / "dev-lane-liveness.pre-repair.yml.captured"
)

JOB = "lab-load-probe"


# --------------------------------------------------------------------------
# the module's own three states
# --------------------------------------------------------------------------
def _run(
    tmp_path: Path, probe_src: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Drive the real module with a stub probe injected on sys.path.

    The stub is supplied through the module's own ``--probe-module`` seam.
    Everything under test -- the record assembly, the serialisation, the exit
    status -- is the shipped code path, not a re-implementation.
    """
    (tmp_path / "stub_probe.py").write_text(probe_src, encoding="utf-8")
    out = tmp_path / "lab-load.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out",
            str(out),
            "--probe-module",
            "stub_probe",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path)},
    )
    return result, out


def _record(path: Path) -> dict[str, Any]:
    assert path.is_file(), "the record must exist in every state"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), "the record must never be empty or blank"
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    return parsed


MEASURED = """
def probe_lab_saturation_from_fleet(token, group, api):
    return {"ok": True, "age_seconds": 0,
            "hosts": [{"label": "org:omnibase-ci", "ratio": 0.06, "free_mem_mib": 58856}]}
"""

SATURATED = """
def probe_lab_saturation_from_fleet(token, group, api):
    return {"ok": True, "age_seconds": 0,
            "hosts": [{"label": "org:omnibase-ci", "ratio": 0.98, "free_mem_mib": 1024}]}
"""

CANNOT_MEASURE = """
def probe_lab_saturation_from_fleet(token, group, api):
    return {"ok": False, "error": "http_503"}
"""

RAISES = """
def probe_lab_saturation_from_fleet(token, group, api):
    raise RuntimeError("boom")
"""

UNIMPORTABLE = """
import yaml_that_does_not_exist  # noqa
"""


def test_a_measurement_exits_zero_and_says_so(tmp_path: Path) -> None:
    result, out = _run(tmp_path, MEASURED)
    assert result.returncode == 0, result.stdout + result.stderr
    record = _record(out)
    assert record["ok"] is True
    assert record["hosts"][0]["ratio"] == 0.06
    assert "sampled_at" in record


def test_a_saturated_lab_is_a_measurement(tmp_path: Path) -> None:
    """THE CONTROL THAT MAKES THE SUPPRESSION REMOVAL SAFE.

    A lab with 98% of its runners busy is the condition the saturation monitor
    exists to report. It must exit 0. A repair that let a busy lab fail this job
    would make every busy hour a red workflow, which is the reading the plan
    explicitly rejects, and the suppression would be back within a week.
    """
    result, out = _run(tmp_path, SATURATED)
    assert result.returncode == 0, result.stdout + result.stderr
    record = _record(out)
    assert record["ok"] is True
    assert record["hosts"][0]["ratio"] == 0.98


def test_a_probe_that_cannot_measure_exits_non_zero_and_names_why(
    tmp_path: Path,
) -> None:
    result, out = _run(tmp_path, CANNOT_MEASURE)
    assert result.returncode == 1, result.stdout + result.stderr
    record = _record(out)
    assert record["ok"] is False
    assert record["error"] == "http_503"


def test_an_exception_still_leaves_an_honest_record(tmp_path: Path) -> None:
    """The zero-byte case, closed at its source rather than only at the uploader."""
    result, out = _run(tmp_path, RAISES)
    assert result.returncode == 1, result.stdout + result.stderr
    record = _record(out)
    assert record["ok"] is False
    assert record["error"].startswith("unhandled:RuntimeError")


def test_an_unimportable_probe_is_recorded_not_crashed(tmp_path: Path) -> None:
    """The exact shape of the incident: the import died, so nothing was written.

    ``ModuleNotFoundError: No module named 'yaml'`` killed 10 of this job's
    first 12 runs and left a zero-byte artifact behind the shell redirect.
    """
    result, out = _run(tmp_path, UNIMPORTABLE)
    assert result.returncode == 1, result.stdout + result.stderr
    record = _record(out)
    assert record["ok"] is False
    assert record["error"].startswith("probe_unimportable:ModuleNotFoundError")


def test_every_state_writes_a_non_empty_record(tmp_path: Path) -> None:
    """Stated as its own assertion because it is the artifact half of AC3."""
    for index, src in enumerate(
        (MEASURED, SATURATED, CANNOT_MEASURE, RAISES, UNIMPORTABLE)
    ):
        workdir = tmp_path / f"state{index}"
        workdir.mkdir()
        _, out = _run(workdir, src)
        assert out.stat().st_size > 0


# --------------------------------------------------------------------------
# the job's own conclusion
# --------------------------------------------------------------------------
def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_the_probe_job_no_longer_suppresses_its_own_failure() -> None:
    """AC3, conclusion half: the failure reaches the job's own conclusion."""
    job = _workflow()["jobs"][JOB]
    assert "continue-on-error" not in job, (
        "lab-load-probe carries job-level continue-on-error again. Every step "
        "failure inside it becomes irrelevant to the run's conclusion, which is "
        "the form the plan's motivating example used."
    )
    for step in job["steps"]:
        assert not step.get("continue-on-error"), (
            f"step {step.get('name') or step.get('uses')!r} suppresses its own failure"
        )


def test_the_pre_repair_capture_shows_the_suppression_this_removed() -> None:
    """A gate only ever seen green is indistinguishable from one that checks nothing.

    The capture is the byte copy of this workflow at ``b080d7eb``, the head this
    repair was written against -- after OMN-18247 landed the artifact assertion,
    so job-level suppression is the ONLY remaining violation in it. That makes it
    the sharp RED fixture for OMN-18249's suppression gate.
    """
    assert PRE_REPAIR.is_file(), f"{PRE_REPAIR} is missing"
    captured = yaml.safe_load(PRE_REPAIR.read_text(encoding="utf-8"))
    assert captured["jobs"][JOB].get("continue-on-error") is True


def test_the_lane_probe_is_not_blocked_by_this_job() -> None:
    """The repair must not turn a lab data point into a lane outage.

    Two structural facts carry that: the lane liveness job does not depend on
    this one at all, and the saturation record still runs under ``always()``, so
    a failing probe is recorded rather than dropped.
    """
    jobs = _workflow()["jobs"]
    assert "needs" not in jobs["dev-lane-liveness"], (
        "the lane liveness job must not depend on the lab-load probe"
    )
    saturation = jobs["saturation-record"]
    assert JOB in saturation["needs"]
    assert "always()" in str(saturation["if"]), (
        "saturation-record must still run when the probe fails; an absent lab "
        "record is alert condition 4, not a reason to record nothing"
    )


def test_the_job_runs_the_module_rather_than_an_inline_heredoc() -> None:
    """The record-writing logic must be testable, which an inline heredoc is not."""
    runs = "\n".join(
        str(step.get("run") or "") for step in _workflow()["jobs"][JOB]["steps"]
    )
    assert "scripts/ci/probe_lab_load.py" in runs
    assert "PYEOF" not in runs, (
        "the probe is back inline; nothing but a live workflow run can exercise it there"
    )
