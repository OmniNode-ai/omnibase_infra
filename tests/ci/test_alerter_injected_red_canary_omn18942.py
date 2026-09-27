# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The injected-red canary for the failure-rate alerter (OMN-18942, AC-4).

AC-4 asks for a deliberately injected red SCHEDULED run arriving at the chat
destination. A scheduled run cannot be dispatched by hand (a dispatch is
`event == workflow_dispatch`, which the evaluator does not count), so the
injection has to be a scheduled workflow whose next tick fails on purpose.

The canary is green unless an injection window is open. The window is a
repository Actions variable holding an absolute UTC end time, so an injection
expires on its own: nobody has to remember to switch it off, and a forgotten
one cannot leave a permanent red.

Fixture pair (plan section 2d): each branch below is executed, the known-bad
input (an open window, a malformed value) and the valid one (unset, expired).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "alerter-injected-red-canary.yml"
VARIABLE = "ALERTER_CANARY_INJECT_RED_UNTIL"


def _workflow() -> dict[Any, Any]:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _canary_step() -> dict[str, Any]:
    jobs = _workflow()["jobs"]
    assert len(jobs) == 1
    (job,) = jobs.values()
    steps = [s for s in job["steps"] if "run" in s]
    assert len(steps) == 1, "the canary is one run step"
    step = steps[0]
    assert isinstance(step, dict)
    return step


def _canary_source() -> str:
    """The exact Python the workflow runs: the body of its `python3 - <<'PY'`."""
    run = str(_canary_step()["run"])
    lines = run.splitlines()
    assert lines[0].strip() == "python3 - <<'PY'", lines[0]
    end = lines.index("PY")
    return "\n".join(lines[1:end]) + "\n"


def _run(value: str | None) -> subprocess.CompletedProcess[str]:
    env: dict[str, str] = {"PATH": "/usr/bin:/bin"}
    if value is not None:
        env["INJECT_RED_UNTIL"] = value
    return subprocess.run(
        [sys.executable, "-c", _canary_source()],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_no_injection_is_green() -> None:
    result = _run(None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no injection window" in result.stdout


def test_an_empty_value_is_green() -> None:
    """An unset repository variable renders as an empty string in `vars.`."""
    result = _run("")
    assert result.returncode == 0, result.stdout + result.stderr


def test_an_open_window_is_red_and_says_so() -> None:
    until = _stamp(datetime.now(UTC) + timedelta(minutes=30))
    result = _run(until)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "::error" in result.stdout
    assert "INJECTED RED" in result.stdout
    assert until in result.stdout


def test_an_expired_window_is_green() -> None:
    result = _run(_stamp(datetime.now(UTC) - timedelta(minutes=1)))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "expired" in result.stdout


def test_a_malformed_value_is_red_not_ignored() -> None:
    """A typo in the injection must not read as a clean canary."""
    result = _run("tomorrow")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "not a UTC timestamp" in result.stdout


def test_the_canary_is_a_scheduled_workflow() -> None:
    """YAML 1.1 reads a bare `on` key as True."""
    triggers = _workflow().get(True) or _workflow().get("on")
    assert isinstance(triggers, dict)
    assert "schedule" in triggers, "the canary must produce event == schedule runs"
    assert "workflow_dispatch" not in triggers, (
        "a dispatch run is not counted by the evaluator, so it can only mislead "
        "someone into believing the injection was exercised"
    )


def test_the_canary_reads_the_window_from_the_repository_variable() -> None:
    step = _canary_step()
    assert step["env"]["INJECT_RED_UNTIL"] == f"${{{{ vars.{VARIABLE} }}}}"
    assert _workflow().get("permissions") == {}, (
        "the canary reads nothing and writes nothing; it takes no token scope"
    )
