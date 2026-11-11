# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The CI Summary poll must not destroy its own input, nor blame healthy gates.

On 2026-11-07 three omnibase_infra pull requests (#5702, #5763, #5781) each had
``CI Summary`` fail closed after the full 90 minute poll, naming 57 gates and 34
external contexts as "missing/pending" while every one of them had finished
green. The cause was two faults in one line of the poll loop:

* ``> jobs.json`` truncates the file BEFORE the pipeline runs, so a failed fetch
  leaves an empty listing, which the gate reads as "no gate has reported"; and
* the deadline message conflated "gating jobs still pending" with "the API was
  unreachable", so the log never said which had happened.

The outage failed 90 consecutive reads of the run's own job listing with zero
successful reads. These tests drive the step's real shell with a failing ``gh``
and a controlled clock, and pin both directions: a failed fetch leaves the last
good listing alone, and a deadline reached without one successful read names the
unreadable input instead of printing a gate list that would be its artefact.

Ticket: OMN-23027
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: What the step prints when the deadline is reached and no listing was ever read.
UNREADABLE_INPUT = "NOT a gating job red"
#: What it prints when gates really are pending; must not appear in the other case.
GATES_PENDING = "gating jobs or external contexts still pending"

SENTINEL = '[{"name": "a job read before the outage", "conclusion": "success"}]'

#: What the gate script is asked for when the deadline reports the gates.
REPORT_ONLY = "--report-only"


def _poll_step() -> dict[str, object]:
    """The CI Summary poll step, located by the text only it contains."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["ci-summary"]["steps"]
    found = [s for s in steps if "CI Summary poll deadline" in (s.get("run") or "")]
    assert len(found) == 1, f"expected exactly one poll step, found {len(found)}"
    return found[0]


def _run_poll(
    tmp_path: Path,
    *,
    clock_jumps_after: int,
    seed_jobs_json: str | None = None,
    gh_ok_calls: int = 0,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run the step's own shell with a failing ``gh`` and a controlled clock.

    ``clock_jumps_after`` is the number of ``date +%s`` calls answered with the
    real time before every later call jumps far past the deadline. The step calls
    it once to compute the deadline and once per loop iteration, so 2 lets
    exactly one fetch attempt happen before the deadline branch is taken, and
    each further read the poll makes needs one more real call.

    ``gh_ok_calls`` is how many ``gh`` calls succeed, returning one finished job,
    before every later call fails. A stub ``ci_summary_gate.py`` that reports
    PENDING records its arguments in ``gate_calls`` so a test can see whether the
    deadline asked it for a gate report.
    """
    step = _poll_step()
    script = tmp_path / "poll.sh"
    script.write_text(step["run"], encoding="utf-8")

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    # gh fails, on stderr, the way a 502 from the jobs listing does, once its
    # allowance of successful calls is spent.
    gh_counter = tmp_path / "gh_calls"
    gh_counter.write_text("0", encoding="utf-8")
    (stub_bin / "gh").write_text(
        "#!/bin/sh\n"
        f'n=$(cat "{gh_counter}")\n'
        f'echo $((n + 1)) > "{gh_counter}"\n'
        f'if [ "$n" -lt "{gh_ok_calls}" ]; then\n'
        '  echo \'{"name": "a finished job", "conclusion": "success"}\'\n'
        "  exit 0\n"
        "fi\n"
        'echo "gh: Something went wrong (HTTP 502)" >&2\n'
        "exit 1\n",
        encoding="utf-8",
    )
    gate_dir = tmp_path / "scripts" / "ci"
    gate_dir.mkdir(parents=True)
    (gate_dir / "ci_summary_gate.py").write_text(
        "import sys\n"
        f'open("{tmp_path / "gate_calls"}", "a").write(" ".join(sys.argv[1:]) + "\\n")\n'
        "sys.exit(2)\n",
        encoding="utf-8",
    )
    counter = tmp_path / "date_calls"
    counter.write_text("0", encoding="utf-8")
    (stub_bin / "date").write_text(
        "#!/bin/sh\n"
        f'n=$(cat "{counter}")\n'
        f'echo $((n + 1)) > "{counter}"\n'
        f'if [ "$n" -ge "{clock_jumps_after}" ]; then\n'
        f"  echo $(( $({shutil.which('date')} +%s) + 99999 ))\n"
        "else\n"
        f'  exec {shutil.which("date")} "$@"\n'
        "fi\n",
        encoding="utf-8",
    )
    for stub in ("gh", "date"):
        (stub_bin / stub).chmod(0o755)

    if seed_jobs_json is not None:
        (tmp_path / "jobs.json").write_text(seed_jobs_json, encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{stub_bin}:{env['PATH']}",
            "GH_TOKEN": "stub",
            "GH_REPO": "OmniNode-ai/omnibase_infra",
            "RUN_ID": "1",
            "RUN_ATTEMPT": "1",
            "EVENT_NAME": "push",
            "BASE_REF": "dev",
            "PR_AUTHOR": "",
            "PR_NUMBER": "",
            "HEAD_SHA": "0" * 40,
            "DEADLINE_MINUTES": "90",
            "POLL_INTERVAL_SECONDS": "0",
        }
    )
    return subprocess.run(
        ["bash", "-e", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.parametrize("seed", [SENTINEL, None])
def test_a_failed_fetch_never_empties_jobs_json(
    tmp_path: Path, seed: str | None
) -> None:
    """AC1, both directions: a prior listing survives, and none is invented."""
    result = _run_poll(tmp_path, clock_jumps_after=2, seed_jobs_json=seed)
    assert result.returncode == 1, result.stdout + result.stderr

    jobs_json = tmp_path / "jobs.json"
    if seed is None:
        assert not jobs_json.exists(), (
            "a failed fetch wrote a jobs listing that was never read: "
            f"{jobs_json.read_text(encoding='utf-8')!r}"
        )
    else:
        assert jobs_json.read_text(encoding="utf-8") == seed, (
            "the failed fetch truncated or replaced the last good job listing"
        )
    assert not (tmp_path / "jobs.json.tmp").exists(), (
        "the temporary listing was left behind after a failed fetch"
    )


def test_the_deadline_names_the_unreadable_input_not_the_gates(tmp_path: Path) -> None:
    """AC2. With no successful read there is nothing to report on, so report that."""
    result = _run_poll(tmp_path, clock_jumps_after=2)
    out = result.stdout + result.stderr

    assert result.returncode == 1, out
    assert UNREADABLE_INPUT in out, (
        "the deadline did not say the job listing was unreadable; it is about to "
        f"blame gates that may be green:\n{out}"
    )
    assert GATES_PENDING not in out, (
        "the deadline claimed gating jobs were pending although it never read "
        f"the job listing once:\n{out}"
    )
    assert "failed read(s)" in out, "the count of failed reads was not reported"
    assert not any(REPORT_ONLY in call for call in _gate_calls(tmp_path)), (
        "the deadline asked the gate script for a report of an unread listing; "
        "the stub gate records every call, so this is the omission AC2 requires"
    )


def _gate_calls(tmp_path: Path) -> list[str]:
    """The argument lines the stub gate script was called with, in order."""
    calls = tmp_path / "gate_calls"
    return calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []


def test_a_stale_listing_is_not_evidence_of_current_state(tmp_path: Path) -> None:
    """One good read, then failures to the deadline: the current state is unread.

    The first poll reads a listing and the gate reports PENDING; every later read
    fails until the deadline. The retained jobs.json is a snapshot, so the
    deadline must name the unreadable input and must not ask the gate script for
    a report of it, exactly as when no read ever succeeded.
    """
    result = _run_poll(tmp_path, clock_jumps_after=3, gh_ok_calls=1)
    out = result.stdout + result.stderr

    assert result.returncode == 1, out
    assert (tmp_path / "jobs.json").exists(), "the first good listing was not read"
    assert UNREADABLE_INPUT in out, (
        "an earlier successful read suppressed the outage attribution at the "
        f"deadline:\n{out}"
    )
    assert GATES_PENDING not in out, (
        "the deadline reported gates as pending off a listing the newest reads "
        f"failed to refresh:\n{out}"
    )
    assert "1 consecutive failed read(s)" in out, (
        f"the failure count was not the count since the last good read:\n{out}"
    )
    assert not any(REPORT_ONLY in call for call in _gate_calls(tmp_path)), (
        "the deadline printed a gate report from a stale listing"
    )


def test_a_deadline_with_a_current_listing_still_reports_the_gates(
    tmp_path: Path,
) -> None:
    """Positive control: when the newest read succeeded, the gates are reported.

    Without this the two tests above would pass for a deadline that always
    claimed an outage.
    """
    result = _run_poll(tmp_path, clock_jumps_after=2, gh_ok_calls=1)
    out = result.stdout + result.stderr

    assert result.returncode == 1, out
    assert GATES_PENDING in out, f"a current listing was reported as unreadable:\n{out}"
    assert UNREADABLE_INPUT not in out, out
    assert any(REPORT_ONLY in call for call in _gate_calls(tmp_path)), (
        "the deadline did not ask the gate script for its report"
    )


def test_the_failing_fetch_surfaces_the_http_status(tmp_path: Path) -> None:
    """AC3. Without the gh error text the cause is unreadable after the fact."""
    result = _run_poll(tmp_path, clock_jumps_after=2)
    # gh's own stderr reaches the step log by accident of ordering; what this
    # pins is the DELIBERATE capture, on stdout beside the failure line, so the
    # status is readable at the point of failure rather than wherever the
    # runner happens to interleave it.
    assert "  gh: " in result.stdout, (
        "the gh error behind the failed fetch was not captured beside the "
        f"failure line:\n{result.stdout}"
    )
    assert "HTTP 502" in result.stdout, (
        f"the captured gh error does not carry the HTTP status:\n{result.stdout}"
    )


def test_every_deadline_branch_still_fails_closed() -> None:
    """AC4. Nothing added here may let a deadline resolve the gate green."""
    run = str(_poll_step()["run"])
    lines = run.splitlines()
    deadline_lines = [
        i for i, line in enumerate(lines) if "CI Summary poll deadline" in line
    ]
    assert deadline_lines, "the poll step no longer announces a deadline"
    for i in deadline_lines:
        tail = "\n".join(lines[i : i + 4])
        assert "exit 1" in tail, (
            f"a deadline branch does not fail closed within 4 lines:\n{tail}"
        )
    assert "exit 0" not in run.split("case")[-1].split("esac")[0].replace(
        '0) echo "::notice::CI Summary = SUCCESS (all gating jobs and asserted '
        'external contexts provably passed)"; exit 0 ;;',
        "",
    ), "a second exit 0 path appeared in the verdict case"
