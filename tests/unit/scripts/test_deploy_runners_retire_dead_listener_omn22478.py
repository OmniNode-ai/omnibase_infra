# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Control for ``--retire-surplus`` judging a surplus runner idle with no GitHub read.

The recorded states are on the primary runner host. ``omninode-runner-18`` on
2026-11-06 at 15:38Z (OMN-22478): a container running for days with an unhealthy
health status, whose listener exits "Runner version v2.336.0 is deprecated and
cannot receive messages", whose entrypoint keeps looping, whose process list
carries no ``Runner.Worker``. And ``omninode-runner-1..10`` on 2026-11-16 at
14:46Z and 17:04Z (OMN-24864): every runner read "unknown" by the GitHub-backed
idle check because the lane ``gh`` shim refuses the org runner read, so the
retire dry run removed 0 of 10 and failed closed.

``surplus_runner_is_idle`` now judges on the container's own evidence
(``relabel_container_idle``): stopped is idle; running needs no ``Runner.Worker``
in ``docker top`` and no unfinished "Running job:" in the listener log; anything
unreadable is not idle. It makes no ``gh`` call, and ``--rolling`` (which keeps
the GitHub-backed ``runner_is_idle``) is unchanged.

The functions run for real out of ``scripts/deploy-runners.sh``; a fake
``docker`` on ``PATH`` (reached through a fake ``ssh`` that runs the remote
command locally) replays recorded docker output, and a fake ``gh`` records any
call so a test fails if the retire route reaches GitHub.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"

pytestmark = pytest.mark.unit

RUNNER = "omninode-runner-18"

# `docker top omninode-runner-18` as read 2026-11-06T15:38Z: the entrypoint loop
# and its sleep, no listener, no Runner.Worker.
RECORDED_TOP_RUNNER_18 = (
    "UID                 PID                 PPID                C                   STIME               TTY                 TIME                CMD\n"
    "root                747806              747766              0                   Oct29               ?                   00:00:28            bash /usr/local/bin/entrypoint.sh\n"
    "root                2017049             747806              0                   10:38               ?                   00:00:00            sleep 60\n"
)

# The same container with a live job worker in it (the busy control).
TOP_WITH_WORKER = RECORDED_TOP_RUNNER_18 + (
    "root                3000001             747806              9                   15:40               ?                   00:00:09            "
    "bin/Runner.Worker spawnclient 123 456\n"
)


def _extract_function(script_text: str, name: str) -> str:
    start_marker = f"{name}() {{"
    start = script_text.index(start_marker)
    depth = 0
    end = None
    for i in range(start + len(start_marker) - 1, len(script_text)):
        ch = script_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end is not None, f"unbalanced braces extracting {name}()"
    return script_text[start:end]


@pytest.fixture(scope="module")
def script_functions() -> str:
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    names = (
        "relabel_remote",
        "relabel_job_log_probe",
        "relabel_container_idle",
        "runner_container_state",
        "runner_is_idle",
        "surplus_runner_is_idle",
    )
    # Names the script lacks are skipped so a base script still loads and the
    # recorded-state test fails on its assertion.
    return "\n".join(_extract_function(text, n) for n in names if f"{n}() {{" in text)


def _write_exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def fakebin(tmp_path: Path) -> Path:
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    _write_exe(
        bindir / "ssh",
        '#!/usr/bin/env bash\nset -euo pipefail\nshift\nexec bash -c "$1"\n',
    )
    # Replays recorded docker answers. Any call outside the recorded set fails
    # loud so a test cannot pass through an unrecorded path. A missing top file
    # is a `docker top` failure.
    _write_exe(
        bindir / "docker",
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        '  inspect) printf "%s\\n" "${FAKE_DOCKER_RUNNING}" ;;\n'
        '  top) [[ -f "${FAKE_DOCKER_TOP_FILE}" ]] || exit 1; cat "${FAKE_DOCKER_TOP_FILE}" ;;\n'
        '  logs) printf "%s" "${FAKE_DOCKER_LOGS:-}" ;;\n'
        '  ps) [[ "${FAKE_DOCKER_RUNNING}" == "true" ]] && printf "%s running\\n" "${FAKE_DOCKER_NAME}" ;;\n'
        '  *) echo "unrecorded docker call: $*" >&2; exit 97 ;;\n'
        "esac\n",
    )
    # The retire route must make no GitHub read: record any call and fail.
    _write_exe(
        bindir / "gh",
        '#!/usr/bin/env bash\necho "$*" >> "${FAKE_GH_CALLS}"\nexit 1\n',
    )
    return bindir


def _run(
    script_functions: str,
    fakebin: Path,
    tmp_path: Path,
    *,
    call: str,
    top: str | None,
    running: str = "true",
    logs: str = "",
    github_state: str = "unknown unknown",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    top_file = tmp_path / "top.txt"
    if top is not None:
        top_file.write_text(top)
    gh_calls = tmp_path / "gh_calls.txt"
    env = dict(os.environ)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["RUNNER_HOST"] = "fake-host.invalid"
    env["FAKE_DOCKER_RUNNING"] = running
    env["FAKE_DOCKER_TOP_FILE"] = str(top_file)
    env["FAKE_DOCKER_LOGS"] = logs
    env["FAKE_DOCKER_NAME"] = RUNNER
    env["FAKE_GH_CALLS"] = str(gh_calls)
    script = (
        "set -uo pipefail\n"
        "SECONDARY_HOST=false\n"
        f'github_runner_state() {{ echo "{github_state}"; }}\n'
        f"{script_functions}\n"
        f'{call} "{RUNNER}"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )
    return result, gh_calls


def test_recorded_runner_18_state_reads_idle(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """Running, no Runner.Worker, no unfinished job in the log: reads idle."""
    result, gh_calls = _run(
        script_functions,
        fakebin,
        tmp_path,
        call="surplus_runner_is_idle",
        top=RECORDED_TOP_RUNNER_18,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert not gh_calls.exists(), "the retire idle check read GitHub"


def test_github_state_does_not_decide_the_retire_idle_check(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """The recorded 2026-11-16 state: GitHub unreadable ("unknown") no longer
    blocks a container whose own evidence reads idle."""
    result, _ = _run(
        script_functions,
        fakebin,
        tmp_path,
        call="surplus_runner_is_idle",
        top=RECORDED_TOP_RUNNER_18,
        github_state="unknown unknown",
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_live_job_worker_reads_busy(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """Failing control: a Runner.Worker in the process list is never idle, even
    when GitHub (stubbed) says the runner is offline and not busy."""
    result, _ = _run(
        script_functions,
        fakebin,
        tmp_path,
        call="surplus_runner_is_idle",
        top=TOP_WITH_WORKER,
        github_state="offline false",
    )
    assert result.returncode != 0, (result.stdout, result.stderr)


def test_unfinished_job_in_listener_log_reads_busy(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """Failing control: no worker yet, but the listener took a job and has not
    reported it complete (the moment before Runner.Worker appears)."""
    result, _ = _run(
        script_functions,
        fakebin,
        tmp_path,
        call="surplus_runner_is_idle",
        top=RECORDED_TOP_RUNNER_18,
        logs="2026-11-16 17:00:00Z: Running job: pytest shard 3\n",
    )
    assert result.returncode != 0, (result.stdout, result.stderr)


def test_finished_job_in_listener_log_reads_idle(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """Positive control for the log probe: the last job completed."""
    result, _ = _run(
        script_functions,
        fakebin,
        tmp_path,
        call="surplus_runner_is_idle",
        top=RECORDED_TOP_RUNNER_18,
        logs=(
            "2026-11-16 17:00:00Z: Running job: pytest shard 3\n"
            "2026-11-16 17:05:00Z: Job pytest shard 3 completed with result: Succeeded\n"
        ),
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_unreadable_process_list_stays_fail_closed(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """`docker top` failing is unknown, never idle."""
    result, _ = _run(
        script_functions,
        fakebin,
        tmp_path,
        call="surplus_runner_is_idle",
        top=None,
    )
    assert result.returncode != 0, (result.stdout, result.stderr)


def test_stopped_container_still_reads_idle(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """The existing stopped-container read is unchanged."""
    result, _ = _run(
        script_functions,
        fakebin,
        tmp_path,
        call="surplus_runner_is_idle",
        top=RECORDED_TOP_RUNNER_18,
        running="false",
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_rolling_idle_check_still_refuses_offline(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """Control: the same recorded state through plain runner_is_idle (the
    --rolling caller) stays not idle, so the retire route is the only change."""
    result, _ = _run(
        script_functions,
        fakebin,
        tmp_path,
        call="runner_is_idle",
        top=RECORDED_TOP_RUNNER_18,
        github_state="offline false",
    )
    assert result.returncode != 0, (result.stdout, result.stderr)
