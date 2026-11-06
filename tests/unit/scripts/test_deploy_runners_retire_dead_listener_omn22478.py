# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Control for ``--retire-surplus`` reading a running-but-dead container as idle.

The recorded state is ``omninode-runner-18`` read on the primary runner host on
2026-11-06 at 15:38Z: a container running for days with an unhealthy health
status, whose listener exits with "Runner version v2.336.0 is deprecated and
cannot receive messages", whose entrypoint keeps looping, whose process list
carries no ``Runner.Worker``, and which GitHub reports offline.

``runner_is_idle`` needs GitHub to report the runner ``online`` and so read that
container "busy or unknown" on every pass, which left ``--retire-surplus`` with
no way to remove it. ``surplus_runner_is_idle`` now runs it with
``IDLE_ACCEPTS_OFFLINE=1``, which also accepts a dead listener: offline on
GitHub with busy false AND no ``Runner.Worker`` in the process list. Every other
reading stays fail-closed, and ``--rolling`` (which does not set the switch)
still reads an offline running container as not idle.

The functions run for real out of ``scripts/deploy-runners.sh``; a fake
``docker`` on ``PATH`` (reached through a fake ``ssh`` that runs the remote
command locally) replays the recorded docker output, and ``github_runner_state``
is replaced with a stub that answers the recorded GitHub status.
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
        "container_is_running",
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
    # loud so a test cannot pass through an unrecorded path.
    _write_exe(
        bindir / "docker",
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        '  inspect) printf "%s\\n" "${FAKE_DOCKER_RUNNING}" ;;\n'
        '  top) cat "${FAKE_DOCKER_TOP_FILE}" ;;\n'
        '  ps) [[ "${FAKE_DOCKER_RUNNING}" == "true" ]] && printf "%s running\\n" "${FAKE_DOCKER_NAME}" ;;\n'
        '  *) echo "unrecorded docker call: $*" >&2; exit 97 ;;\n'
        "esac\n",
    )
    return bindir


def _surplus_idle(
    script_functions: str,
    fakebin: Path,
    tmp_path: Path,
    *,
    github_state: str,
    top: str,
    running: str = "true",
) -> subprocess.CompletedProcess[str]:
    top_file = tmp_path / "top.txt"
    top_file.write_text(top)
    env = dict(os.environ)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["RUNNER_HOST"] = "fake-host.invalid"
    env["FAKE_DOCKER_RUNNING"] = running
    env["FAKE_DOCKER_TOP_FILE"] = str(top_file)
    env["FAKE_DOCKER_NAME"] = RUNNER
    script = (
        "set -uo pipefail\n"
        f'github_runner_state() {{ echo "{github_state}"; }}\n'
        f"{script_functions}\n"
        f'surplus_runner_is_idle "{RUNNER}"\n'
    )
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )


def test_recorded_runner_18_state_reads_idle(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """Running, offline on GitHub, no Runner.Worker: the recorded state reads idle."""
    result = _surplus_idle(
        script_functions,
        fakebin,
        tmp_path,
        github_state="offline false",
        top=RECORDED_TOP_RUNNER_18,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_offline_runner_with_a_live_job_worker_reads_busy(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """Failing control: offline on GitHub but a Runner.Worker is alive is NOT idle."""
    result = _surplus_idle(
        script_functions,
        fakebin,
        tmp_path,
        github_state="offline false",
        top=TOP_WITH_WORKER,
    )
    assert result.returncode != 0, (result.stdout, result.stderr)


def test_unknown_github_state_stays_fail_closed(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """An API failure or a name GitHub does not list reads "unknown", never idle."""
    result = _surplus_idle(
        script_functions,
        fakebin,
        tmp_path,
        github_state="unknown unknown",
        top=RECORDED_TOP_RUNNER_18,
    )
    assert result.returncode != 0, (result.stdout, result.stderr)


def test_online_busy_runner_stays_busy(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """An online runner that GitHub reports busy is never read as a dead listener."""
    result = _surplus_idle(
        script_functions,
        fakebin,
        tmp_path,
        github_state="online true",
        top=TOP_WITH_WORKER,
    )
    assert result.returncode != 0, (result.stdout, result.stderr)


def test_stopped_container_still_reads_idle(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """The existing stopped-container read is unchanged."""
    result = _surplus_idle(
        script_functions,
        fakebin,
        tmp_path,
        github_state="unknown unknown",
        top=RECORDED_TOP_RUNNER_18,
        running="false",
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_rolling_idle_check_without_the_switch_still_refuses_offline(
    script_functions: str, fakebin: Path, tmp_path: Path
) -> None:
    """Control: the same recorded state through plain runner_is_idle (the
    --rolling caller) stays not idle, so the retire route is the only change."""
    top_file = tmp_path / "top.txt"
    top_file.write_text(RECORDED_TOP_RUNNER_18)
    env = dict(os.environ)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["RUNNER_HOST"] = "fake-host.invalid"
    env["FAKE_DOCKER_RUNNING"] = "true"
    env["FAKE_DOCKER_TOP_FILE"] = str(top_file)
    env["FAKE_DOCKER_NAME"] = RUNNER
    script = (
        "set -uo pipefail\n"
        'github_runner_state() { echo "offline false"; }\n'
        f"{script_functions}\n"
        f'runner_is_idle "{RUNNER}"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0, (result.stdout, result.stderr)
