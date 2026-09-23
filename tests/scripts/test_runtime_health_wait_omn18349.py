# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The deploy health wait honours the runtime container's own declared budget.

Measured defect (OMN-18349, .201, 2026-09-23T21:33Z): a governed
``refresh_stability_lane.sh --ref origin/dev --execute`` recreated the
stability-test runtime on 0.38.57. ``deploy-runtime.sh`` then polled
``/health`` 15 times 4 s apart, about 60 s, while the runtime took about 73 s
to boot (``/health`` 503 at 21:34:13Z, docker ``healthy`` at 21:34:33Z). The
deploy declared the lane dead, re-tagged every image back to its pre-build id
and wrote no refresh receipt, leaving healthy containers running the new build
under tags that name the old one.

The container declares its own start budget (``StartPeriod`` 1800 s,
``Interval`` 30 s, ``Retries`` 5). The fix keeps polling past the fixed floor
only while docker reports that container still ``starting``, bounded by that
declared budget, and stops at once when docker reports it unhealthy or not
running, so a crash still fails fast.

The helpers run as real bash against a fake ``docker``.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "runtime_health_wait.sh"

_FAKE_DOCKER = """#!/usr/bin/env bash
set -euo pipefail
# Fake `docker inspect --format FMT NAME`. FAKE_HEALTHCHECK is
# "<start_ns> <interval_ns> <retries> <timeout_ns>" and FAKE_STATE is
# "<status> <health>"; an empty FAKE_STATE means no such container.
[[ "$1" == "inspect" ]] || { echo "unsupported: $1" >&2; exit 2; }
fmt="$3"
if [[ -z "${FAKE_STATE:-}" ]]; then exit 1; fi
case "${fmt}" in
    # The real template must print nanoseconds through printf "%d": a bare
    # duration field renders "30m0s". Refuse any other form, as docker would
    # answer it with a string the helper cannot add up.
    *'printf "%d %d %d %d" .Config.Healthcheck.StartPeriod'*) printf '%s\\n' "${FAKE_HEALTHCHECK}" ;;
    *StartPeriod*) printf '30m0s 30s 5 10s\\n' ;;
    *State.Status*) printf '%s %s\\n' "${FAKE_STATE}" "${FAKE_STARTED_AT:-t0}" ;;
    *State.StartedAt*) printf '%s\\n' "${FAKE_STARTED_AT:-t0}" ;;
    *) echo "unsupported format: ${fmt}" >&2; exit 2 ;;
esac
"""


@pytest.fixture
def fake_docker_path(tmp_path: Path) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    return f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


def _run(snippet: str, path: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{LIB_SCRIPT}"\n{snippet}'],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": path, **env},
    )


_STABILITY_RUNTIME = "1800000000000 30000000000 5 10000000000"


@pytest.mark.unit
def test_budget_is_the_containers_declared_start_period_plus_its_retries(
    fake_docker_path: str,
) -> None:
    result = _run(
        "runtime_health_budget_seconds omninode-stability-test-runtime",
        fake_docker_path,
        FAKE_HEALTHCHECK=_STABILITY_RUNTIME,
        FAKE_STATE="running starting",
    )

    assert result.returncode == 0, result.stderr
    # 1800 s start period + 5 retries x (30 s interval + 10 s timeout).
    assert result.stdout.strip() == "2000"


@pytest.mark.unit
def test_budget_is_zero_for_a_container_that_declares_no_healthcheck(
    fake_docker_path: str,
) -> None:
    result = _run(
        "runtime_health_budget_seconds some-container",
        fake_docker_path,
        FAKE_HEALTHCHECK="0 0 0 0",
        FAKE_STATE="running none",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("running starting", "starting"),
        ("running healthy", "healthy"),
        ("running unhealthy", "unhealthy"),
        ("running none", "no-healthcheck"),
        ("restarting starting", "not-running"),
        ("exited none", "not-running"),
    ],
)
def test_state_names_what_docker_reports(
    fake_docker_path: str, state: str, expected: str
) -> None:
    result = _run(
        "runtime_health_state omninode-stability-test-runtime",
        fake_docker_path,
        FAKE_HEALTHCHECK=_STABILITY_RUNTIME,
        FAKE_STATE=state,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.unit
def test_a_restart_since_the_baseline_reads_restarted_not_starting(
    fake_docker_path: str,
) -> None:
    """A crash loop re-enters the start period on every restart; it must not
    read as a slow boot for the whole declared budget."""
    same = _run(
        "runtime_health_state rt t0",
        fake_docker_path,
        FAKE_HEALTHCHECK=_STABILITY_RUNTIME,
        FAKE_STATE="running starting",
        FAKE_STARTED_AT="t0",
    )
    moved = _run(
        "runtime_health_state rt t0",
        fake_docker_path,
        FAKE_HEALTHCHECK=_STABILITY_RUNTIME,
        FAKE_STATE="running starting",
        FAKE_STARTED_AT="t1",
    )
    baseline = _run(
        "runtime_started_at rt",
        fake_docker_path,
        FAKE_HEALTHCHECK=_STABILITY_RUNTIME,
        FAKE_STATE="running starting",
        FAKE_STARTED_AT="t0",
    )

    assert same.stdout.strip() == "starting"
    assert moved.stdout.strip() == "restarted"
    assert baseline.stdout.strip() == "t0"


@pytest.mark.unit
def test_state_of_an_absent_container_is_absent(fake_docker_path: str) -> None:
    result = _run(
        "runtime_health_state gone",
        fake_docker_path,
        FAKE_HEALTHCHECK=_STABILITY_RUNTIME,
        FAKE_STATE="",
    )

    assert result.stdout.strip() == "absent"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("elapsed", "state", "keep_waiting"),
    [
        # The 2026-09-23 boot: 73 s in, docker still says starting.
        ("73", "starting", True),
        ("1999", "starting", True),
        # Past the container's own declared budget: stop.
        ("2000", "starting", False),
        # Docker's own check passed; the host probe gets the rest of the budget.
        ("80", "healthy", True),
        # A crash or a failed check fails fast, whatever the budget.
        ("73", "unhealthy", False),
        ("73", "not-running", False),
        ("73", "absent", False),
        ("73", "restarted", False),
        # No declared healthcheck: nothing to extend the fixed floor with.
        ("73", "no-healthcheck", False),
    ],
)
def test_keep_waiting_only_while_the_container_is_still_starting(
    fake_docker_path: str, elapsed: str, state: str, keep_waiting: bool
) -> None:
    result = _run(
        f"runtime_health_keep_waiting {elapsed} 2000 {state}",
        fake_docker_path,
    )

    assert (result.returncode == 0) is keep_waiting, result.stderr


@pytest.mark.unit
def test_deploy_runtime_consults_the_container_before_giving_up() -> None:
    """The verify loop sources the helper and uses it; the fixed 15 x 4 s is a
    floor, no longer the whole wait."""
    script = (REPO_ROOT / "scripts" / "deploy-runtime.sh").read_text(encoding="utf-8")

    assert "runtime_build/runtime_health_wait.sh" in script
    assert "runtime_health_keep_waiting" in script
    assert "runtime_health_budget_seconds" in script
    assert "runtime_started_at" in script
