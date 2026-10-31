# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-21807: ``deploy-runners.sh --rolling`` waits for the host to list no Dead
container before it recreates the next runner.

Incident (lane dev-restore-1515, TERMINAL 2026-10-31T16:03:34Z): each runner
recreate on the runner host removed an old container whose writable layer held the
uv cache and the shared ci-envs. containerd's garbage collector held its metadata
write lock for the whole delete, about 28 minutes, and every ``docker rm`` on the
host blocked behind it, so the dev deploy agent's 14:57Z and 15:26Z redeploys
failed. While that delete is pending Docker lists the container as Dead (ledger
row 2026-10-31T15:07:26Z, lane ledger-projection-1405: "every dev runtime
container on the runner host Created or Dead").

The listings below are the shape ``docker ps -a --filter status=dead --format
'{{.Names}}'`` prints, one container name per line, built from the container
names the ledger rows of that incident record. No raw ``docker ps`` transcript of
the 14:47Z to 15:15Z block was preserved, so these are not captured output.

The wait function and the rolling loop are extracted from the real script and run
for real in bash, with a fake ``ssh`` on PATH that serves scripted listings and
records every remote command it is asked to run.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"

# Names the incident's ledger rows record as pending removal on the runner host.
INCIDENT_DEAD_LISTING = (
    "omninode-runner-9\nomninode-runtime\nomninode-runtime-effects\n"
)

pytestmark = pytest.mark.unit


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
def script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _fake_ssh(bindir: Path, dead_listings: list[str], *, ssh_fails: bool) -> Path:
    """A fake ssh. The Nth ``docker ps`` call prints the Nth scripted listing (the
    last one repeats). Every remote command is appended to ``calls.log``."""
    bindir.mkdir(parents=True, exist_ok=True)
    for index, listing in enumerate(dead_listings):
        (bindir / f"listing.{index}").write_text(listing)
    ssh = bindir / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        f'dir="{bindir}"\n'
        'printf "%s\\n" "$2" >> "$dir/calls.log"\n'
        'case "$2" in\n'
        "  *'docker ps -a --filter status=dead'*)\n"
        f"    {'exit 255' if ssh_fails else ':'}\n"
        '    n=$(cat "$dir/ps.count" 2>/dev/null || echo 0)\n'
        '    echo $((n + 1)) > "$dir/ps.count"\n'
        f"    last={len(dead_listings) - 1}\n"
        "    idx=$(( n > last ? last : n ))\n"
        '    cat "$dir/listing.$idx"\n'
        "    ;;\n"
        "esac\n"
        "exit 0\n"
    )
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)
    return bindir


def _run(
    tmp_path: Path, body: str, dead_listings: list[str], *, ssh_fails: bool = False
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bindir = _fake_ssh(tmp_path / "bin", dead_listings, ssh_fails=ssh_fails)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    result = subprocess.run(
        ["bash", "-c", body],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls_file = bindir / "calls.log"
    calls = calls_file.read_text().splitlines() if calls_file.exists() else []
    return result, calls


def _wait_prelude(script_text: str) -> str:
    return (
        "set -euo pipefail\n"
        'log() { echo "[log] $*"; }\n'
        'warn() { echo "[warn] $*" >&2; }\n'
        "sleep() { :; }\n"
        'RUNNER_HOST="fake-host.invalid"\n'
        'ROLL_DEAD_MAX_SECONDS="${ROLL_DEAD_MAX_SECONDS:-30}"\n'
        'ROLL_ONLINE_INTERVAL_SECONDS="${ROLL_ONLINE_INTERVAL_SECONDS:-10}"\n'
        + _extract_function(script_text, "wait_for_no_dead_containers")
        + "\n"
    )


def test_quiet_host_returns_at_once_after_one_listing(
    script_text: str, tmp_path: Path
) -> None:
    result, calls = _run(
        tmp_path,
        _wait_prelude(script_text) + "wait_for_no_dead_containers",
        [""],
    )
    assert result.returncode == 0, result.stderr
    ps_calls = [c for c in calls if "status=dead" in c]
    assert len(ps_calls) == 1
    assert "docker ps -a" in ps_calls[0]


def test_incident_listing_makes_the_wait_poll_until_it_clears(
    script_text: str, tmp_path: Path
) -> None:
    """Positive control: three polls see the pending removals, the fourth is empty."""
    result, calls = _run(
        tmp_path,
        _wait_prelude(script_text) + "wait_for_no_dead_containers",
        [INCIDENT_DEAD_LISTING] * 3 + [""],
    )
    assert result.returncode == 0, result.stderr
    assert len([c for c in calls if "status=dead" in c]) == 4


def test_a_listing_that_never_clears_times_out_and_fails(
    script_text: str, tmp_path: Path
) -> None:
    result, calls = _run(
        tmp_path,
        _wait_prelude(script_text) + "wait_for_no_dead_containers",
        [INCIDENT_DEAD_LISTING],
    )
    assert result.returncode == 1
    # 30 s cap at a 10 s interval: polls at 0, 10, 20 and 30 s.
    assert len([c for c in calls if "status=dead" in c]) == 4


def test_an_unreadable_host_is_not_a_quiet_host(
    script_text: str, tmp_path: Path
) -> None:
    result, _ = _run(
        tmp_path,
        _wait_prelude(script_text) + "wait_for_no_dead_containers",
        [""],
        ssh_fails=True,
    )
    assert result.returncode == 1
    assert "listing is unreadable" in result.stdout


def _rolling_body(script_text: str) -> str:
    funcs = "\n".join(
        _extract_function(script_text, name)
        for name in (
            "wait_for_no_dead_containers",
            "roll_one_runner",
            "rolling_deploy",
        )
    )
    return (
        "set -euo pipefail\n"
        'log() { echo "[log] $*"; }\n'
        'warn() { echo "[warn] $*" >&2; }\n'
        'err() { echo "[err] $*" >&2; exit 1; }\n'
        "sleep() { :; }\n"
        "rsync_artifacts() { :; }\n"
        "install_runner_slice() { :; }\n"
        "fleet_services() { printf 'omninode-runner-1\\nomninode-runner-2\\n'; }\n"
        "runner_is_idle() { return 0; }\n"
        "_runner_cache_key_for_service() { echo key; }\n"
        "_runner_cache_ready() { return 0; }\n"
        "runner_container_state() { echo running; }\n"
        "encode_token() { echo dG9rZW4=; }\n"
        "wait_for_runner_online() { return 0; }\n"
        'RUNNER_HOST="fake-host.invalid"\n'
        'RUNNER_HOST_DIR="/fake/dir"\n'
        'DRY_RUN=false\nTOKEN_FILE=""\nROLL_ONLY=""\nROLL_LIMIT=0\n'
        "ROLL_SKIP_RETRY_PASSES=0\n"
        'COMPOSE_FILE="docker/docker-compose.runners.yml"\n'
        "ROLL_DEAD_MAX_SECONDS=30\nROLL_ONLINE_INTERVAL_SECONDS=10\nROLL_ONLINE_MAX_SECONDS=240\n"
        'DEPLOY_RUNNER_TOKEN="placeholder-not-a-token"\n' + funcs + "\nrolling_deploy\n"
    )


def test_rolling_waits_for_the_previous_delete_before_each_recreate(
    script_text: str, tmp_path: Path
) -> None:
    """The first runner finds one Dead container and waits; both then roll, and
    every recreate comes after a listing that was empty."""
    result, calls = _run(
        tmp_path,
        _rolling_body(script_text),
        [INCIDENT_DEAD_LISTING, ""],
    )
    assert result.returncode == 0, result.stderr
    kinds = [
        "ps" if "status=dead" in c else "up"
        for c in calls
        if "status=dead" in c or "force-recreate" in c
    ]
    assert kinds == ["ps", "ps", "up", "ps", "up"]


def test_rolling_halts_and_recreates_nothing_while_a_delete_is_pending(
    script_text: str, tmp_path: Path
) -> None:
    result, calls = _run(
        tmp_path,
        _rolling_body(script_text),
        [INCIDENT_DEAD_LISTING],
    )
    assert result.returncode == 1
    assert "HALTED at omninode-runner-1" in result.stderr
    assert "still lists a Dead container" in result.stderr
    assert not [c for c in calls if "force-recreate" in c]


def test_the_wait_comes_before_the_idle_recheck_and_the_recreate(
    script_text: str,
) -> None:
    body = _extract_function(script_text, "roll_one_runner")
    wait = body.index("wait_for_no_dead_containers")
    idle = body.index('runner_is_idle "${name}"')
    recreate = body.index("--force-recreate --no-deps --no-build")
    assert wait < idle < recreate
