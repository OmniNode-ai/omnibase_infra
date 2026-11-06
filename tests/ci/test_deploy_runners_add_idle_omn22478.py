# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""`deploy-runners.sh --add` leaves a running job alone and mints no token through gh.

Run against recording ssh, gh and rsync boundaries; no live runner or credential
is used. Two properties, each shown failing on the merge-base script:

* a target that is busy, or whose state cannot be read, is never handed to
  ``docker compose up`` (which recreates a running container whose rendered
  config changed, and a not-online one is forced to recreate);
* the registration token is the supplied one (DEPLOY_RUNNER_TOKEN or
  --token-file); the script makes no ``gh api --method POST`` of its own.

Ticket: OMN-22604
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/deploy-runners.sh"
SERVICE = "omninode-verify-runner-1"
SUPPLIED = "supplied-registration-value"


@pytest.fixture
def boundary(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    env = dict(os.environ)
    env.pop("DEPLOY_RUNNER_TOKEN", None)
    env.pop("RUNNER_TOKEN", None)
    env.update(
        PATH=f"{bindir}:{env['PATH']}",
        RUNNER_TEST_ROOT=str(tmp_path),
        RELABEL_DRAIN_TIMEOUT_SECONDS="2",
        RELABEL_DRAIN_INTERVAL_SECONDS="1",
        ROLL_ONLINE_MAX_SECONDS="3",
        ROLL_ONLINE_INTERVAL_SECONDS="1",
    )
    programs = {
        "ssh": rf"""
            import os, sys
            from pathlib import Path
            root = Path(os.environ["RUNNER_TEST_ROOT"])
            command = sys.argv[2]
            with (root / "calls").open("a") as log:
                log.write("ssh\n")
            if "up -d" in command:
                sys.stdin.read()
                (root / "upped").write_text(command)
            elif "docker inspect" in command:
                print(os.environ.get("RUNNER_TEST_RUNNING", "true"))
            elif "docker ps -a --filter 'name=" in command:
                if os.environ.get("RUNNER_TEST_CONTAINER", "running") != "absent":
                    print("{SERVICE} " + os.environ.get("RUNNER_TEST_CONTAINER", "running"))
            elif "docker top" in command:
                print("TOPFAIL" if os.environ.get("RUNNER_TEST_TOP_FAIL") else os.environ.get("RUNNER_TEST_WORKERS", "0"))
            elif "docker logs" in command:
                print(os.environ.get("RUNNER_TEST_JOBLOG", "free"))
        """,
        "gh": rf"""
            import json, os, sys
            from pathlib import Path
            root = Path(os.environ["RUNNER_TEST_ROOT"])
            with (root / "gh_calls").open("a") as log:
                log.write(" ".join(sys.argv[1:]) + "\n")
            if os.environ.get("RUNNER_TEST_GH_FAIL"):
                sys.exit(1)
            status = os.environ.get("RUNNER_TEST_STATUS", "online")
            if (root / "upped").exists():
                status = "online"
            print(json.dumps({{"runners": [{{"name": "{SERVICE}", "status": status,
                "busy": bool(os.environ.get("RUNNER_TEST_BUSY"))}}]}}))
        """,
        "rsync": r"""
            import os
            from pathlib import Path
            with (Path(os.environ["RUNNER_TEST_ROOT"]) / "calls").open("a") as log:
                log.write("rsync\n")
        """,
    }
    for name, body in programs.items():
        program = bindir / name
        program.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body).lstrip())
        program.chmod(0o755)
    return env, tmp_path


def run_add(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), f"--add={SERVICE}", "--skip-build", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    "evidence",
    [
        {"RUNNER_TEST_WORKERS": "1"},
        {"RUNNER_TEST_JOBLOG": "busy"},
        {"RUNNER_TEST_TOP_FAIL": "1"},
        {"RUNNER_TEST_WORKERS": "1", "RUNNER_TEST_STATUS": "offline"},
    ],
    ids=[
        "job-worker",
        "unfinished-job-in-listener-log",
        "docker-top-unreadable",
        "offline-with-worker",
    ],
)
def test_a_target_that_may_be_mid_job_is_left_running_and_untouched(
    boundary: tuple[dict[str, str], Path], evidence: dict[str, str]
) -> None:
    env, root = boundary
    env.update(DEPLOY_RUNNER_TOKEN=SUPPLIED, **evidence)
    result = run_add(env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (root / "upped").exists(), (
        "compose up reached a runner that may be mid-job"
    )
    assert "left running and untouched" in result.stderr


def test_an_idle_online_target_converges_without_a_recreate(
    boundary: tuple[dict[str, str], Path],
) -> None:
    env, root = boundary
    env.update(DEPLOY_RUNNER_TOKEN=SUPPLIED)
    result = run_add(env)
    assert result.returncode == 0, result.stdout + result.stderr
    command = (root / "upped").read_text()
    assert "up -d --no-deps" in command
    assert "--force-recreate" not in command


def test_an_idle_offline_target_is_recreated(
    boundary: tuple[dict[str, str], Path],
) -> None:
    env, root = boundary
    env.update(DEPLOY_RUNNER_TOKEN=SUPPLIED, RUNNER_TEST_STATUS="offline")
    result = run_add(env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--force-recreate" in (root / "upped").read_text()


def test_an_absent_container_is_created_without_waiting(
    boundary: tuple[dict[str, str], Path],
) -> None:
    env, root = boundary
    env.update(
        DEPLOY_RUNNER_TOKEN=SUPPLIED,
        RUNNER_TEST_CONTAINER="absent",
        RUNNER_TEST_STATUS="offline",
    )
    result = run_add(env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "upped").exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_no_supplied_token_refuses_before_any_remote_action_and_mints_nothing(
    boundary: tuple[dict[str, str], Path], dry_run: bool
) -> None:
    env, root = boundary
    result = run_add(env, *(["--dry-run"] if dry_run else []))
    if dry_run:
        # A dry run plans only; it never needs a value.
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert "DEPLOY_RUNNER_TOKEN" in result.stderr
        assert "registration-token" in result.stderr
        assert not (root / "calls").exists(), "preflight must precede sync and all ssh"
    assert "--method POST" not in (
        (root / "gh_calls").read_text() if (root / "gh_calls").exists() else ""
    )


def test_a_supplied_token_file_reaches_the_host_and_is_never_printed(
    boundary: tuple[dict[str, str], Path],
) -> None:
    env, root = boundary
    token_file = root / "operator-supplied-token"
    token_file.write_text(SUPPLIED)
    token_file.chmod(0o600)
    result = run_add(env, f"--token-file={token_file}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert SUPPLIED not in result.stdout + result.stderr
    assert "--method POST" not in (root / "gh_calls").read_text()
