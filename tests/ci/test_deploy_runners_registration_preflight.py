# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Run the rolling entrypoint against recording SSH and registry boundaries.

The cache probe deliberately succeeds: a directory alone cannot establish
that the recreated runner can restore a working registration. No live runner
or credential is used by these execution tests.
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
FAKE_TOKEN = "test-registration-handle"


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
        ROLL_ONLINE_MAX_SECONDS="4",
        ROLL_ONLINE_INTERVAL_SECONDS="1",
        ROLL_SKIP_RETRY_PASSES="0",
    )
    programs = {
        "ssh": r"""
            import os, subprocess, sys
            from pathlib import Path
            root = Path(os.environ["RUNNER_TEST_ROOT"])
            command = sys.argv[2]
            with (root / "calls").open("a") as log:
                log.write("ssh\n")
            if "systemctl show" in command:
                print(os.environ["RUNNER_TEST_SLICE"])
            elif " config " in command:
                print("    RUNNER_LABELS: test-label\n    GITHUB_ORG_URL: https://github.com/FakeOrg")
            elif "docker inspect" in command:
                print("true")
            elif "docker top" in command:
                print("0")
            elif "docker ps" in command:
                print("omninode-runner-2 running")
            elif "--force-recreate" in command:
                command = command.replace("cd /home/jonah/.omnibase/runners", "cd " + str(root))
                sys.exit(subprocess.run(["bash", "-c", command], check=False).returncode)
        """,
        "docker": r"""
            import os, sys
            from pathlib import Path
            root = Path(os.environ["RUNNER_TEST_ROOT"])
            if "--force-recreate" in sys.argv:
                if os.environ.get("RUNNER_TOKEN") != "test-registration-handle":
                    print("recreate did not receive the registration handle", file=sys.stderr)
                    sys.exit(1)
                (root / "recreated").write_text(sys.argv[-1])
        """,
        "gh": r"""
            import json, os
            from pathlib import Path
            root = Path(os.environ["RUNNER_TEST_ROOT"])
            status = "online"
            if (root / "recreated").exists():
                polls = root / "polls"
                count = int(polls.read_text()) + 1 if polls.exists() else 1
                polls.write_text(str(count))
                status = "offline" if count < 2 or os.environ.get("RUNNER_TEST_OFFLINE") else "online"
            print(json.dumps({"runners": [{"name": "omninode-runner-2", "status": status,
                                         "busy": bool(os.environ.get("RUNNER_TEST_BUSY"))}]}))
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
    # Match the existing slice installation readback without touching systemd.
    values = []
    for line in (
        (ROOT / "docker/runners/systemd/omnirunners.slice").read_text().splitlines()
    ):
        key, _, value = line.partition("=")
        if key in {"MemoryHigh", "MemoryMax", "MemorySwapMax", "CPUWeight"}:
            if value.endswith("G"):
                value = str(int(value[:-1]) * 1024**3)
            elif value.endswith("M"):
                value = str(int(value[:-1]) * 1024**2)
            values.append(f"{key}={value}")
    assert values
    env["RUNNER_TEST_SLICE"] = "\n".join(sorted(values))
    return env, tmp_path


def run_roll(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--rolling", "--only=omninode-runner-2", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("token", [None, "", " \t\n"])
def test_missing_token_refuses_before_any_remote_action(
    boundary: tuple[dict[str, str], Path], dry_run: bool, token: str | None
) -> None:
    env, root = boundary
    if token is not None:
        env["DEPLOY_RUNNER_TOKEN"] = token
    result = run_roll(env, *(["--dry-run"] if dry_run else []))
    assert result.returncode != 0, result.stdout + result.stderr
    assert "DEPLOY_RUNNER_TOKEN" in result.stderr
    assert not (root / "calls").exists(), "preflight must precede sync and all SSH"
    assert not (root / "recreated").exists()


@pytest.mark.parametrize("source", ["environment", "file"])
def test_supplied_token_reaches_recreate_and_online_poll(
    boundary: tuple[dict[str, str], Path], source: str
) -> None:
    env, root = boundary
    args = []
    if source == "environment":
        env["DEPLOY_RUNNER_TOKEN"] = FAKE_TOKEN
    else:
        token_file = root / "operator-supplied-token"
        token_file.write_text(FAKE_TOKEN)
        token_file.chmod(0o600)
        args.append(f"--token-file={token_file}")
    result = run_roll(env, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "recreated").read_text() == "omninode-runner-2"
    assert int((root / "polls").read_text()) >= 2
    assert "back online after ~2s" in result.stdout
    assert FAKE_TOKEN not in result.stdout + result.stderr
    for artifact in ("calls", "recreated", "polls"):
        assert FAKE_TOKEN not in (root / artifact).read_text()


def test_whitespace_token_file_blocks_even_with_an_environment_token(
    boundary: tuple[dict[str, str], Path],
) -> None:
    env, root = boundary
    env["DEPLOY_RUNNER_TOKEN"] = FAKE_TOKEN
    token_file = root / "operator-supplied-token"
    token_file.write_text(" \t\n")
    result = run_roll(env, f"--token-file={token_file}")
    assert result.returncode != 0
    assert "DEPLOY_RUNNER_TOKEN" in result.stderr
    assert not (root / "calls").exists()


def test_supplied_token_dry_run_never_recreates(
    boundary: tuple[dict[str, str], Path],
) -> None:
    env, root = boundary
    env["DEPLOY_RUNNER_TOKEN"] = FAKE_TOKEN
    result = run_roll(env, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--force-recreate" in result.stdout
    assert not (root / "recreated").exists()
    assert FAKE_TOKEN not in result.stdout + result.stderr


def test_supplied_token_does_not_override_busy_check(
    boundary: tuple[dict[str, str], Path],
) -> None:
    env, root = boundary
    env.update(DEPLOY_RUNNER_TOKEN=FAKE_TOKEN, RUNNER_TEST_BUSY="1")
    result = run_roll(env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Still busy" in result.stderr
    assert not (root / "recreated").exists()


def test_runner_that_stays_offline_halts_after_recreate(
    boundary: tuple[dict[str, str], Path],
) -> None:
    env, root = boundary
    env.update(DEPLOY_RUNNER_TOKEN=FAKE_TOKEN, RUNNER_TEST_OFFLINE="1")
    result = run_roll(env)
    assert result.returncode != 0
    assert (root / "recreated").read_text() == "omninode-runner-2"
    assert "HALTED at omninode-runner-2" in result.stderr


def test_production_online_window_is_240_seconds() -> None:
    assert (
        'ROLL_ONLINE_MAX_SECONDS="${ROLL_ONLINE_MAX_SECONDS:-240}"'
        in SCRIPT.read_text()
    )


@pytest.mark.live_contact(
    "tests/ci/fixtures/rolling_runner_registration_preflight.json"
)
@pytest.mark.parametrize("mode", ["normal", "dry-run"])
def test_replay_actual_tokenless_cli_refusal(
    recorded_response: dict[str, object], mode: str
) -> None:
    """Replay captured, unstubbed preflight failures; no live registration claim."""
    responses = recorded_response["responses"]
    assert isinstance(responses, dict)
    response = responses[mode]
    assert response["returncode"] == 1
    assert response["sync_started"] is False
    assert response["recreate_started"] is False
    assert "DEPLOY_RUNNER_TOKEN" in response["stderr"]
    assert "before recreating any container" in response["stderr"]
    assert ("--dry-run" in response["command"]) == (mode == "dry-run")
