# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The cache ownership restore run as a periodic entry runs it (OMN-26113).

Runs ``python -m omnibase_infra.nodes.node_deploy_runners.runtime_runner_cache_ownership_restore`` in a
subprocess against the real ``DockerCacheCli``; only the ``docker`` executable on PATH is a stand-in that answers
ps, inspect and exec from a state file and records every chown.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

MODULE = (
    "omnibase_infra.nodes.node_deploy_runners.runtime_runner_cache_ownership_restore"
)
CONTAINER = "hp051-omnibase-ci-runner-9"
HOME = "/home/runner"
MOUNTS = [
    {"Type": "bind", "Destination": f"{HOME}/.npm"},
    {"Type": "bind", "Destination": f"{HOME}/.cache/pre-commit"},
    {"Type": "bind", "Destination": f"{HOME}/actions-runner/_work"},
    {"Type": "volume", "Destination": f"{HOME}/.runner-creds"},
]


def _fake_docker(bin_dir: Path, processes: list[str]) -> None:
    (bin_dir / "state.json").write_text(
        json.dumps({"processes": processes, "owned": False})
    )
    exe = bin_dir / "docker"
    exe.write_text(
        f"""#!{sys.executable}
import json, sys
here = {str(bin_dir)!r}
state = json.load(open(here + "/state.json"))
args = sys.argv[1:]
if args[0] == "ps":
    print({CONTAINER!r})
elif args[0] == "inspect":
    print(json.dumps([{{"Mounts": {MOUNTS!r}}}]))
elif args[0] == "exec":
    rest = args[1:]
    if rest[0] == "-u":
        rest = rest[2:]
    rest = rest[1:]  # the container name
    if rest[0] == "ps":
        print("COMMAND")
        print(chr(10).join(state["processes"]))
    elif rest[0] == "find":
        if not state["owned"]:
            print(rest[1])
    elif rest[0] == "chown":
        open(here + "/chown.log", "a").write(" ".join(rest) + chr(10))
        state["owned"] = True
        json.dump(state, open(here + "/state.json", "w"))
"""
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path: Path, processes: list[str]) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_docker(bin_dir, processes)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            MODULE,
            "--container-match",
            "^hp051-omnibase-ci-runner-",
            "--docker-timeout",
            "10",
            "--chown-timeout",
            "30",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120,
    )


def test_cli_restores_the_declared_cache_mounts_of_an_idle_runner(
    tmp_path: Path,
) -> None:
    done = _run(tmp_path, ["bash", "Runner.Listener"])
    assert done.returncode == 0, done.stderr
    assert '"outcome":"ok"' in done.stdout
    assert '"outcome":"restored"' in done.stdout
    calls = (tmp_path / "bin" / "chown.log").read_text().splitlines()
    assert calls == [
        f"chown -hR 1001:1001 {HOME}/.cache/pre-commit {HOME}/.npm",
    ]


def test_cli_never_writes_where_a_runner_worker_is_running(tmp_path: Path) -> None:
    done = _run(tmp_path, ["bash", "Runner.Listener", "Runner.Worker"])
    assert done.returncode == 0, done.stderr
    assert CONTAINER in json.loads(done.stdout.split("] ", 1)[1])["busy"]
    assert not (tmp_path / "bin" / "chown.log").exists()
