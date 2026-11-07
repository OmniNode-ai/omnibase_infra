# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The runner hook mount drift check run as the cron runs it (OMN-22478).

Runs ``python -m omnibase_infra.nodes.node_deploy_runners.runtime_hook_mount_drift``
in a subprocess against the real ``DockerCli`` and real host files; only the
``docker`` executable on PATH is a stand-in that answers ps, inspect and exec.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SCRIPTS = ("runner-job-started.sh", "entrypoint.sh", "healthcheck.sh")
MODULE = "omnibase_infra.nodes.node_deploy_runners.runtime_hook_mount_drift"
CONTAINER = "omninode-runner-1"


def _fake_docker(bin_dir: Path, hook_dir: Path, served: dict[str, str]) -> None:
    inspect = {
        "Config": {
            "Env": [
                f"ACTIONS_RUNNER_HOOK_JOB_STARTED={hook_dir}/runner-job-started.sh"
            ],
            "Entrypoint": [f"{hook_dir}/entrypoint.sh"],
            "Healthcheck": {"Test": ["CMD-SHELL", f"{hook_dir}/healthcheck.sh"]},
        }
    }
    (bin_dir / "inspect.json").write_text(json.dumps([inspect]))
    (bin_dir / "sums.json").write_text(json.dumps(served))
    exe = bin_dir / "docker"
    exe.write_text(
        f"""#!{sys.executable}
import json, sys
here = {str(bin_dir)!r}
cmd = sys.argv[1]
if cmd == "ps":
    print({CONTAINER!r})
elif cmd == "inspect":
    print(open(here + "/inspect.json").read())
elif cmd == "exec":
    name = sys.argv[-1].rsplit("/", 1)[-1]
    print(json.load(open(here + "/sums.json"))[name], sys.argv[-1])
"""
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path: Path, served: dict[str, str]) -> subprocess.CompletedProcess[str]:
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    host: dict[str, str] = {}
    for script in SCRIPTS:
        (hook_dir / script).write_text(f"#!/bin/sh\n# {script}\n")
        host[script] = hashlib.sha256((hook_dir / script).read_bytes()).hexdigest()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_docker(bin_dir, hook_dir, served or host)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            MODULE,
            "--hook-dir",
            str(hook_dir),
            "--container-match",
            "^omninode-runner-",
            "--state-file",
            str(tmp_path / "state"),
            "--docker-timeout",
            "10",
            "--alert-env-file",
            str(tmp_path / "absent.env"),
            "--notify-timeout",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120,
    )


def test_cli_exits_zero_when_the_executed_hooks_equal_the_host_copies(
    tmp_path: Path,
) -> None:
    done = _run(tmp_path, {})
    assert done.returncode == 0, done.stderr
    assert '"outcome":"ok"' in done.stdout


def test_cli_exits_one_when_an_executed_hook_differs_from_the_host_copy(
    tmp_path: Path,
) -> None:
    served = dict.fromkeys(SCRIPTS, "0" * 64)
    done = _run(tmp_path, served)
    assert done.returncode == 1, done.stderr
    assert '"outcome":"drift"' in done.stdout
    assert CONTAINER in done.stdout
