# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the lifecycle adapter with the recorded retained workspace sizes."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "docker/runners/runner-job-started.sh"
RECORDED_BYTES = 6_593_306_624
pytestmark = pytest.mark.unit


def _replay(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    runner = tmp_path / "runner"
    work = runner / "_work"
    incoming = work / "incoming/incoming"
    incoming.mkdir(parents=True)
    (incoming / "stale").write_text("reset positive control")
    (runner / ".runner").write_text("identity positive control")
    sizes = {
        ".uv-cache": 2_559_918_080,
        "_tool": 1_740_861_440,
        ".pnpm-store": 692_617_216,
    }
    sizes["completed/completed"] = RECORDED_BYTES - sum(sizes.values())
    for name, size in sizes.items():
        directory = work / name
        directory.mkdir(parents=True)
        with (directory / "retained").open("wb") as stream:
            stream.truncate(size)
    env = dict(os.environ)
    env.update(
        RUNNER_HOME=str(runner),
        GITHUB_WORKSPACE=str(incoming),
        RUNNER_NAME="retention-replay",
        TMPDIR=str(tmp_path),
        DISK_ADMISSION_DF_OVERRIDE_KB="52428800",
        RUNNER_DISK_ADMISSION_PAUSE_DIR=str(tmp_path / "pause"),
        GITHUB_REPOSITORY="",
        OMNI_GIT_MIRROR_HOST="",
        CAPTURE=str(tmp_path / "request.json"),
    )
    return runner, incoming, env


def _stub(tmp_path: Path) -> Path:
    command = tmp_path / "onex"
    command.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
request = json.loads(args[args.index('--payload') + 1])
work = pathlib.Path(request['runner_home']) / '_work'
pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps({
    'args': args, 'request': request,
    'stale_at_dispatch': (pathlib.Path(request['workspace']) / 'stale').exists(),
    'bytes_at_dispatch': sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
}))
print(json.dumps({'ok': True, 'runtime_mode': 'local',
    'dispatch_result': {'status': 'completed', 'payload': {
    'status': os.environ.get('RETENTION_RESULT', 'completed'), 'dry_run': False,
    'bytes_before': 6593306645, 'bytes_after': 0, 'bytes_removed': 6593306645}}}))
""")
    command.chmod(0o755)
    return command


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(HOOK)],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_start_dispatches_recorded_case_before_incoming_reset(tmp_path: Path) -> None:
    runner, incoming, env = _replay(tmp_path)
    env["RUNNER_RETENTION_PYTHON"] = str(_stub(tmp_path))
    result = _run(env)
    assert result.returncode == 0, result.stdout + result.stderr
    capture = Path(env["CAPTURE"])
    assert capture.is_file(), result.stdout + result.stderr
    observed = json.loads(capture.read_text())
    assert observed["bytes_at_dispatch"] >= RECORDED_BYTES
    assert observed["stale_at_dispatch"] is True
    assert observed["request"] == {
        "runner_home": str(runner),
        "workspace": str(incoming),
        "phase": "job_started",
        "lifecycle_boundary": True,
        "active_workspaces": [],
        "dry_run": False,
    }
    assert observed["args"][:2] == ["-m", "omnimarket.adapters.codex.runtime_client"]
    assert (
        "node_runner_workspace_retention_effect.retain_runner_workspace"
        in observed["args"]
    )
    assert "local" in observed["args"]
    assert not (incoming / "stale").exists()
    assert (runner / ".runner").read_text() == "identity positive control"


@pytest.mark.parametrize("status", ["failed", "refused", "locked"])
def test_unsatisfied_retention_stops_before_reset(tmp_path: Path, status: str) -> None:
    _, incoming, env = _replay(tmp_path)
    env.update(RUNNER_RETENTION_PYTHON=str(_stub(tmp_path)), RETENTION_RESULT=status)
    result = _run(env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert (incoming / "stale").is_file()


def test_unconfigured_pool_keeps_incoming_reset(tmp_path: Path) -> None:
    _, incoming, env = _replay(tmp_path)
    env.pop("RUNNER_RETENTION_PYTHON", None)
    result = _run(env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (incoming / "stale").exists()
    assert not Path(env["CAPTURE"]).exists()


def test_h053_services_require_external_runtime() -> None:
    compose = yaml.safe_load(
        (ROOT / "docker/docker-compose.runners-hp053-ci-runner.yml").read_text()
    )
    services = compose["services"]
    assert len(services) == 8
    for service in services.values():
        assert (
            service["environment"].get("RUNNER_RETENTION_PYTHON")
            == "/opt/onex-runner-retention/bin/python"
        )
        mounts = [
            m
            for m in service["volumes"]
            if isinstance(m, dict) and m.get("target") == "/opt/onex-runner-retention"
        ]
        assert len(mounts) == 1
        assert mounts[0]["read_only"] is True
        assert mounts[0]["bind"]["create_host_path"] is False
        assert "RUNNER_RETENTION_RUNTIME_HOST_DIR:?" in mounts[0]["source"]


def test_missing_runtime_refuses_before_reset(tmp_path: Path) -> None:
    _, incoming, env = _replay(tmp_path)
    env["RUNNER_RETENTION_PYTHON"] = str(tmp_path / "not-installed")
    result = _run(env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert (incoming / "stale").is_file()
    assert not Path(env["CAPTURE"]).exists()


@pytest.mark.parametrize("workspace", ["work_root", "filesystem_root"])
def test_workspace_roots_are_refused(tmp_path: Path, workspace: str) -> None:
    runner, incoming, env = _replay(tmp_path)
    env["GITHUB_WORKSPACE"] = str(runner / "_work") if workspace == "work_root" else "/"
    env["RUNNER_RETENTION_PYTHON"] = str(_stub(tmp_path))
    result = _run(env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "Refusing to clean workspace outside" in result.stderr
    assert (incoming / "stale").is_file()
    assert not Path(env["CAPTURE"]).exists()
