# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: runner hook files survive a redeploy, and a broken hook stops the runner.

Why this exists. ``scripts/deploy-runners.sh`` rsyncs ``docker/runners/`` to the
runner host, and rsync replaces each file (new inode). A container that
bind-mounts the hook files one at a time keeps the deleted inode. On Docker
Desktop that path then reads "No such file or directory", so every job the
runner accepts fails at "Set up runner". ``omninode-studio-runner-1`` failed 9
jobs between 15:32Z and 16:30Z on 2026-10-16 that way while Docker marked it
unhealthy and its listener kept taking work, because ``unhealthy`` stops nothing.
``rsync --inplace`` was tried and rejected: Docker Desktop then served a
truncated 3-byte file.

Two invariants, pinned here:

1. No runner service bind-mounts a hook file singly. Every service mounts the
   ``./runners`` directory read-only at ``/opt/onex-runner-hooks`` and points the
   hook env var, the entrypoint wrapper and the healthcheck inside it.
2. ``healthcheck.sh`` layer 0 fails when the job-started hook is missing, empty
   or not a script, and stops the container once the defect outlasts a grace so
   the listener stops polling GitHub.

The real-docker negative control (old layout strands, new layout does not) lives
in ``tests/integration/runner_hooks/``.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKER_DIR = _REPO_ROOT / "docker"
_HEALTHCHECK = _DOCKER_DIR / "runners" / "healthcheck.sh"
_RUNNER_COMPOSE_FILES = sorted(
    [
        *_DOCKER_DIR.glob("docker-compose.runners*.yml"),
        _DOCKER_DIR / "docker-compose.model-review-canary.yml",
    ]
)
_HOOK_DIR_MOUNT = "./runners:/opt/onex-runner-hooks:ro"
_HOOK_DIR = "/opt/onex-runner-hooks"
_HOOK_ENV = "ACTIONS_RUNNER_HOOK_JOB_STARTED"


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Passthrough constructor for Docker Compose `!override` / `!reset` tags."""
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that unwraps compose override tags."""


_ComposeLoader.add_constructor("!override", _construct_compose_value)
_ComposeLoader.add_constructor("!reset", _construct_compose_value)


def _services() -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for path in _RUNNER_COMPOSE_FILES:
        loader = _ComposeLoader(path.read_text(encoding="utf-8"))
        try:
            data: dict[str, Any] = loader.get_single_data()
        finally:
            loader.dispose()
        for name, service in (data.get("services") or {}).items():
            rows.append((path.name, name, service or {}))
    return rows


def _env(service: dict[str, Any]) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        return dict(item.split("=", 1) for item in env if "=" in item)
    return {str(k): str(v) for k, v in env.items()}


def _hook_services() -> list[tuple[str, str, dict[str, Any]]]:
    return [row for row in _services() if _HOOK_ENV in _env(row[2])]


@pytest.mark.unit
def test_the_population_is_not_empty() -> None:
    """A zero-violation result over zero services would prove nothing."""
    rows = _hook_services()
    assert len(rows) >= 40, f"only {len(rows)} hook services found"
    files = {file for file, _, _ in rows}
    assert {
        "docker-compose.runners.yml",
        "docker-compose.runners-omnipc2-ci-runner.yml",
        "docker-compose.runners-omninode-studio-runner.yml",
    } <= files


@pytest.mark.unit
def test_no_service_bind_mounts_a_hook_file_singly() -> None:
    """The stranded-inode layout: ``./runners/<file>.sh:/usr/local/bin/<file>.sh``."""
    offenders = []
    for file, name, service in _services():
        for entry in service.get("volumes") or []:
            text = (
                f"{entry.get('source', '')}:{entry.get('target', '')}"
                if isinstance(entry, dict)
                else str(entry)
            )
            if text.startswith("./runners/"):
                offenders.append(f"{file}:{name}: {text}")
    assert not offenders, "single-file runner hook mounts:\n" + "\n".join(offenders)


@pytest.mark.unit
def test_every_hook_service_mounts_the_hook_directory_read_only() -> None:
    missing = [
        f"{file}:{name}"
        for file, name, service in _hook_services()
        if _HOOK_DIR_MOUNT not in (service.get("volumes") or [])
    ]
    assert not missing, f"services without {_HOOK_DIR_MOUNT}:\n" + "\n".join(missing)


@pytest.mark.unit
def test_hook_env_healthcheck_and_entrypoint_point_inside_the_hook_directory() -> None:
    bad = []
    for file, name, service in _hook_services():
        hook = _env(service)[_HOOK_ENV]
        if hook != f"{_HOOK_DIR}/runner-job-started.sh":
            bad.append(f"{file}:{name}: {_HOOK_ENV}={hook}")
        test = (service.get("healthcheck") or {}).get("test")
        if test is not None and test != ["CMD-SHELL", f"{_HOOK_DIR}/healthcheck.sh"]:
            bad.append(f"{file}:{name}: healthcheck.test={test}")
        # A service with no entrypoint falls back to the image ENTRYPOINT, the
        # baked-in copy under /usr/local/bin, so a redeployed entrypoint.sh would
        # never run. Every service names the mounted one, directly or by
        # `exec` from a root-phase wrapper.
        entrypoint = " ".join(str(part) for part in service.get("entrypoint") or [])
        if entrypoint != f"{_HOOK_DIR}/entrypoint.sh" and (
            f"exec {_HOOK_DIR}/entrypoint.sh" not in entrypoint
        ):
            bad.append(
                f"{file}:{name}: entrypoint={entrypoint!r} is outside {_HOOK_DIR}"
            )
    assert not bad, "\n".join(bad)


def _run_healthcheck(
    tmp_path: Path,
    *,
    hook: Path | None,
    grace: str = "0",
    stand_down: str = "1",
    stamp_exists: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run healthcheck.sh with a stub ``docker`` that records its argv."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    docker_log = tmp_path / "docker.log"
    stub = stub_dir / "docker"
    stub.write_text(f'#!/bin/sh\necho "$@" >> "{docker_log}"\n', encoding="utf-8")
    stub.chmod(0o755)
    stamp = tmp_path / "hook-stamp"
    if stamp_exists:
        stamp.write_text("", encoding="utf-8")
    env = {
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "RUNNER_HOME": str(tmp_path / "no-runner-home"),
        "RUNNER_NAME": "test-runner-1",
        "RUNNER_HEALTH_HOOK_STAMP": str(stamp),
        "RUNNER_HEALTH_HOOK_STAND_DOWN_GRACE_SECONDS": grace,
        "RUNNER_HEALTH_HOOK_STAND_DOWN": stand_down,
    }
    if hook is not None:
        env[_HOOK_ENV] = str(hook)
    result = subprocess.run(
        ["bash", str(_HEALTHCHECK)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result, docker_log


def _wait_for_docker_call(docker_log: Path, seconds: float = 6.0) -> str:
    """The stand-down is detached with a 2 s delay; wait for the stub to log it."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if docker_log.exists() and docker_log.read_text(encoding="utf-8").strip():
            return docker_log.read_text(encoding="utf-8")
        time.sleep(0.25)
    return ""


@pytest.mark.unit
def test_missing_hook_fails_and_stops_the_container(tmp_path: Path) -> None:
    result, docker_log = _run_healthcheck(tmp_path, hook=tmp_path / "gone.sh")
    assert result.returncode == 1
    assert "gone.sh does not exist" in result.stdout
    assert "standing down" in result.stdout
    assert _wait_for_docker_call(docker_log).strip() == "stop test-runner-1"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("content", "expected"),
    [("", "is empty"), ("abc", "does not start with a shebang")],
)
def test_empty_or_truncated_hook_fails_and_stops(
    tmp_path: Path, content: str, expected: str
) -> None:
    """Docker Desktop served a truncated 3-byte file after ``rsync --inplace``."""
    hook = tmp_path / "hook.sh"
    hook.write_text(content, encoding="utf-8")
    result, docker_log = _run_healthcheck(tmp_path, hook=hook)
    assert result.returncode == 1
    assert expected in result.stdout
    assert _wait_for_docker_call(docker_log).strip() == "stop test-runner-1"


@pytest.mark.unit
def test_a_defect_inside_the_grace_reports_unhealthy_without_stopping(
    tmp_path: Path,
) -> None:
    """A deploy that is mid-sync must not take a runner out of the pool."""
    result, docker_log = _run_healthcheck(
        tmp_path, hook=tmp_path / "gone.sh", grace="3600"
    )
    assert result.returncode == 1
    assert "stand-down after 3600s" in result.stdout
    time.sleep(3.0)
    assert not docker_log.exists()


@pytest.mark.unit
def test_stand_down_can_be_disabled(tmp_path: Path) -> None:
    result, docker_log = _run_healthcheck(
        tmp_path, hook=tmp_path / "gone.sh", stand_down="0"
    )
    assert result.returncode == 1
    assert "stand-down disabled" in result.stdout
    time.sleep(3.0)
    assert not docker_log.exists()


@pytest.mark.unit
def test_an_intact_hook_passes_layer_zero_and_clears_the_stamp(tmp_path: Path) -> None:
    """Positive control: with a good hook the first failure is the listener layer."""
    hook = tmp_path / "hook.sh"
    hook.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    result, docker_log = _run_healthcheck(tmp_path, hook=hook, stamp_exists=True)
    assert result.returncode == 1
    assert "Runner.Listener not running" in result.stdout
    assert "OMN-19998" not in result.stdout
    assert not (tmp_path / "hook-stamp").exists()
    time.sleep(3.0)
    assert not docker_log.exists()


@pytest.mark.unit
def test_runners_without_a_hook_skip_layer_zero(tmp_path: Path) -> None:
    result, _ = _run_healthcheck(tmp_path, hook=None)
    assert result.returncode == 1
    assert "Runner.Listener not running" in result.stdout
