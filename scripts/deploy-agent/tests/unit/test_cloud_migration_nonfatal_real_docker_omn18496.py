# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18496: the cloud-migration gate against a real Docker daemon.

``test_cloud_migration_nonfatal_omn18496.py`` replaces every Docker boundary,
so it proves the executor's decisions and never that compose carries them out.
This file drives the same executor methods against real ``docker compose`` in a
throwaway project of its own: a warm ``onex-api`` reading ``omninode_cloud``,
a runtime service beside it, and a ``cloud-migration`` one-shot that exits 1.
The project name is unique per run and is torn down after it, so no declared
lane's containers are named.

It skips, saying why, where no Docker daemon answers or ``postgres:16-alpine``
is not already local; it never pulls an image.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, Phase, Scope
from deploy_agent.executor import DeployExecutor, ModelLaneConfig

pytestmark = [pytest.mark.unit, pytest.mark.real_docker]

IMAGE = "postgres:16-alpine"
# PID 1 exits on SIGTERM so a `stop --timeout 30` returns at once.
_SERVE = "trap 'exit 0' TERM; sleep 3600 & wait"
COMPOSE = f"""\
services:
  forward-migration:
    image: {IMAGE}
    profiles: ["runtime"]
    entrypoint: ["sh", "-c", "exit 0"]
    restart: "no"
  migration-gate:
    image: {IMAGE}
    profiles: ["runtime"]
    entrypoint: ["sh", "-c", "{_SERVE}"]
  cloud-migration-files:
    image: {IMAGE}
    profiles: ["runtime"]
    entrypoint: ["sh", "-c", "exit 0"]
    restart: "no"
  cloud-migration:
    image: {IMAGE}
    profiles: ["runtime"]
    entrypoint:
      - sh
      - -c
      - "echo 'ERROR: permission denied to create database' >&2; exit ${{CLOUD_MIGRATION_RC:-1}}"
    restart: "no"
    depends_on:
      cloud-migration-files:
        condition: service_completed_successfully
  omninode-runtime:
    image: {IMAGE}
    profiles: ["runtime"]
    entrypoint: ["sh", "-c", "{_SERVE}"]
  onex-api:
    image: {IMAGE}
    profiles: ["runtime"]
    entrypoint: ["sh", "-c", "{_SERVE}"]
    depends_on:
      cloud-migration:
        condition: service_completed_successfully
"""


def _docker_unavailable_reason() -> str:
    if shutil.which("docker") is None:
        return "no docker binary on PATH"
    for probe in (
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        ["docker", "image", "inspect", IMAGE],
    ):
        result = subprocess.run(
            probe, capture_output=True, text=True, check=False, timeout=30
        )
        if result.returncode != 0:
            return f"{' '.join(probe)} exited {result.returncode}"
    return ""


def _compose(
    compose_file: Path, project: str, *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "-p", project, *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def _states(compose_file: Path, project: str) -> dict[str, tuple[str, int | None]]:
    result = _compose(compose_file, project, "ps", "-a", "--format", "json")
    assert result.returncode == 0, result.stderr
    states: dict[str, tuple[str, int | None]] = {}
    for line in result.stdout.splitlines():
        if line.strip():
            row = json.loads(line)
            states[row["Service"]] = (row["State"], row.get("ExitCode"))
    return states


@pytest.fixture
def warm_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, str]]:
    """A project whose onex-api is serving, then a cloud one-shot that fails."""
    reason = _docker_unavailable_reason()
    if reason:
        pytest.skip(f"real Docker proof needs a daemon and a local {IMAGE}: {reason}")
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text(COMPOSE, encoding="utf-8")
    project = f"omn18496-proof-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("CLOUD_MIGRATION_RC", "0")
    try:
        up = _compose(compose_file, project, "--profile", "runtime", "up", "-d")
        assert up.returncode == 0, up.stderr
        assert _states(compose_file, project)["onex-api"][0] == "running"
        monkeypatch.setenv("CLOUD_MIGRATION_RC", "1")
        config = ModelLaneConfig(
            lane=EnumRuntimeLane.DEV,
            compose_files=(str(compose_file),),
            compose_project=project,
            postgres_container=f"{project}-postgres",
            runtime_health_targets=(("omninode-runtime", 0),),
            build_project=project,
        )
        monkeypatch.setattr(executor_mod, "lane_config_for", lambda lane: config)
        monkeypatch.setattr(
            executor_mod, "_compose_env", lambda *args: dict(os.environ)
        )
        monkeypatch.setattr(executor_mod, "REQUIRED_PROJECTION_TABLES", ())
        monkeypatch.setattr(
            executor_mod, "read_checkout_migration_fingerprint", lambda: "tree:new"
        )
        yield compose_file, project
    finally:
        _compose(
            compose_file,
            project,
            "--profile",
            "runtime",
            "down",
            "--volumes",
            "--remove-orphans",
            "--timeout",
            "5",
        )


def test_failed_cloud_one_shot_stops_only_onex_api_on_real_compose(
    warm_lane: tuple[Path, str], tmp_path: Path
) -> None:
    compose_file, project = warm_lane
    record_dir = tmp_path / "records"
    executor_mod.write_applied_migration_fingerprint(
        record_dir, EnumRuntimeLane.DEV, "tree:old"
    )
    executor = DeployExecutor(migration_record_dir=record_dir)

    executor._ensure_runtime_migrations_ready(
        lane=EnumRuntimeLane.DEV, timeout=60, build=False
    )

    assert "cloud-migration" in executor.cloud_migration_failure
    assert "permission denied to create database" in executor.cloud_migration_failure
    states = _states(compose_file, project)
    assert states["cloud-migration"] == ("exited", 1)
    assert states["onex-api"][0] == "exited", states["onex-api"]
    assert states["migration-gate"][0] == "running"
    assert states["omninode-runtime"][0] == "running"
    assert (
        executor_mod.read_applied_migration_fingerprint(record_dir, EnumRuntimeLane.DEV)
        == "tree:old"
    )


def test_runtime_up_after_a_failed_cloud_one_shot_leaves_onex_api_down(
    warm_lane: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    compose_file, project = warm_lane
    executor = DeployExecutor()
    monkeypatch.setattr(executor, "plan_oneshot_reruns", lambda lane: [])
    before = _states(compose_file, project)["omninode-runtime"]

    executor._compose_up(
        Phase.RUNTIME,
        Scope.RUNTIME,
        ["omninode-runtime", "onex-api"],
        lambda *args, **kwargs: None,
        lane=EnumRuntimeLane.DEV,
        build=False,
    )

    assert "cloud-migration" in executor.cloud_migration_failure
    states = _states(compose_file, project)
    assert before[0] == "running"
    assert states["omninode-runtime"][0] == "running"
    assert states["onex-api"][0] == "exited", states["onex-api"]
    assert states["cloud-migration"] == ("exited", 1)
