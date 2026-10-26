# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's candidate-probe override stops compose from rebuilding (OMN-21279).

Docker Compose v5.1.0 rebuilds a service and its ``service:`` build dependency on
every ``docker compose run`` when the service build declares
``additional_contexts: service:<dep>``, which is how the dev lane composes its
runtime image. ``candidate_probe_file_args`` adds a temporary override that resets
``build`` and pins the already-built ``<project>-<service>`` image.

This module drives the real context manager against a compose fixture shaped like
the dev lane's composed runtime build, then asks the real compose client what it
would do. Only ``config`` and ``run --dry-run`` are invoked: nothing is built,
created or started, and the fixture project is not a declared lane.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane

PROJECT = "probe-fixture-omn21279"
SERVICE = "svc"
DEPENDENCY = "dep"

COMPOSE_FIXTURE = f"""\
services:
  {DEPENDENCY}:
    build:
      context: .
  {SERVICE}:
    build:
      context: .
      additional_contexts:
        infra-runtime: "service:{DEPENDENCY}"
"""


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _docker_compose_available(),
        reason="docker compose is required for the non-mutating compose probe check",
    ),
]


@pytest.fixture
def lane_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Point the executor's lane at the fixture compose file."""
    compose = tmp_path / "compose.yml"
    compose.write_text(COMPOSE_FIXTURE)
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    args = ["-f", str(compose)]
    monkeypatch.setattr(executor_mod, "_compose_file_args", lambda lane: list(args))
    monkeypatch.setattr(
        executor_mod,
        "lane_config_for",
        lambda lane: SimpleNamespace(compose_project=PROJECT),
    )
    return args


def _compose(args: list[str], *tail: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args, "-p", PROJECT, *tail],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _dry_run_probe(args: list[str]) -> str:
    result = _compose(
        args,
        "run",
        "--dry-run",
        "--rm",
        "--no-deps",
        "--pull",
        "never",
        "-T",
        SERVICE,
        "true",
    )
    # Compose ends every dry-run `run` with "interactive run is not supported in
    # dry-run mode", so the plan it printed first is the evidence, not the exit code.
    return result.stdout + result.stderr


def test_fixture_without_override_builds_the_service_and_its_dependency(
    lane_files: list[str],
) -> None:
    """The control: this is the rebuild the override has to prevent."""
    rendered = json.loads(_compose(lane_files, "config", "--format", "json").stdout)
    assert rendered["services"][SERVICE]["build"]["additional_contexts"] == {
        "infra-runtime": f"service:{DEPENDENCY}"
    }
    output = _dry_run_probe(lane_files)
    assert f"Image {PROJECT}-{SERVICE} Building" in output
    assert f"Image {PROJECT}-{DEPENDENCY} Building" in output


def test_override_pins_the_built_image_and_drops_the_build(
    lane_files: list[str],
) -> None:
    with executor_mod.candidate_probe_file_args(
        EnumRuntimeLane.DEV, SERVICE
    ) as file_args:
        assert file_args[: len(lane_files)] == lane_files
        rendered = json.loads(_compose(file_args, "config", "--format", "json").stdout)
    service = rendered["services"][SERVICE]
    assert "build" not in service
    assert service["image"] == f"{PROJECT}-{SERVICE}"
    # The override names one service only; the dependency is untouched.
    assert rendered["services"][DEPENDENCY]["build"]["context"]


def test_probe_run_with_override_plans_no_build(lane_files: list[str]) -> None:
    with executor_mod.candidate_probe_file_args(
        EnumRuntimeLane.DEV, SERVICE
    ) as file_args:
        output = _dry_run_probe(file_args)
    assert "Building" not in output
    assert f"Container {PROJECT}-{SERVICE}-run-" in output


def test_override_file_is_removed_after_the_probe(lane_files: list[str]) -> None:
    with executor_mod.candidate_probe_file_args(
        EnumRuntimeLane.DEV, SERVICE
    ) as file_args:
        override = Path(file_args[-1])
        assert override.is_file()
    assert not override.exists()
    assert not override.parent.exists()
