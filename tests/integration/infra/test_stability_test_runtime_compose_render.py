# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-mutating compose render checks for the OMN-10281 stability-test lane."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILES = (
    "docker/docker-compose.infra.yml",
    "docker/docker-compose.stability-test.yml",
)
REQUIRED_RUNTIME_SERVICES = {
    "omninode-runtime",
    "runtime-effects",
    "runtime-worker",
}


def _docker_compose_command(*args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        COMPOSE_FILES[0],
        "-f",
        COMPOSE_FILES[1],
        "--profile",
        "runtime",
        *args,
    ]


pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker is required for non-mutating compose render validation",
)


@pytest.mark.integration
def test_stability_lane_runtime_services_render_with_runtime_profile() -> None:
    result = subprocess.run(
        _docker_compose_command("config", "--services"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    rendered_services = set(result.stdout.splitlines())

    assert rendered_services >= REQUIRED_RUNTIME_SERVICES


@pytest.mark.integration
def test_stability_lane_render_contains_isolated_runtime_identity() -> None:
    result = subprocess.run(
        _docker_compose_command("config"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    rendered_config = result.stdout

    assert "container_name: omninode-stability-test-runtime" in rendered_config
    assert "container_name: omninode-stability-test-runtime-effects" in rendered_config
    assert "container_name: omninode-stability-test-runtime-worker" in rendered_config
    assert "ONEX_ENVIRONMENT: stability-test" in rendered_config
    assert "ONEX_GROUP_ID: onex-stability-test-runtime-main" in rendered_config
    assert "KAFKA_INSTANCE_ID: stability-test-main" in rendered_config
    assert "image: runtime:stability-test-workspace" in rendered_config
