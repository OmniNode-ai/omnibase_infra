# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-mutating compose render checks for the OMN-10281 stability-test lane."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

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
COMPOSE_RENDER_ENV = {
    "INFISICAL_AUTH_SECRET": "render-only-infisical-auth-secret",
    "INFISICAL_DB_CONNECTION_URI": "postgresql://postgres:postgres@postgres:5432/infisical",
    "INFISICAL_ENCRYPTION_KEY": "render-only-infisical-encryption-key-32",
    "INFISICAL_REDIS_URL": "redis://valkey:6379",
    "LINEAR_API_KEY": "render-only-linear-api-key",
    "LLM_CODER_FAST_URL": "http://llm-coder-fast.invalid",
    "LLM_CODER_URL": "http://llm-coder.invalid",
    "LLM_DEEPSEEK_R1_URL": "http://llm-deepseek.invalid",
    "LLM_EMBEDDING_URL": "http://llm-embedding.invalid",
    "ONEX_REGISTRATION_AUTO_ACK": "false",
    "ONEX_SERVICE_CLIENT_SECRET": "render-only-client-secret",
    "POSTGRES_PASSWORD": "postgres",
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


def _compose_render_env() -> dict[str, str]:
    return {**os.environ, **COMPOSE_RENDER_ENV}


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
        env=_compose_render_env(),
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
        env=_compose_render_env(),
        text=True,
    )

    rendered_config = result.stdout
    rendered = yaml.safe_load(rendered_config)
    services = rendered["services"]
    main_ports = services["omninode-runtime"]["ports"]
    effects_ports = services["runtime-effects"]["ports"]
    networks = rendered["networks"]

    assert "container_name: omninode-stability-test-runtime" in rendered_config
    assert "container_name: omninode-stability-test-runtime-effects" in rendered_config
    assert "container_name: omninode-stability-test-runtime-worker" in rendered_config
    assert (
        "container_name: omnibase-infra-stability-test-forward-migration"
        in rendered_config
    )
    assert "container_name: omnibase-forward-migration" not in rendered_config
    assert "ONEX_ENVIRONMENT: stability-test" in rendered_config
    assert "ONEX_BOX_ID: omninode-pc" in rendered_config
    assert (
        "ONEX_RUNTIME_ADDRESS: runtime://omninode-pc/stability-test/main"
        in rendered_config
    )
    assert (
        "ONEX_RUNTIME_ADDRESS: runtime://omninode-pc/stability-test/effects"
        in rendered_config
    )
    assert (
        "ONEX_RUNTIME_ADDRESS: runtime://omninode-pc/stability-test/worker"
        in rendered_config
    )
    assert "ONEX_RUNTIME_ID: stability-test-main" in rendered_config
    assert "ONEX_RUNTIME_ID: stability-test-effects" in rendered_config
    assert "ONEX_RUNTIME_ID: stability-test-worker" in rendered_config
    assert "ONEX_GROUP_ID: onex-stability-test-runtime-main" in rendered_config
    assert "KAFKA_INSTANCE_ID: stability-test-main" in rendered_config
    assert (
        "com.omninode.runtime.address: runtime://omninode-pc/stability-test/main"
        in rendered_config
    )
    assert (
        "com.omninode.runtime.address: runtime://omninode-pc/stability-test/effects"
        in rendered_config
    )
    assert (
        "com.omninode.runtime.address: runtime://omninode-pc/stability-test/worker"
        in rendered_config
    )
    assert "com.omninode.runtime.id: stability-test-main" in rendered_config
    assert "com.omninode.runtime.id: stability-test-effects" in rendered_config
    assert "com.omninode.runtime.id: stability-test-worker" in rendered_config
    assert "image: runtime:stability-test-workspace" in rendered_config
    assert {port["published"] for port in main_ports} == {"18085"}
    assert {port["published"] for port in effects_ports} == {"18086"}
    assert 'published: "8085"' not in rendered_config
    assert 'published: "8086"' not in rendered_config
    assert networks["omnibase-infra-network"]["name"] == (
        "omnibase-infra-stability-test-network"
    )
    assert networks["omnimemory-network"]["name"] == (
        "omnibase-infra-stability-test-omnimemory-network"
    )
    assert networks["omnimemory-network"].get("external", False) is False
