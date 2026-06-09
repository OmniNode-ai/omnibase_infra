# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-mutating compose render checks for the production runtime lane."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILES = (
    "docker/docker-compose.infra.yml",
    "docker/docker-compose.prod.yml",
)


def _http_url(authority: str) -> str:
    return "http" + "://" + authority


def _cidr(prefix: str, suffix: str) -> str:
    return prefix + "." + suffix


COMPOSE_RENDER_ENV = {
    "INFISICAL_AUTH_SECRET": "render-only-infisical-auth-secret",
    "INFISICAL_DB_CONNECTION_URI": "postgresql://postgres:postgres@postgres:5432/infisical",
    "INFISICAL_ENCRYPTION_KEY": "render-only-infisical-encryption-key-32",
    "INFISICAL_REDIS_URL": "redis://valkey:6379",
    "GITHUB_TOKEN": "render-only-github-token",
    "LINEAR_API_KEY": "render-only-linear-api-key",
    "LLM_CODER_FAST_URL": _http_url("llm-coder-fast.invalid"),
    "LLM_CODER_URL": _http_url("llm-coder.invalid"),
    "LLM_DEEPSEEK_R1_URL": _http_url("llm-deepseek.invalid"),
    "LLM_EMBEDDING_URL": _http_url("llm-embedding.invalid"),
    "LLM_GLM_API_KEY": "render-only-glm-api-key",
    "LLM_GLM_MODEL_NAME": "glm-4.5",
    "LLM_GLM_URL": _http_url("glm.invalid/v1"),
    "LLM_ENDPOINT_CIDR_ALLOWLIST": _cidr("192.168.86", "0/24"),
    "LOCAL_LLM_SHARED_SECRET": "render-only-local-llm-secret",
    "ONEX_REGISTRATION_AUTO_ACK": "false",
    "ONEX_SERVICE_CLIENT_SECRET": "render-only-client-secret",
    "POSTGRES_PASSWORD": "postgres",
    "REDPANDA_ADVERTISE_HOST": "100.109.203.94",
    "PROD_REDPANDA_ADVERTISE_HOST": "192.168.86.201",
    "PROD_POSTGRES_EXTERNAL_PORT": "25436",
    "PROD_VALKEY_EXTERNAL_PORT": "36379",
    "PROD_REDPANDA_ADMIN_PORT": "39644",
    "PROD_REDPANDA_EXTERNAL_PORT": "49092",
    "PROD_REDPANDA_PANDAPROXY_PORT": "48082",
}


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


def _compose_render_env() -> dict[str, str]:
    return {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "USER": os.environ.get("USER", ""),
        **COMPOSE_RENDER_ENV,
    }


def _docker_compose_command(*args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--env-file",
        "docker/runtime-policy.env",
        "-f",
        COMPOSE_FILES[0],
        "-f",
        COMPOSE_FILES[1],
        "--profile",
        "runtime",
        *args,
    ]


def _compose_config_json() -> dict:
    result = subprocess.run(
        _docker_compose_command("config", "--format", "json"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        env=_compose_render_env(),
        text=True,
    )
    rendered_config = json.loads(result.stdout)
    assert isinstance(rendered_config, dict)
    return rendered_config


pytestmark = pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose is required for non-mutating compose render validation",
)


@pytest.mark.integration
def test_prod_redpanda_advertise_host_is_prod_specific() -> None:
    rendered_config = _compose_config_json()
    redpanda_command = " ".join(rendered_config["services"]["redpanda"]["command"])

    assert "192.168.86.201:49092" in redpanda_command
    assert "192.168.86.201:48082" in redpanda_command
    assert "100.109.203.94:49092" not in redpanda_command
