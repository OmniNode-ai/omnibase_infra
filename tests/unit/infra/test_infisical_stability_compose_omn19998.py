# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render the standalone store without a Docker daemon or real secret inputs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

_COMPOSE = (
    Path(__file__).resolve().parents[3]
    / "docker/docker-compose.infisical-stability.yml"
)
_REQUIRED = {
    "POSTGRES_PASSWORD_URLENC": "stability POSTGRES_PASSWORD_URLENC required",
    "VALKEY_PASSWORD": "stability VALKEY_PASSWORD required",
    "INFISICAL_ENCRYPTION_KEY": "stability INFISICAL_ENCRYPTION_KEY required",
    "INFISICAL_AUTH_SECRET": "stability INFISICAL_AUTH_SECRET required",
    "INFISICAL_SMTP_HOST": "INFISICAL_SMTP_HOST required",
    "INFISICAL_SMTP_USERNAME": "INFISICAL_SMTP_USERNAME required",
    "INFISICAL_SMTP_PASSWORD": "INFISICAL_SMTP_PASSWORD required",
    "INFISICAL_SMTP_FROM_ADDRESS": "INFISICAL_SMTP_FROM_ADDRESS required",
    "INFISICAL_STABILITY_SITE_URL": "stability invite-link SITE_URL required",
    "INFISICAL_POSTGRES_HOST": "existing Postgres host required",
    "INFISICAL_VALKEY_HOST": "existing Valkey host required",
    "INFISICAL_NETWORK_NAME": "existing external network required",
    "INFISICAL_LANE_LABEL": "existing lane label required",
}


def _render(missing: str = "") -> subprocess.CompletedProcess[str]:
    # Deliberately exclude the host's credentials and Compose project overrides.
    env = {
        key: os.environ[key]
        for key in (
            "PATH",
            "HOME",
            "LANDING_REAL_DOCKER",
            "LANDING_COMPOSE_STATE",
            "ONEX_LANE",
        )
        if key in os.environ
    }
    env.update(dict.fromkeys(_REQUIRED, "synthetic-compose-input"))
    if missing:
        env.pop(missing)
    return subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            os.devnull,
            "-f",
            str(_COMPOSE),
            "config",
            "--format",
            "json",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_standalone_store_preserves_identity_and_mail() -> None:
    rendered = _render()
    assert rendered.returncode == 0, rendered.stderr
    config = json.loads(rendered.stdout)
    assert config["name"] == "omnibase-infra-infisical-stability"
    assert set(config["services"]) == {"infisical"}
    service = config["services"]["infisical"]
    assert service["container_name"] == "omnibase-infra-infisical-stability"
    assert service["image"] == "infisical/infisical:v0.146.0-postgres"
    assert service["ports"][0]["published"] == "8880"
    assert service["ports"][0]["target"] == 8080
    network = config["networks"]["stability"]
    assert network["name"] == "synthetic-compose-input"
    assert network["external"] is True
    env = service["environment"]
    assert "@synthetic-compose-input:5432/infisical_db" in env["DB_CONNECTION_URI"]
    assert "@synthetic-compose-input:6379" in env["REDIS_URL"]
    for name in ("HOST", "USERNAME", "PASSWORD", "FROM_ADDRESS"):
        assert env[f"SMTP_{name}"] == "synthetic-compose-input"
    assert env["SMTP_PORT"] == "587"
    assert env["SMTP_REQUIRE_TLS"] == "true"
    assert env["SMTP_FROM_NAME"] == "OmniNode"
    assert service["restart"] == "unless-stopped"
    assert "depends_on" not in service
    assert "volumes" not in config


@pytest.mark.parametrize("missing", _REQUIRED)
def test_missing_required_input_refuses_render(missing: str) -> None:
    rendered = _render(missing)
    assert rendered.returncode != 0
    assert (
        f"required variable {missing} is missing a value: {_REQUIRED[missing]}"
        in rendered.stderr
    ), rendered.stderr
