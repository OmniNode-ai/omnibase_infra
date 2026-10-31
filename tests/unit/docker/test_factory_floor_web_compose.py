# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The factory floor web service binds one address, serves read-only and needs its deployment values (OMN-21733)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"
COMPOSE = DOCKER_DIR / "docker-compose.factory-floor-web.yml"
NGINX = DOCKER_DIR / "factory-floor-web" / "nginx.conf"
ALL_INTERFACES = ".".join(("0",) * 4)
REQUIRED = (
    "FACTORY_FLOOR_BIND_ADDR",
    "FACTORY_FLOOR_PORT",
    "FACTORY_FLOOR_SERVE_DIR",
)
RENDER_ENV = {
    "FACTORY_FLOOR_BIND_ADDR": "192.0.2.1",
    "FACTORY_FLOOR_PORT": "8088",
    "FACTORY_FLOOR_SERVE_DIR": "/srv/example-site",
}


def _service() -> dict[str, Any]:
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    service = document["services"]["factory-floor-web"]
    assert isinstance(service, dict)
    return service


def _compose_config(
    env_values: dict[str, str], *extra: str
) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", ""), **env_values}
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", *extra],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


def test_published_port_names_the_bind_address_and_has_no_default() -> None:
    ports = _service()["ports"]
    assert isinstance(ports, list)
    assert len(ports) == 1
    port = str(ports[0])
    assert port.startswith("${FACTORY_FLOOR_BIND_ADDR:?")
    assert port.endswith(":8080")
    assert ALL_INTERFACES not in COMPOSE.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", REQUIRED)
def test_every_deployment_value_is_required_without_a_default(name: str) -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert f"${{{name}:?" in text
    assert f"${{{name}:-" not in text
    assert f"${{{name}-" not in text


def test_serve_directory_is_mounted_read_only_and_the_container_is_locked_down() -> (
    None
):
    service = _service()
    volumes = [str(v) for v in service["volumes"]]
    serve = [v for v in volumes if "FACTORY_FLOOR_SERVE_DIR" in v]
    assert len(serve) == 1
    assert serve[0].endswith(":/srv/site:ro")
    assert all(v.endswith(":ro") for v in volumes)
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["user"] == "101:101"
    assert not str(service["image"]).endswith(":latest")


def test_nginx_serves_only_get_with_no_store_and_a_same_origin_policy() -> None:
    conf = NGINX.read_text(encoding="utf-8")
    assert "listen 8080;" in conf
    assert "gzip on;" in conf
    assert "limit_except GET HEAD" in conf
    assert 'Cache-Control "no-store"' in conf
    assert "default-src 'self'" in conf
    assert "root /srv/site;" in conf
    assert "try_files $uri $uri/index.html =404;" in conf


@pytest.mark.skipif(not _compose_available(), reason="docker compose is not available")
@pytest.mark.parametrize("missing", REQUIRED)
def test_compose_refuses_to_render_when_a_required_value_is_missing(
    missing: str,
) -> None:
    values = {k: v for k, v in RENDER_ENV.items() if k != missing}
    done = _compose_config(values)
    assert done.returncode != 0
    assert missing in done.stderr


@pytest.mark.skipif(not _compose_available(), reason="docker compose is not available")
def test_compose_renders_one_published_port_on_the_given_address() -> None:
    done = _compose_config(RENDER_ENV, "--format", "json")
    assert done.returncode == 0, done.stderr
    published = json.loads(done.stdout)["services"]["factory-floor-web"]["ports"]
    assert [(p["host_ip"], str(p["published"]), p["target"]) for p in published] == [
        ("192.0.2.1", "8088", 8080)
    ]
