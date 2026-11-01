# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Factory floor declarations (OMN-21733).

The contract also requires test_factory_floor_web_runtime.py: it runs nginx -t,
HTTP requests and Compose refusal checks, and fails if Docker is unavailable.
"""

from __future__ import annotations

import json
import os
import re
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
REFUSAL_MESSAGES = {
    "FACTORY_FLOOR_BIND_ADDR": "set FACTORY_FLOOR_BIND_ADDR to the host tailnet address",
    "FACTORY_FLOOR_PORT": "set FACTORY_FLOOR_PORT",
    "FACTORY_FLOOR_SERVE_DIR": "set FACTORY_FLOOR_SERVE_DIR",
}
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
    assert re.fullmatch(r"nginx:1\.29-alpine@sha256:[0-9a-f]{64}", service["image"])


def test_nginx_serves_only_get_with_no_store_and_a_same_origin_policy() -> None:
    conf = NGINX.read_text(encoding="utf-8")
    assert "listen 8080;" in conf
    assert "gzip on;" in conf
    assert "limit_except GET HEAD" in conf
    assert 'Cache-Control "no-store"' in conf
    assert "default-src 'self'" in conf
    assert "root /srv/site;" in conf
    assert "try_files $uri $uri/index.html =404;" in conf


def _location_block(conf: str, header: str) -> str:
    start = conf.index(header)
    depth = 0
    for index in range(conf.index("{", start), len(conf)):
        depth += {"{": 1, "}": -1}.get(conf[index], 0)
        if depth == 0:
            return conf[start : index + 1]
    raise AssertionError(f"unterminated block {header}")


def test_only_the_decision_map_directory_may_load_two_pinned_cdn_scripts() -> None:
    # These declaration checks complement the contract's mandatory HTTP probe.
    conf = NGINX.read_text(encoding="utf-8")
    cdn = "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"
    elk = "https://cdn.jsdelivr.net/npm/elkjs@0.9.2/lib/elk.bundled.js"
    policies = re.findall(r'Content-Security-Policy "([^"]+)"', conf)
    assert len(policies) == 2
    default, scoped = policies
    assert "http" not in default
    assert f"script-src 'self' 'unsafe-inline' {cdn} {elk};" in scoped
    assert scoped.count("http") == 2
    block = _location_block(conf, "location /reports/decision-map/ {")
    assert scoped in block
    assert "limit_except GET HEAD" in block
    assert 'Cache-Control "no-store"' in block
    assert 'X-Content-Type-Options "nosniff"' in block
    # nginx tries the regex locations after the longest prefix match and lets a regex win unless the prefix is
    # `^~`, so a plain prefix keeps the dot-file rule in force for the directory whatever the order in the file.
    assert "location ^~" not in conf
    assert re.search(r"location\s+~\s+/\\\.\s*\{\s*return 404;", conf)
    assert re.search(r"location\s+/reports/decision-map/\s*\{", conf)


@pytest.mark.parametrize("empty", [False, True], ids=["unset", "empty"])
@pytest.mark.parametrize("missing", REQUIRED)
def test_compose_refuses_to_render_when_a_required_value_is_missing(
    missing: str, empty: bool
) -> None:
    values = {k: v for k, v in RENDER_ENV.items() if k != missing}
    if empty:
        values[missing] = ""
    done = _compose_config(values)
    assert done.returncode != 0
    assert REFUSAL_MESSAGES[missing] in done.stderr


def test_every_refusal_message_is_the_one_the_compose_file_declares() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert set(REFUSAL_MESSAGES) == set(REQUIRED)
    for name, message in REFUSAL_MESSAGES.items():
        assert f"${{{name}:?{message}}}" in text


def test_compose_renders_one_published_port_on_the_given_address() -> None:
    done = _compose_config(RENDER_ENV, "--format", "json")
    assert done.returncode == 0, done.stderr
    published = json.loads(done.stdout)["services"]["factory-floor-web"]["ports"]
    assert [(p["host_ip"], str(p["published"]), p["target"]) for p in published] == [
        ("192.0.2.1", "8088", 8080)
    ]
