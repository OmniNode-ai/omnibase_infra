# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19513: every runtime-image service carries a raised file-descriptor limit.

The dev lane's ``omninode-runtime`` ran with the container default of 1024 open
files. That one process joins one consumer group per declared projection topic
(280+ groups at boot, each holding two to three broker sockets) and opens a
Postgres pool per dispatched message, so its descriptor count climbs past 1024
about four minutes after every start. From there every write fails with
``OSError: [Errno 24] Too many open files``, the health endpoint reports 503, the
kernel shuts down in order (exit 0) and ``restart: unless-stopped`` starts it
again. The ``claude_hook_events`` consumer lives in that process, so its group sat
Empty or wedged most of each cycle and the table ran 18-20 minutes behind.

The limit belongs to the service definition, so it lives on the shared
``x-runtime-base`` anchor and in the catalog manifests that render the same
services, and every lane that overlays the base file inherits it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCKER_DIR = REPO_ROOT / "docker"
BASE_COMPOSE = DOCKER_DIR / "docker-compose.infra.yml"
CATALOG_SERVICES = DOCKER_DIR / "catalog" / "services"

RUNTIME_DOCKERFILE = "docker/Dockerfile.runtime"
RUNTIME_CATALOG_MANIFESTS = ("omninode-runtime", "runtime-effects", "runtime-worker")
MIN_NOFILE = 65535


def _runtime_services() -> dict[str, dict[str, Any]]:
    compose = yaml.safe_load(BASE_COMPOSE.read_text(encoding="utf-8"))
    return {
        name: service
        for name, service in compose["services"].items()
        if (service.get("build") or {}).get("dockerfile") == RUNTIME_DOCKERFILE
    }


def test_the_base_compose_declares_runtime_image_services() -> None:
    """Positive control: the selector finds the main, effects and worker runtimes."""
    names = set(_runtime_services())
    assert {"omninode-runtime", "runtime-effects", "runtime-worker"} <= names


@pytest.mark.parametrize("name", sorted(_runtime_services()))
def test_runtime_image_service_raises_the_nofile_limit(name: str) -> None:
    nofile = (_runtime_services()[name].get("ulimits") or {}).get("nofile")
    assert nofile is not None, f"{name} keeps the container default of 1024 open files"
    assert nofile["soft"] >= MIN_NOFILE
    assert nofile["hard"] >= nofile["soft"]


@pytest.mark.parametrize("manifest", RUNTIME_CATALOG_MANIFESTS)
def test_runtime_catalog_manifest_raises_the_nofile_limit(manifest: str) -> None:
    raw = yaml.safe_load(
        (CATALOG_SERVICES / f"{manifest}.yaml").read_text(encoding="utf-8")
    )
    nofile = (raw.get("ulimits") or {}).get("nofile")
    assert nofile is not None, f"{manifest} manifest keeps the default of 1024"
    assert nofile["soft"] >= MIN_NOFILE
    assert nofile["hard"] >= nofile["soft"]
