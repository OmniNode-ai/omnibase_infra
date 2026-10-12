# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lab compose files and host units that moved to the private config root stay out (OMN-20933).

Operator ruling 2026-10-10T21:57:38Z: no lab configuration of any kind lives in a
shipped repository; deployment facts come from overlays supplied by whoever runs
the system. Each path below moved byte-identical to the operator's private config
root (``$ONEX_WORKSPACE_CONFIG_ROOT``). A copy reappearing here is lab
configuration every consumer of this repository would inherit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

# path in this repository -> path under the private config root
MOVED = {
    "docker/docker-compose.infisical-stability.yml": (
        "deploy/lab/compose/docker-compose.infisical-stability.yml"
    ),
    "docker/docker-compose.dns-canary.yml": (
        "deploy/lab/compose/docker-compose.dns-canary.yml"
    ),
    "deploy/lane-census/onex-disk-gc.service.d/20-lane-census.conf": (
        "systemd/lane-census/onex-disk-gc.service.d/20-lane-census.conf"
    ),
    "deploy/lane-census/standalone/onex-lane-census.service": (
        "systemd/lane-census/standalone/onex-lane-census.service"
    ),
    "deploy/lane-census/standalone/onex-lane-census.timer": (
        "systemd/lane-census/standalone/onex-lane-census.timer"
    ),
}


def test_the_repository_root_is_resolved() -> None:
    """Positive control: an absence below means absent, not a wrong root."""
    assert (REPO_ROOT / "docker" / "docker-compose.infra.yml").is_file()
    assert (REPO_ROOT / "deploy" / "lane-census" / "install-lane-census.sh").is_file()


@pytest.mark.parametrize("path", sorted(MOVED))
def test_moved_lab_config_is_not_shipped(path: str) -> None:
    assert not (REPO_ROOT / path).exists(), (
        f"{path} is lab configuration; it lives at {MOVED[path]} under the private "
        "config root (ONEX_WORKSPACE_CONFIG_ROOT), never in this repository"
    )
