# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24562: the stability build function names the real overlay's composed services.

Runs ``build_dev_lane_services`` against the committed stability overlay file
(no mocked service list) and reads back which services the overlay builds on
omnimarket's composed image.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from omnibase_infra.nodes.node_setup_local_provision_effect.runtime_cost_estimator_build import (
    STABILITY_LANE_WRITER_SERVICES,
    STABILITY_TEST_COMPOSED_RUNTIME_SERVICES,
    build_dev_lane_services,
)

pytestmark = pytest.mark.integration

_DOCKER = Path(__file__).resolve().parents[3] / "docker"
_COMPOSE_TAG = re.compile(r"(?m)(:\s+|-\s+)!(?:override|reset)\b[ \t]*")


def _overlay_composed_services() -> set[str]:
    overlay = sorted(_DOCKER.glob("docker-compose.stability-t*.yml"))
    assert len(overlay) == 1, overlay
    text = overlay[0].read_text(encoding="utf-8")
    services = yaml.safe_load(_COMPOSE_TAG.sub(r"\1", text))["services"]
    return {
        name
        for name, service in services.items()
        if isinstance(service.get("build"), Mapping)
        and str(service["build"].get("context", "")).endswith(
            "sibling-repos/omnimarket"
        )
    }


def test_stability_build_names_exactly_the_overlay_composed_services() -> None:
    stability: Any = SimpleNamespace(value="stability")
    dev: Any = object()
    execution: Any = SimpleNamespace(
        EnumRuntimeLane=SimpleNamespace(STABILITY_TEST=stability, DEV=dev),
        Scope=SimpleNamespace(RUNTIME="runtime"),
        _compose_env=dict,
        lane_config_for=lambda lane: SimpleNamespace(
            compose_files=(f"/clone/docker/docker-compose.{stability.value}.yml",),
            disabled_services=frozenset(),
        ),
    )
    calls: list[dict[str, Any]] = []

    def _build(*args: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    # The writers' image-label read needs a real docker; this test reads argv.
    with patch(
        f"{build_dev_lane_services.__module__}._assert_images_at_deploy_sha",
        return_value=None,
    ):
        result = build_dev_lane_services(
            execution,
            _build,
            "0" * 40,
            lambda *a, **k: None,
            build_source="release",
            lane=stability,
            git_ref="origin/dev",
        )
    assert result == {}
    assert len(calls) == 2
    assert tuple(calls[0]["services"]) == STABILITY_TEST_COMPOSED_RUNTIME_SERVICES
    assert tuple(calls[1]["services"]) == STABILITY_LANE_WRITER_SERVICES
    built = {name for call in calls for name in call["services"]}
    assert built == _overlay_composed_services()
