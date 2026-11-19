# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The shipped stability compose file declares what the writer roll rebuilds.

OMN-24903: reads the stability compose overlay from disk, with no
stand-in, and checks the two things the deploy relies on: every writer the roll
names is a service of the file, and the runtime main reads the dev lane's
overlay roots through the same two mounts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_setup_local_provision_effect.runtime_cost_estimator_build import (
    STABILITY_LANE_WRITER_SERVICES,
)

(COMPOSE,) = sorted(
    path
    for path in (Path(__file__).resolve().parents[3] / "docker").glob(
        "docker-compose.stability*.yml"
    )
    if "host-broker" not in path.name
)


@pytest.fixture(scope="module")
def services() -> dict[str, dict[str, object]]:
    text = re.sub(r"!(?:override|reset)\b", "", COMPOSE.read_text(encoding="utf-8"))
    declared = yaml.safe_load(text)["services"]
    assert isinstance(declared, dict)
    return declared


@pytest.mark.integration
def test_every_rolled_writer_is_a_service_of_the_stability_overlay(
    services: dict[str, dict[str, object]],
) -> None:
    missing = [name for name in STABILITY_LANE_WRITER_SERVICES if name not in services]
    assert missing == []


@pytest.mark.integration
def test_runtime_main_reads_the_dev_overlay_roots(
    services: dict[str, dict[str, object]],
) -> None:
    runtime = services["omninode-runtime"]
    environment = runtime["environment"]
    assert isinstance(environment, dict)
    assert (
        environment["ONEX_SKILL_OVERLAY_ROOTS"]
        == "/etc/onex/skill-overlays-store:/etc/onex/skill-overlays"
    )
    raw_volumes = runtime["volumes"]
    assert isinstance(raw_volumes, list)
    volumes = [str(volume) for volume in raw_volumes]
    assert any(
        volume.endswith(":/etc/onex/skill-overlays:ro")
        and "DEV_LANE_SKILL_OVERLAY_ROOTS_DIR" in volume
        for volume in volumes
    )
    assert any(
        volume.endswith(":/etc/onex/skill-overlays-store:ro")
        and "DEV_LANE_OVERLAY_STORE_ROOTS_DIR" in volume
        for volume in volumes
    )
