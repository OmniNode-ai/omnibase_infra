# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev broker's Redpanda pin moves one feature release at a time (OMN-21279).

The dev broker crashed inside Kafka ListGroups on v24.3.18 (SIGSEGV at
``group_manager::list_groups`` +0x1cc on 2026-10-25T20:56:59Z and
2026-10-26T12:19:27Z; abort in ``group_state_to_kafka_name`` on 2026-10-23).
Upstream fixed the group erase race in v25.3.16, but v25.3.x refuses to start on
a data volume older than v25.2 (``earliest_version = v25_2_1``). Pinning the dev
broker straight to v25.3 would leave it unable to start on its v24.3 volume.

``docker/redpanda/dev_broker_upgrade_path.yaml`` declares the feature release
the dev volume has run and the hop path to a fixed release. These tests refuse a
dev pin more than one feature release past that, and a path that skips a
feature release or ends before the fix.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import yaml

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]

UPGRADE_PATH = DOCKER_DIR / "redpanda" / "dev_broker_upgrade_path.yaml"
CATALOG_BROKER = DOCKER_DIR / "catalog" / "services" / "redpanda.yaml"
_VERSION = re.compile(r":v(\d+)\.(\d+)\.(\d+)$")


def _declaration() -> dict[str, Any]:
    loaded = yaml.safe_load(UPGRADE_PATH.read_text())
    assert isinstance(loaded, dict), f"{UPGRADE_PATH} did not parse to a mapping"
    return loaded


def _version(image: str) -> tuple[int, int, int]:
    match = _VERSION.search(image)
    assert match, f"no vX.Y.Z tag in {image!r}"
    return int(match[1]), int(match[2]), int(match[3])


def _feature(image: str) -> str:
    major, minor, _ = _version(image)
    return f"{major}.{minor}"


def pin_refusal(
    image: str, deployed_feature_release: str, feature_releases: list[str]
) -> str:
    """Why ``image`` may not be the dev pin over a volume at ``deployed``; empty if it may."""
    feature = _feature(image)
    if feature not in feature_releases:
        return f"{image}: feature release {feature} is not declared"
    step = feature_releases.index(feature) - feature_releases.index(
        deployed_feature_release
    )
    if step < 0:
        return f"{image}: downgrade from {deployed_feature_release}"
    if step > 1:
        return (
            f"{image}: skips {step - 1} feature release(s) past "
            f"{deployed_feature_release}"
        )
    return ""


def test_dev_pin_is_at_most_one_feature_release_past_the_deployed_volume() -> None:
    declared = _declaration()
    image = str(yaml.safe_load(CATALOG_BROKER.read_text())["image"])
    assert not pin_refusal(
        image, declared["deployed_feature_release"], declared["feature_releases"]
    )


@pytest.mark.parametrize(
    ("image", "deployed", "refused"),
    [
        ("redpandadata/redpanda:v24.3.18", "24.3", False),
        ("redpandadata/redpanda:v25.1.12", "24.3", False),
        ("redpandadata/redpanda:v25.2.14", "24.3", True),
        ("redpandadata/redpanda:v25.3.18", "24.3", True),
        ("redpandadata/redpanda:v24.2.7", "24.3", True),
        ("redpandadata/redpanda:v25.3.18", "25.2", False),
    ],
)
def test_pin_refusal_fixtures(image: str, deployed: str, refused: bool) -> None:
    releases = _declaration()["feature_releases"]
    assert bool(pin_refusal(image, deployed, releases)) is refused


def test_upgrade_path_steps_one_feature_release_and_ends_fixed() -> None:
    declared = _declaration()
    releases: list[str] = declared["feature_releases"]
    deployed = str(declared["deployed_feature_release"])
    path = [str(image) for image in declared["path"]]
    assert path, "the upgrade path is empty"
    for image in path:
        assert not pin_refusal(image, deployed, releases), image
        assert _feature(image) != deployed, f"{image} does not advance past {deployed}"
        deployed = _feature(image)
    fixed = tuple(int(part) for part in str(declared["first_fixed_release"]).split("."))
    assert _version(path[-1]) >= fixed, f"{path[-1]} predates {fixed}"
