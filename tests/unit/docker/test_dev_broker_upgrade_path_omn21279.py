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
They also require the dev pin past the deployed volume to be the next proven
hop, and every path hop to have been proven on a copy of the dev data volume.
"""

from __future__ import annotations

import re
from copy import deepcopy
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


def _assert_deployed_readback(declared: dict[str, Any]) -> None:
    """Advancing the deployed release requires a finalized, healthy live readback."""
    observed = declared["deployed_readback"]
    image = observed["image"]
    assert _feature(image) == declared["deployed_feature_release"]
    assert observed["cluster_version"] == declared["logical_versions"][image]["latest"]
    assert observed["node_latest_version"] == observed["cluster_version"]
    assert observed["health"] == "healthy"
    assert observed["restart_count"] == 0
    assert observed["host"] == "h201"
    assert observed["container"] == "omnibase-infra-redpanda"


def test_deployed_release_has_a_finalized_live_readback() -> None:
    _assert_deployed_readback(_declaration())


@pytest.mark.parametrize(
    "change",
    [
        {"image": "redpandadata/redpanda:v24.3.18"},
        {"cluster_version": 14},
        {"node_latest_version": 16},
        {"health": "starting"},
        {"restart_count": 1},
        {"container": "scratch-redpanda"},
    ],
)
def test_incomplete_or_wrong_broker_readback_is_refused(change: dict[str, Any]) -> None:
    declared = deepcopy(_declaration())
    declared["deployed_readback"].update(change)
    with pytest.raises(AssertionError):
        _assert_deployed_readback(declared)


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


def unproven_refusal(
    image: str, deployed_feature_release: str, path: list[str], proven: list[str]
) -> str:
    """Why a pin past the deployed volume is not the next proven hop; empty if it is."""
    if _feature(image) == deployed_feature_release:
        return ""
    if not path or image != path[0]:
        return f"{image}: the pin must take the next hop on the declared path"
    if image not in proven:
        return f"{image}: has not been proven on a copy of the dev data volume"
    return ""


def test_dev_pin_is_at_most_one_feature_release_past_the_deployed_volume() -> None:
    declared = _declaration()
    image = str(yaml.safe_load(CATALOG_BROKER.read_text())["image"])
    assert not pin_refusal(
        image, declared["deployed_feature_release"], declared["feature_releases"]
    )


def test_dev_pin_past_the_deployed_volume_is_the_next_proven_hop() -> None:
    declared = _declaration()
    image = str(yaml.safe_load(CATALOG_BROKER.read_text())["image"])
    proven = [str(hop["image"]) for hop in declared["proven_on_data_copy"]]
    assert (
        unproven_refusal(
            image, declared["deployed_feature_release"], declared["path"], proven
        )
        == ""
    )


_HOP_1 = "redpandadata/redpanda:v25.1.12"
_HOP_2 = "redpandadata/redpanda:v25.2.14"
_HOP_3 = "redpandadata/redpanda:v25.3.18"
_OFF_HOP = "redpandadata/redpanda:v25.1.11"
_PATH = [_HOP_1, _HOP_2, _HOP_3]
_OFF_PATH = "the pin must take the next hop on the declared path"
_UNPROVEN = "has not been proven on a copy of the dev data volume"


@pytest.mark.parametrize(
    ("image", "deployed", "path", "proven", "reason"),
    [
        # A pin on the deployed feature release is not a hop.
        ("redpandadata/redpanda:v24.3.18", "24.3", _PATH, [], ""),
        ("redpandadata/redpanda:v24.3.5", "24.3", _PATH, [], ""),
        # The next hop, proven: accepted.
        (_HOP_1, "24.3", _PATH, [_HOP_1], ""),
        # The next hop, not proven: refused as unproven only.
        (_HOP_1, "24.3", _PATH, [], _UNPROVEN),
        # Off path but proven: refused as off path, not as unproven.
        (_OFF_HOP, "24.3", _PATH, [_OFF_HOP], _OFF_PATH),
        # Off path and unproven: the path check runs first.
        (_OFF_HOP, "24.3", _PATH, [], _OFF_PATH),
        # A later hop taken before the next one is off path.
        (_HOP_2, "24.3", _PATH, [_HOP_1, _HOP_2], _OFF_PATH),
        # After the first hop is live, deployed advances and the hop leaves path.
        (_HOP_1, "25.1", [_HOP_2, _HOP_3], [_HOP_1, _HOP_2], ""),
        (_HOP_2, "25.1", [_HOP_2, _HOP_3], [_HOP_1, _HOP_2], ""),
        (_HOP_3, "25.1", [_HOP_2, _HOP_3], _PATH, _OFF_PATH),
        # An exhausted path admits no further pin.
        (_HOP_3, "25.2", [], [_HOP_3], _OFF_PATH),
    ],
)
def test_unproven_refusal_fixtures(
    image: str, deployed: str, path: list[str], proven: list[str], reason: str
) -> None:
    refusal = unproven_refusal(image, deployed, path, proven)
    if reason:
        other = _UNPROVEN if reason == _OFF_PATH else _OFF_PATH
        assert reason in refusal, refusal
        assert other not in refusal, refusal
    else:
        assert refusal == ""


def test_every_path_hop_was_proven_on_a_data_copy() -> None:
    declared = _declaration()
    proven = [str(hop["image"]) for hop in declared["proven_on_data_copy"]]
    for image in declared["path"]:
        assert image in proven, f"{image} has not been proven on a data copy"


@pytest.mark.parametrize(
    ("image", "deployed", "refused"),
    [
        ("redpandadata/redpanda:v24.3.18", "24.3", False),
        ("redpandadata/redpanda:v25.1.12", "24.3", False),
        ("redpandadata/redpanda:v25.2.14", "24.3", True),
        ("redpandadata/redpanda:v25.3.18", "24.3", True),
        ("redpandadata/redpanda:v24.2.7", "24.3", True),
        ("redpandadata/redpanda:v25.2.14", "25.1", False),
        ("redpandadata/redpanda:v25.3.18", "25.1", True),
        ("redpandadata/redpanda:v24.3.18", "25.1", True),
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
