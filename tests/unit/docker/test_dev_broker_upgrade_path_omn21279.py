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

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

import pytest
import yaml

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]

UPGRADE_PATH = DOCKER_DIR / "redpanda" / "dev_broker_upgrade_path.yaml"
CATALOG_BROKER = DOCKER_DIR / "catalog" / "services" / "redpanda.yaml"
READBACK = (
    DOCKER_DIR.parent / "docs/incidents/OMN-21279/dev-broker-deployed-readback.json"
)
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
    """Validate the plan against retained command output, without contacting a lane."""
    assert (
        declared["readback_sha256"] == hashlib.sha256(READBACK.read_bytes()).hexdigest()
    )
    capture = json.loads(READBACK.read_text())
    for command in ("inspect", "features", "inspect_after"):
        assert capture[command]["exit_code"] == 0
    inspected = json.loads(capture["inspect"]["stdout"])
    assert inspected == json.loads(capture["inspect_after"]["stdout"])
    assert capture["features"]["argv"] == [
        "docker",
        "exec",
        inspected["id"],
        "curl",
        "-fsS",
        "http://127.0.0.1:9644/v1/features",
    ]
    assert inspected["project"] == "omnibase-infra"
    assert inspected["service"] == "redpanda"
    assert inspected["data_volume"] == "omnibase-infra-redpanda-data"
    features = json.loads(capture["features"]["stdout"])
    observed = declared["deployed_readback"]
    for field in ("image", "started_at", "health", "restart_count"):
        assert observed[field] == inspected[field], f"{field}: differs from capture"
    assert capture["host"]["argv"] == ["hostname"]
    assert capture["host"]["exit_code"] == 0
    assert observed["host"] == "sha256:" + capture["host"]["stdout_sha256"], (
        "host: differs from capture"
    )
    assert observed["container"] == inspected["name"].removeprefix("/"), (
        "container: differs from capture"
    )
    for field in ("cluster_version", "node_latest_version", "node_earliest_version"):
        assert observed[field] == features[field], f"{field}: differs from capture"
    began = datetime.fromisoformat(capture["capture_started_at"])
    finished = datetime.fromisoformat(capture["capture_finished_at"])
    observed_at = datetime.fromisoformat(observed["observed_at"])
    planned_at = datetime.fromisoformat(declared["planned_at"])
    assert began <= observed_at <= finished, "observed_at: outside capture window"
    assert finished - began <= timedelta(minutes=1), "capture took too long"
    assert finished <= planned_at <= finished + timedelta(hours=1), "stale capture"
    assert datetime.fromisoformat(observed["started_at"]) < began
    image = observed["image"]
    assert _feature(image) == declared["deployed_feature_release"]
    assert observed["cluster_version"] == declared["logical_versions"][image]["latest"]
    assert (
        observed["node_earliest_version"]
        == declared["logical_versions"][image]["earliest"]
    )
    assert observed["node_latest_version"] == observed["cluster_version"]
    assert observed["health"] == "healthy"
    assert observed["restart_count"] == 0


def test_deployed_release_has_a_finalized_live_readback() -> None:
    _assert_deployed_readback(_declaration())


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"image": "redpandadata/redpanda:v24.3.18"}, "image"),
        ({"cluster_version": 15}, "cluster_version"),
        ({"node_latest_version": 17}, "node_latest_version"),
        ({"node_earliest_version": 12}, "node_earliest_version"),
        ({"health": "starting"}, "health"),
        ({"restart_count": 1}, "restart_count"),
        ({"host": "h200"}, "host"),
        ({"container": "scratch-redpanda"}, "container"),
        ({"observed_at": "2026-10-27T16:31:45Z"}, "observed_at"),
        ({"started_at": "2026-10-27T11:44:44Z"}, "started_at"),
    ],
)
def test_incomplete_or_wrong_broker_readback_is_refused(
    change: dict[str, Any], field: str
) -> None:
    declared = deepcopy(_declaration())
    declared["deployed_readback"].update(change)
    with pytest.raises(AssertionError, match=field):
        _assert_deployed_readback(declared)


def test_readback_and_version_table_cannot_validate_each_others_edits() -> None:
    declared = deepcopy(_declaration())
    image = declared["deployed_readback"]["image"]
    declared["logical_versions"][image]["latest"] = 17
    declared["deployed_readback"].update(cluster_version=17, node_latest_version=17)
    with pytest.raises(AssertionError, match="cluster_version: differs from capture"):
        _assert_deployed_readback(declared)


def test_a_new_plan_cannot_reuse_a_stale_readback() -> None:
    declared = deepcopy(_declaration())
    declared["planned_at"] = (
        datetime.fromisoformat(declared["planned_at"]) + timedelta(days=1)
    ).isoformat()
    with pytest.raises(AssertionError, match="stale capture"):
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
