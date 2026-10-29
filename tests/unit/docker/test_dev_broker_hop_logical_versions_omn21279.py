# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Each declared dev broker hop starts on the logical version the previous hop left (OMN-21279).

``docker/redpanda/dev_broker_upgrade_path.yaml`` records, per image, the
logical version it finalizes at (``latest``) and the oldest one it starts on
(``earliest``), read from ``/v1/features`` on scratch brokers. A Redpanda image
refuses a volume whose cluster_version is below its ``earliest``: v25.3.18 on a
v24.3 volume asserted "incompatible logical version 14 to logical version 17".
These tests walk the live readback, the catalog pin and the declared path, refusing a hop whose
image cannot start on what the previous image left, and an image on the chain
with no measured entry.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import pairwise
from typing import Any

import pytest
import yaml

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]

UPGRADE_PATH = DOCKER_DIR / "redpanda" / "dev_broker_upgrade_path.yaml"
CATALOG_BROKER = DOCKER_DIR / "catalog" / "services" / "redpanda.yaml"


def _declaration() -> dict[str, Any]:
    loaded = yaml.safe_load(UPGRADE_PATH.read_text())
    assert isinstance(loaded, dict), f"{UPGRADE_PATH} did not parse to a mapping"
    return loaded


def hop_refusals(chain: list[str], versions: dict[str, dict[str, int]]) -> list[str]:
    """Why each hop of ``chain`` cannot start on its predecessor's volume; empty if all can."""
    refusals = [
        f"{image}: no measured logical version"
        for image in chain
        if image not in versions
    ]
    if refusals:
        return refusals
    for previous, image in pairwise(chain):
        left = versions[previous]["latest"]
        needs = versions[image]["earliest"]
        if versions[image]["latest"] <= left:
            refusals.append(f"{image}: does not advance past {previous}")
        if needs > left:
            refusals.append(
                f"{image}: needs logical version {needs}, {previous} leaves {left}"
            )
    return refusals


def _assert_plan(declared: dict[str, Any], pin: str) -> None:
    deployed = str(declared["deployed_readback"]["image"])
    path = [str(image) for image in declared["path"]]
    assert path, "remaining upgrade path must be declared"
    releases = declared["feature_releases"]
    deployed_feature = deployed.rsplit(":v", 1)[1].rsplit(".", 1)[0]
    pin_feature = pin.rsplit(":v", 1)[1].rsplit(".", 1)[0]
    # A target already read back live is a redeploy, not another upgrade hop.
    if pin != deployed:
        assert pin == path[0], "pin must be the first remaining path hop"
        assert releases.index(pin_feature) - releases.index(deployed_feature) == 1, (
            "a new pin must advance exactly one feature release from the readback"
        )
    # The pin is either deployed or path[0]; never insert a duplicate image.
    chain = [deployed, *path]
    assert hop_refusals(chain, declared["logical_versions"]) == []


def test_declared_chain_starts_every_hop() -> None:
    pin = str(yaml.safe_load(CATALOG_BROKER.read_text())["image"])
    _assert_plan(_declaration(), pin)


def test_an_already_deployed_pin_is_a_redeploy_not_an_upgrade() -> None:
    declared = _declaration()
    pin = declared["deployed_readback"]["image"]
    _assert_plan(declared, pin)


def test_the_next_distinct_pin_is_an_upgrade() -> None:
    declared = _declaration()
    _assert_plan(declared, declared["path"][0])


def test_repeating_the_readback_in_the_path_is_refused() -> None:
    declared = deepcopy(_declaration())
    pin = declared["deployed_readback"]["image"]
    declared["path"].insert(0, pin)
    with pytest.raises(AssertionError, match="does not advance"):
        _assert_plan(declared, pin)


def test_a_pin_off_the_remaining_path_is_refused() -> None:
    declared = _declaration()
    with pytest.raises(AssertionError, match="first remaining path hop"):
        _assert_plan(declared, "redpandadata/redpanda:v25.1.12")


@pytest.mark.parametrize(
    ("chain", "refused"),
    [
        (["redpandadata/redpanda:v24.3.18", "redpandadata/redpanda:v25.3.18"], True),
        (["redpandadata/redpanda:v24.3.18", "redpandadata/redpanda:v25.2.14"], True),
        (["redpandadata/redpanda:v24.3.18", "redpandadata/redpanda:v25.1.12"], False),
        (["redpandadata/redpanda:v25.1.12", "redpandadata/redpanda:v25.2.14"], False),
        (["redpandadata/redpanda:v25.1.12", "redpandadata/redpanda:v25.3.18"], True),
        (["redpandadata/redpanda:v25.2.14", "redpandadata/redpanda:v25.3.18"], False),
        (["redpandadata/redpanda:v24.3.18", "redpandadata/redpanda:v25.1.99"], True),
        (["redpandadata/redpanda:v25.2.14", "redpandadata/redpanda:v25.2.14"], True),
        (["redpandadata/redpanda:v25.2.14", "redpandadata/redpanda:v25.1.12"], True),
    ],
)
def test_hop_refusal_fixtures(chain: list[str], refused: bool) -> None:
    versions = _declaration()["logical_versions"]
    assert bool(hop_refusals(chain, versions)) is refused
