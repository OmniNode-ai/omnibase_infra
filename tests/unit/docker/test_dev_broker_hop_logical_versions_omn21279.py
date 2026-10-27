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
        if needs > left:
            refusals.append(
                f"{image}: needs logical version {needs}, {previous} leaves {left}"
            )
    return refusals


def test_declared_chain_starts_every_hop() -> None:
    declared = _declaration()
    pin = str(yaml.safe_load(CATALOG_BROKER.read_text())["image"])
    chain = [
        str(declared["deployed_readback"]["image"]),
        pin,
        *(str(image) for image in declared["path"]),
    ]
    assert hop_refusals(chain, declared["logical_versions"]) == []


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
    ],
)
def test_hop_refusal_fixtures(chain: list[str], refused: bool) -> None:
    versions = _declaration()["logical_versions"]
    assert bool(hop_refusals(chain, versions)) is refused
