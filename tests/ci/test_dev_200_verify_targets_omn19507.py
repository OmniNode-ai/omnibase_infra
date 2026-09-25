# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19507: dev-200's two declarations in the routing table agree.

``config/deploy_lane_routing.yaml`` describes an instance twice. The ``lane:``
block (OMN-19543) is what the deploy agent builds the instance's composition
from; the ``verify:`` block (OMN-19507 AC2) is what the per-merge verify job
reads. Both name the receipt lane, the compose project and the containers, so
they can drift apart silently: a verify job would then poll one lane while the
agent deployed another, or emit a receipt under a lane the agent does not
prove. These tests pin them together, and pin dev-200's verify targets to its
compose overlay, as ``test_dev_202_targets_agree_with_its_overlay`` does for
dev-202.

They also pin that no route names dev-200 yet. No runner carries ``host-200``
(org runners API 2026-09-25T14:58Z), so a verify job for a merge routed there
would queue with nowhere to run. When a verify runner is registered on .200,
this pin is the one test that changes with the route.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.deploy_lane_verify_route import load_table, targets_for_receipt_lane
from scripts.ci.lane_settle_budget import (
    assert_declaration_within_bounds,
    load_declaration,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_200_OVERLAY = REPO_ROOT / "docker" / "docker-compose.dev-200.yml"


class _OverrideLoader(yaml.SafeLoader):
    """Compose's ``!override`` and ``!reset`` tags resolve to their value."""


_OverrideLoader.add_constructor(
    "!override",
    lambda loader, node: (
        loader.construct_sequence(node)
        if isinstance(node, yaml.SequenceNode)
        else loader.construct_mapping(node)
        if isinstance(node, yaml.MappingNode)
        else loader.construct_scalar(node)
    ),
)
_OverrideLoader.add_constructor("!reset", lambda loader, node: None)


def _instances() -> dict[str, Any]:
    instances = load_table()["instances"]
    assert isinstance(instances, dict)
    return instances


def test_every_lane_block_names_the_receipt_lane_its_verify_block_emits() -> None:
    checked = 0
    for name, spec in _instances().items():
        lane = spec.get("lane")
        if lane is None:
            continue
        verify = spec["verify"]
        assert lane["receipt_lane"] == verify["receipt_lane"], name
        assert lane["compose_project"] == verify["compose_project"], name
        assert lane["runtime_container"] == verify["runtime_container"], name
        assert lane["postgres_container"] == verify["postgres_container"], name
        for key, url in (("main", "main_url"), ("effects", "effects_url")):
            assert verify[url].endswith(f":{lane['health_ports'][key]}"), (name, key)
        checked += 1
    # Positive control: dev-202 and dev-200 both carry a lane block.
    assert checked >= 2


def test_dev_200_verify_targets_agree_with_its_overlay() -> None:
    targets = targets_for_receipt_lane(load_table(), "compose-dev-200")
    overlay_text = DEV_200_OVERLAY.read_text(encoding="utf-8")
    overlay = yaml.load(overlay_text, Loader=_OverrideLoader)  # noqa: S506
    services = overlay["services"]
    assert overlay["name"] == targets.compose_project
    assert services["omninode-runtime"]["container_name"] == targets.runtime_container
    assert services["runtime-effects"]["container_name"] == targets.effects_container
    assert services["postgres"]["container_name"] == targets.postgres_container
    assert services["redpanda"]["container_name"] == targets.broker_container
    for url in (targets.main_url, targets.effects_url, targets.projection_url):
        port = re.search(r":(\d+)$", url)
        assert port is not None
        assert re.search(rf":{port.group(1)}:\d+", overlay_text), url
    assert "host-200" in targets.runner_labels


def test_no_route_names_dev_200_while_no_runner_carries_host_200() -> None:
    routed = {str(row.get("instance")) for row in load_table().get("routes") or ()}
    assert "dev-200" not in routed


def test_compose_dev_200_declares_a_settle_budget_inside_its_bounds() -> None:
    """The verify job derives its ceiling from the routed lane's declaration.

    Without one, a verify job for a merge routed to dev-200 would refuse at the
    budget read. The observed boots are the lab proof's (job fec4e429).
    """
    declaration = load_declaration("compose-dev-200")
    assert_declaration_within_bounds(declaration)
