# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verify jobs follow the agent's availability fallback (OMN-21402)."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci.deploy_lane_verify_route import (
    VerifyRouteError,
    load_table,
    requester_repository,
    resolve,
    route,
)

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("placeholder_routing_overlay")]

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "deploy-agent"))
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.routing import parse_routing_table


@pytest.mark.parametrize("available", [False, True])
def test_verify_targets_follow_availability(available: bool) -> None:
    table = copy.deepcopy(load_table())
    routed = next(
        row["instance"]
        for row in table["routes"]
        if row["requester_repository"] == "omnimarket"
    )
    table["instances"][routed]["available"] = available
    expected = routed if available else table["default_instance"]
    verify = table["instances"][expected]["verify"]
    resolved = resolve(table, runtime_lane="dev", requested_by="gha/omnimarket/pr-2870")
    assert resolved.instance == expected
    assert resolved.targets.receipt_lane == verify["receipt_lane"]
    assert list(resolved.targets.runner_labels) == verify["runner_labels"]


@pytest.mark.parametrize("value", ["no", 0, 1, None])
def test_non_bool_availability_refuses_route(value: object) -> None:
    table = copy.deepcopy(load_table())
    routed = next(
        row["instance"]
        for row in table["routes"]
        if row["requester_repository"] == "omnimarket"
    )
    table["instances"][routed]["available"] = value
    with pytest.raises(VerifyRouteError, match=rf"{re.escape(routed)}.*available"):
        resolve(table, runtime_lane="dev", requested_by="gha/omnimarket/pr-2870")


def test_unavailable_default_refuses_even_a_matching_route() -> None:
    table = copy.deepcopy(load_table())
    default = table["default_instance"]
    routed = next(
        row["instance"]
        for row in table["routes"]
        if row["requester_repository"] == "omnimarket"
    )
    table["instances"][default]["available"] = False
    table["instances"][routed]["available"] = True
    with pytest.raises(
        VerifyRouteError, match=rf"default_instance.*{re.escape(default)}.*unavailable"
    ):
        resolve(table, runtime_lane="dev", requested_by="gha/omnimarket/pr-2870")


@pytest.mark.parametrize("variant", ["declared", "unavailable", "available"])
@pytest.mark.parametrize(
    "requested_by", ["gha/omnimarket/pr-1", "gha/omnibase_infra/pr-1", "operator/x"]
)
def test_verify_route_agrees_with_agent(variant: str, requested_by: str) -> None:
    table = copy.deepcopy(load_table())
    if variant != "declared":
        routed = next(
            row["instance"]
            for row in table["routes"]
            if row["requester_repository"] == "omnimarket"
        )
        table["instances"][routed]["available"] = variant == "available"
    agent_table = parse_routing_table(yaml.safe_dump(table))
    assert route(table, "dev", requester_repository(requested_by)) == agent_table.route(
        EnumRuntimeLane.DEV, requested_by
    )
