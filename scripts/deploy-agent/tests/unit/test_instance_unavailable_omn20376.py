# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unavailable instances route rebuilds to the default instance (OMN-21402)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested
from deploy_agent.routing import (
    PINNED_DEFAULT_INSTANCE,
    DeployRouter,
    RoutingTableError,
    load_routing_table,
    parse_routing_table,
)

pytestmark = pytest.mark.unit

TABLE_WITH_OMNIMARKET_ROUTE = f"""
default_instance: {PINNED_DEFAULT_INSTANCE}
instances:
  {PINNED_DEFAULT_INSTANCE}:
    hostnames: [host-a]
    consumer_group: group-a
  lane-b:
    hostnames: [host-b]
    consumer_group: group-b
    available: false
routes:
  - runtime_lane: dev
    requester_repository: omnimarket
    instance: lane-b
"""


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("    available: false\n", PINNED_DEFAULT_INSTANCE),
        ("    available: true\n", "lane-b"),
        ("", "lane-b"),
    ],
)
def test_omnimarket_route_follows_availability(declaration: str, expected: str) -> None:
    table = parse_routing_table(
        TABLE_WITH_OMNIMARKET_ROUTE.replace("    available: false\n", declaration)
    )
    assert table.route(EnumRuntimeLane.DEV, "gha/omnimarket/pr-1") == expected


def test_only_default_agent_accepts_when_dev_202_is_unavailable() -> None:
    table = parse_routing_table(TABLE_WITH_OMNIMARKET_ROUTE)
    command = ModelRebuildRequested.model_validate(
        {
            "correlation_id": str(uuid4()),
            "requested_by": "gha/omnimarket/pr-1",
            "scope": "full",
            "runtime_lane": "dev",
            "build_source": "workspace",
            "git_ref": "a" * 40,
        }
    )
    accepted = {
        name: DeployRouter(table, instance, lambda sha: table).is_mine(command)[0]
        for name, instance in table.instances.items()
    }
    assert accepted[PINNED_DEFAULT_INSTANCE] is True
    assert accepted["lane-b"] is False
    assert sum(accepted.values()) == 1


@pytest.mark.parametrize("value", ['"no"', "0", "1", "null"])
def test_non_bool_availability_refuses_table(value: str) -> None:
    with pytest.raises(RoutingTableError, match=r"lane-b.*available"):
        parse_routing_table(
            TABLE_WITH_OMNIMARKET_ROUTE.replace(
                "available: false", f"available: {value}"
            )
        )


def test_unavailable_default_refuses_table() -> None:
    text = TABLE_WITH_OMNIMARKET_ROUTE.replace(
        "consumer_group: group-a\n",
        "consumer_group: group-a\n    available: false\n",
    )
    with pytest.raises(
        RoutingTableError,
        match=rf"default_instance.*{PINNED_DEFAULT_INSTANCE}.*unavailable",
    ):
        parse_routing_table(text)


def test_committed_route_keeps_dev_202_row_and_resolves_to_available_instance() -> None:
    table = load_routing_table()
    routed = next(
        r.instance for r in table.routes if r.requester_repository == "omnimarket"
    )
    assert any(
        row.runtime_lane is EnumRuntimeLane.DEV
        and row.requester_repository == "omnimarket"
        and row.instance == routed
        for row in table.routes
    )
    expected = routed if table.instances[routed].available else table.default_instance
    assert (
        table.route(EnumRuntimeLane.DEV, "gha/omnimarket/runtime-rebuild-trigger")
        == expected
    )
    for row in table.routes:
        assert isinstance(table.instances[row.instance].available, bool)
        resolved = table.route(row.runtime_lane, f"gha/{row.requester_repository}/pr-1")
        assert table.instances[resolved].available is True
