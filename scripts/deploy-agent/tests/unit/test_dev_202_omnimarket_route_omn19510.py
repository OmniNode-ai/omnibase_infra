# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19510 (task B8): the committed table routes omnimarket's dev rebuilds to dev-202.

This is the one row the second deploy slot exists for. Everything else keeps
its .201 route: an omnibase_infra merge, a merge from any other repository, and
a requester that is not a CI run (a hand-dispatched rebuild) all resolve to the
pinned default, so the .201 lane still proves omnibase_infra as ruled
(omni_home ledger RULING 2026-09-25T00:56:45Z).
"""

from __future__ import annotations

import pytest
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.routing import PINNED_DEFAULT_INSTANCE, load_routing_table

pytestmark = pytest.mark.unit


def test_dev_202_route_omnimarket_dev_rebuilds_go_to_dev_202() -> None:
    table = load_routing_table()
    assert (
        table.route(EnumRuntimeLane.DEV, "gha/omnimarket/runtime-rebuild-trigger")
        == "dev-202"
    )


@pytest.mark.parametrize(
    "requested_by",
    [
        "gha/omnibase_infra/runtime-rebuild-trigger",
        "gha/omnibase_core/runtime-rebuild-trigger",
        "operator/jonah",
    ],
)
def test_dev_202_route_everything_else_stays_on_201(requested_by: str) -> None:
    table = load_routing_table()
    assert table.route(EnumRuntimeLane.DEV, requested_by) == PINNED_DEFAULT_INSTANCE


def test_dev_202_route_is_the_only_route() -> None:
    """One row, so no other repository moves off .201 by accident."""
    routes = load_routing_table().routes
    assert [(r.runtime_lane, r.requester_repository, r.instance) for r in routes] == [
        (EnumRuntimeLane.DEV, "omnimarket", "dev-202")
    ]
