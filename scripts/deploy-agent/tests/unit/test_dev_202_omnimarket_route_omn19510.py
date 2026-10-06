# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19510 (task B8): the committed table routes omnimarket's dev rebuilds to dev-202.

This is the one row the second deploy slot exists for. Everything else keeps
its .201 route: a merge from any other repository, and a requester that is not
a CI run (a hand-dispatched rebuild), resolve to the pinned default. omnibase_infra
is the one exception, and only while dev-201 is frozen (OMN-20006): its row
carries ``while_frozen: dev-201``, and the ruling of omni_home ledger RULING
2026-09-25T00:56:45Z otherwise holds.
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


def test_dev_202_route_omnibase_infra_follows_dev_201_freeze_omn20006() -> None:
    """While dev-201 is frozen, omnibase_infra's dev rebuilds run on dev-202."""
    table = load_routing_table()
    assert (
        table.route(EnumRuntimeLane.DEV, "gha/omnibase_infra/runtime-rebuild-trigger")
        == "dev-202"
    )
    stand_in = [r for r in table.routes if r.requester_repository == "omnibase_infra"]
    assert [r.while_frozen for r in stand_in] == ["dev-201"]


@pytest.mark.parametrize(
    "requested_by",
    [
        "gha/omnibase_core/runtime-rebuild-trigger",
        "operator/jonah",
    ],
)
def test_dev_202_route_everything_else_stays_on_201(requested_by: str) -> None:
    table = load_routing_table()
    assert table.route(EnumRuntimeLane.DEV, requested_by) == PINNED_DEFAULT_INSTANCE


def test_dev_202_route_is_the_only_route() -> None:
    """Two rows, so no other repository moves off .201 by accident."""
    routes = load_routing_table().routes
    assert [(r.runtime_lane, r.requester_repository, r.instance) for r in routes] == [
        (EnumRuntimeLane.DEV, "omnimarket", "dev-202"),
        (EnumRuntimeLane.DEV, "omnibase_infra", "dev-202"),
    ]
