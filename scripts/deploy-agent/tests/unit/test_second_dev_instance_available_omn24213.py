# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24213: the committed table declares the second dev instance available and routes omnimarket's dev rebuilds to it.

The instance was declared ``available: false`` while its host was out for repair
(OMN-21402). The host is back, its stack sat stopped after five failed boots, and a
rebuild is the only declared way to recreate it.
"""

from __future__ import annotations

import pytest
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.routing import PINNED_DEFAULT_INSTANCE, load_routing_table

pytestmark = pytest.mark.unit

OMNIMARKET_REBUILD = "gha/omnimarket/runtime-rebuild-trigger"


def _omnimarket_instance() -> str:
    table = load_routing_table()
    return next(
        r.instance for r in table.routes if r.requester_repository == "omnimarket"
    )


def test_omnimarket_instance_is_declared_available() -> None:
    table = load_routing_table()
    routed = _omnimarket_instance()
    assert routed != PINNED_DEFAULT_INSTANCE
    assert table.instances[routed].available is True, (
        f"{routed} is declared unavailable, so no rebuild is ever routed to it and "
        "its stopped stack is never recreated"
    )


def test_omnimarket_dev_rebuild_resolves_to_its_own_instance() -> None:
    table = load_routing_table()
    assert (
        table.route(EnumRuntimeLane.DEV, OMNIMARKET_REBUILD) == _omnimarket_instance()
    )


def test_default_instance_stays_available() -> None:
    table = load_routing_table()
    assert table.default_instance == PINNED_DEFAULT_INSTANCE
    assert table.instances[PINNED_DEFAULT_INSTANCE].available is True
