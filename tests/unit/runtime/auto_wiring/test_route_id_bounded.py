# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-wired route and dispatcher ids stay within the ModelDispatchRoute limits (OMN-20767).

omnimarket 0.4.308 ships two contracts whose natural route id
(``route.auto.<contract>.<handler entry key>.<sanitized topic>``) is longer
than the 200-character ``max_length`` on ``ModelDispatchRoute.route_id``, so
the route constructor raised and the runtime failed at auto-wiring. These
tests pin the bounded form: unchanged within the limit, otherwise a
deterministic prefix plus a digest of the full natural id.
"""

from __future__ import annotations

import pytest

from omnibase_core.enums import EnumMessageCategory
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _derive_dispatcher_id,
    _derive_handler_entry_key,
    _derive_route_id,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelHandlerRef,
    ModelHandlerRoutingEntry,
)

_ROUTE_ID_MAX = 200

# The two real omnimarket 0.4.308 contracts (name, handler class, operation,
# subscribed command topic), read from their contract.yaml handler_routing.
_REAL_CONTRACTS = [
    (
        "node_delegation_acceptance_judged_replay_compute",
        "HandlerDelegationAcceptanceJudgedReplay",
        "delegation_acceptance_judged_replay",
        "onex.cmd.omnimarket.delegation-acceptance-replay-requested.v1",
    ),
    (
        "node_delegation_acceptance_judged_publish_effect",
        "HandlerDelegationAcceptanceJudgedPublish",
        "delegation_acceptance_judged_publish",
        "onex.cmd.omnimarket.delegation-acceptance-judged-publish.v1",
    ),
]


def _key(handler: str, operation: str) -> str:
    return _derive_handler_entry_key(
        ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name=handler, module="omnimarket.fake.handlers"),
            operation=operation,
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("contract_name", "handler", "operation", "topic"), _REAL_CONTRACTS
)
def test_real_omnimarket_contracts_derive_a_valid_route(
    contract_name: str, handler: str, operation: str, topic: str
) -> None:
    handler_key = _key(handler, operation)
    route_id = _derive_route_id(contract_name, handler_key, topic)
    dispatcher_id = _derive_dispatcher_id(contract_name, handler_key)

    assert len(route_id) <= _ROUTE_ID_MAX
    assert route_id.startswith(f"route.auto.{contract_name}.")
    route = ModelDispatchRoute(
        route_id=route_id,
        topic_pattern=topic,
        message_category=EnumMessageCategory.COMMAND,
        handler_id=dispatcher_id,
    )
    assert route.route_id == route_id
    # Deterministic: the same inputs derive the same id on every boot.
    assert _derive_route_id(contract_name, handler_key, topic) == route_id


@pytest.mark.unit
def test_long_ids_sharing_a_prefix_stay_distinct() -> None:
    contract_name = "node_" + "x" * 230
    handler_key = "HandlerShared"
    topic_a = "onex.cmd.omnimarket.alpha.v1"
    topic_b = "onex.cmd.omnimarket.bravo.v1"

    route_a = _derive_route_id(contract_name, handler_key, topic_a)
    route_b = _derive_route_id(contract_name, handler_key, topic_b)

    assert len(route_a) <= _ROUTE_ID_MAX
    assert len(route_b) <= _ROUTE_ID_MAX
    assert route_a != route_b
    assert route_a == _derive_route_id(contract_name, handler_key, topic_a)

    dispatcher_a = _derive_dispatcher_id(contract_name, "HandlerA")
    dispatcher_b = _derive_dispatcher_id(contract_name, "HandlerB")
    assert len(dispatcher_a) <= _ROUTE_ID_MAX
    assert dispatcher_a != dispatcher_b


@pytest.mark.unit
def test_ids_within_the_limit_are_unchanged() -> None:
    assert (
        _derive_route_id("my_node", "my_handler", "onex.evt.platform.my-topic.v1")
        == "route.auto.my_node.my_handler.onex_evt_platform_my_topic_v1"
    )
    assert (
        _derive_dispatcher_id("my_node", "my_handler")
        == "dispatcher.auto.my_node.my_handler"
    )
    # Boundary: a natural id of exactly the limit is kept verbatim.
    contract_name = "n" * (_ROUTE_ID_MAX - len("route.auto...t"))
    exact = _derive_route_id(contract_name, "", "t")
    assert len(exact) == _ROUTE_ID_MAX
    assert exact == f"route.auto.{contract_name}..t"
