# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Production-contract routing regression tests for OMN-15978.

``operation_match`` entries without an explicit topic are assigned every subscribe
topic by the runtime wiring algorithm. The gateway attach node has three differently
typed operations, so that fallback fans attach, heartbeat, and detach messages into
all three dispatchers. Each entry must instead own exactly one command topic.
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _topics_for_handler_entry,
)

pytestmark = pytest.mark.unit

_CONTRACT_NAME = "node_gateway_attach_effect"
_EXPECTED_OPERATION_TOPICS = {
    "gateway.attach": "onex.cmd.omnibase-infra.gateway-attach-request.v1",
    "gateway.heartbeat": "onex.cmd.omnibase-infra.gateway-heartbeat-request.v1",
    "gateway.detach": "onex.cmd.omnibase-infra.gateway-detach-request.v1",
}


def test_each_gateway_operation_owns_exactly_one_subscribe_topic() -> None:
    """The real discovered contract maps each command topic to one handler only."""
    manifest = discover_contracts()
    matches = [
        contract for contract in manifest.contracts if contract.name == _CONTRACT_NAME
    ]
    assert len(matches) == 1, (
        f"expected one discovered {_CONTRACT_NAME} contract, found {len(matches)}"
    )

    contract = matches[0]
    assert contract.event_bus is not None
    assert contract.handler_routing is not None

    entries = {entry.operation: entry for entry in contract.handler_routing.handlers}
    assert set(entries) == set(_EXPECTED_OPERATION_TOPICS)

    topic_owners: dict[str, list[str]] = {}
    for operation, expected_topic in _EXPECTED_OPERATION_TOPICS.items():
        entry = entries[operation]
        assert entry.topic == expected_topic
        assigned_topics = _topics_for_handler_entry(contract, entry)
        assert assigned_topics == (expected_topic,)
        topic_owners.setdefault(expected_topic, []).append(operation)

    assert set(topic_owners) == set(contract.event_bus.subscribe_topics)
    assert all(len(owners) == 1 for owners in topic_owners.values())
