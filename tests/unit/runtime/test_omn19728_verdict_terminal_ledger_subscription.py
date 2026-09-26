# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The immutable DoD terminal must reach event_ledger for bounded replay."""

from __future__ import annotations

from pathlib import Path

import yaml

VERDICT_TOPIC = "onex.evt.omnimarket.dod-verify-completed.v1"
CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_ledger_projection_compute/contract.yaml"
)


def test_verdict_terminal_is_subscribed_and_dispatched_once() -> None:
    """A bare subscription is insufficient: the runtime needs a typed route."""
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    topics = contract["event_bus"]["subscribe_topics"]
    routes = [
        route
        for route in contract["handler_routing"]["handlers"]
        if route["topic"] == VERDICT_TOPIC
    ]

    assert topics.count(VERDICT_TOPIC) == 1
    assert len(routes) == 1
    route = routes[0]
    assert route["message_category"] == "event"
    assert route["event_model"] == {
        "name": "ModelEventMessage",
        "module": "omnibase_infra.event_bus.models.model_event_message",
    }
    assert route["handler"]["name"] == "HandlerLedgerProjection"
    assert route["operation"] == "ledger.project"
