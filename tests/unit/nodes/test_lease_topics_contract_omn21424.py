# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The eight lease topics are declared in the lease node contracts (OMN-21424).

The plan (runtime cluster and generic lease, section 5.4) names three command
topics and five event topics, one concept per topic. The orchestrator
subscribes to the commands and publishes the events; the reducer folds the four
lifecycle events (a refusal carries no lease, so it is not folded).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_NODES = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"

COMMANDS = {
    "onex.cmd.omnibase-infra.lease-acquire-requested.v1",
    "onex.cmd.omnibase-infra.lease-renew-requested.v1",
    "onex.cmd.omnibase-infra.lease-release-requested.v1",
}
LIFECYCLE = {
    "onex.evt.omnibase-infra.lease-granted.v1",
    "onex.evt.omnibase-infra.lease-renewed.v1",
    "onex.evt.omnibase-infra.lease-released.v1",
    "onex.evt.omnibase-infra.lease-expired.v1",
}
EVENTS = LIFECYCLE | {"onex.evt.omnibase-infra.lease-refused.v1"}


def _contract(node: str) -> dict[str, object]:
    loaded = yaml.safe_load((_NODES / node / "contract.yaml").read_text())
    assert isinstance(loaded, dict)
    return loaded


def _topics(node: str, key: str) -> set[str]:
    event_bus = _contract(node).get("event_bus")
    assert isinstance(event_bus, dict), f"{node} declares no event_bus"
    topics = event_bus.get(key) or []
    assert isinstance(topics, list)
    return {str(topic) for topic in topics}


def test_orchestrator_subscribes_the_three_commands_and_publishes_the_five_events() -> (
    None
):
    assert _topics("node_lease_orchestrator", "subscribe_topics") == COMMANDS
    assert _topics("node_lease_orchestrator", "publish_topics") == EVENTS


def test_reducer_folds_the_four_lifecycle_events() -> None:
    assert _topics("node_lease_reducer", "subscribe_topics") == LIFECYCLE
    assert _topics("node_lease_reducer", "publish_topics") == set()


def test_eight_topics_in_all() -> None:
    declared = (
        _topics("node_lease_orchestrator", "subscribe_topics")
        | _topics("node_lease_orchestrator", "publish_topics")
        | _topics("node_lease_reducer", "subscribe_topics")
    )
    assert declared == COMMANDS | EVENTS
    assert len(declared) == 8


def test_each_command_topic_routes_to_exactly_one_handler() -> None:
    routing = _contract("node_lease_orchestrator")["handler_routing"]
    assert isinstance(routing, dict)
    handlers = routing["handlers"]
    assert isinstance(handlers, list)
    by_topic: dict[str, list[str]] = {}
    for entry in handlers:
        assert isinstance(entry, dict)
        by_topic.setdefault(str(entry.get("topic")), []).append(
            str(entry["handler"]["name"])
        )
    assert by_topic == {
        "onex.cmd.omnibase-infra.lease-acquire-requested.v1": ["HandlerLeaseAcquire"],
        "onex.cmd.omnibase-infra.lease-renew-requested.v1": ["HandlerLeaseRenew"],
        "onex.cmd.omnibase-infra.lease-release-requested.v1": ["HandlerLeaseRelease"],
    }


def test_orchestrator_declares_its_etcd_and_kind_configuration() -> None:
    dependencies = _contract("node_lease_orchestrator").get("dependencies")
    assert isinstance(dependencies, list)
    env_vars = {
        str(dep.get("env_var"))
        for dep in dependencies
        if isinstance(dep, dict) and dep.get("type") == "environment"
    }
    assert {"ONEX_LEASE_ETCD_ENDPOINT", "ONEX_LEASE_KINDS"} <= env_vars
