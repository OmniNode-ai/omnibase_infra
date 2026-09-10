# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18119 -- the runtime must supply a consumer for every DECLARED DLQ topic.

`node_dlq_replay_effect` declares three subscribe topics. The kernel built one
`ModelDlqReplayEngineConfig` pinned to the events topic and one `DLQConsumer`
from it, and the resolver keys the materialized dependency map by handler NAME,
so all three of the per-topic dispatcher entries OMN-18013 split the routing
into resolved to that single consumer. Two of the three declared topics were
drained by nothing at all, silently, for as long as the node has existed.

The subject set here is DISCOVERED from `contract.yaml`, not written down, so a
fourth declared topic that nobody wires is red here rather than silent on the
lane. That is the whole point: the previous state was not a bug anyone could
see, it was a disagreement between two files that no check compared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import omnibase_infra.nodes.node_dlq_replay_effect as _dlq_replay_pkg
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQ_REPLAY_CONSUMER_GROUP,
    DLQConsumer,
)
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

pytestmark = pytest.mark.integration

_HANDLER_NAME = "HandlerDlqReplay"
_BOOTSTRAP = "localhost:9092"


def _declared_subscribe_topics() -> set[str]:
    contract_path = Path(str(_dlq_replay_pkg.__file__)).parent / "contract.yaml"
    contract: dict[str, Any] = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    return set(contract["event_bus"]["subscribe_topics"])


def _wired_consumers() -> dict[str, DLQConsumer]:
    dependencies = _build_runtime_handler_dependencies(None, _BOOTSTRAP)
    assert dependencies is not None
    consumers = dependencies[_HANDLER_NAME]["consumers"]
    assert isinstance(consumers, dict), consumers
    return consumers


def test_every_declared_dlq_topic_gets_its_own_consumer() -> None:
    """RED on origin/dev: one consumer wired against three declared topics."""
    declared = _declared_subscribe_topics()
    assert len(declared) > 1, (
        "guard on the guard: this test is vacuous if the contract stops "
        "declaring more than one subscribe topic"
    )

    consumers = _wired_consumers()
    wired = {consumer.config.dlq_topic for consumer in consumers.values()}

    assert wired == declared, (
        f"contract.yaml declares {sorted(declared)} but the runtime supplies "
        f"consumers for {sorted(wired)}; the difference is drained by nothing"
    )
    assert set(consumers) == declared, (
        f"the mapping must be keyed by topic: {sorted(consumers)}"
    )


def test_every_consumer_joins_the_one_persistent_replay_group() -> None:
    """Kafka commits are per topic-partition, so one group covers all three
    declared topics. Minting a group per topic would fragment the committed
    position of a single logical drain for no benefit."""
    groups = {
        consumer.config.consumer_group for consumer in _wired_consumers().values()
    }
    assert groups == {DLQ_REPLAY_CONSUMER_GROUP}, groups


def test_the_topic_list_is_read_from_the_contract_and_fails_closed() -> None:
    """A silent fallback to a hardcoded topic would reinstate this defect in
    the one situation where nobody would look for it, so an unreadable or empty
    declaration raises."""
    from omnibase_infra.errors import ProtocolConfigurationError
    from omnibase_infra.runtime import service_kernel

    original = service_kernel.yaml.safe_load
    try:
        service_kernel.yaml.safe_load = lambda _text: {"event_bus": {}}  # type: ignore[assignment]
        with pytest.raises(ProtocolConfigurationError, match="subscribe_topics"):
            service_kernel._dlq_replay_subscribe_topics()

        service_kernel.yaml.safe_load = lambda _text: {  # type: ignore[assignment]
            "event_bus": {"subscribe_topics": []}
        }
        with pytest.raises(ProtocolConfigurationError, match="non-empty"):
            service_kernel._dlq_replay_subscribe_topics()
    finally:
        service_kernel.yaml.safe_load = original  # type: ignore[assignment]


def test_producers_are_shared_and_not_duplicated_per_topic() -> None:
    """The producers publish to the original topic and to the single quarantine
    sink, so they are topic-agnostic; one of each is correct and three of each
    would be three times the Kafka connections for no behavioural difference."""
    dependencies = _build_runtime_handler_dependencies(None, _BOOTSTRAP)
    assert dependencies is not None
    entry = dependencies[_HANDLER_NAME]
    assert "producer" in entry and "quarantine_producer" in entry, sorted(entry)
    assert not isinstance(entry["producer"], dict), entry["producer"]
    assert not isinstance(entry["quarantine_producer"], dict), entry[
        "quarantine_producer"
    ]
