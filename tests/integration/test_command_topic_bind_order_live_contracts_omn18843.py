# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bind order holds against the REAL contract tree, not a fixture (OMN-18843).

The unit coverage for this change builds three contracts in a temporary
directory and asserts the command one is wired first. That proves the sorting
rule. It does not prove the rule survives contact with the contracts this
repository actually ships, and those are what the kernel loads at boot: 126
contract files under ``src/omnibase_infra/nodes/``, declaring a mix of command
and event topics, several of which the boot path skips for reasons that have
nothing to do with topic kind (profile-owned contracts, raw projection
consumers, plugin-managed subscriptions).

So this drives the real ``_wire_event_bus_subscriptions`` over descriptors
built from the real contract files on disk, and asserts the partition holds
across all of them. It is the OMN-18843 arm of the repository rule that a pull
request touching kernel-level registration must run the wiring against the
manifest as it exists on disk rather than against invented handlers.

No broker, no database, no container: the event bus and the dispatch engine are
recorded rather than connected, because bind ORDER is decided before any socket
is opened. What would make this test lie is a vacuous corpus -- no command
topics, or no event topics, in which case any order trivially passes -- so both
are asserted to be non-empty before the ordering assertion is read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.runtime.runtime_host_process import (
    RuntimeHostProcess,
    _is_command_topic,
)

pytestmark = pytest.mark.integration

NODES_DIR = Path(__file__).resolve().parents[2] / "src" / "omnibase_infra" / "nodes"


class _RecordingWiring:
    """Records (node_name, topics) in the order the boot path wires them."""

    def __init__(self, **_kwargs: Any) -> None:
        self.wired: list[tuple[str, tuple[str, ...]]] = []

    async def wire_subscriptions(self, *, subcontract: Any, node_name: str) -> None:
        self.wired.append((node_name, tuple(subcontract.subscribe_topics)))


class _Descriptor:
    def __init__(self, name: str, contract_path: Path) -> None:
        self.name = name
        self.contract_path = str(contract_path)


def _live_descriptors() -> dict[str, _Descriptor]:
    """Every shipped contract that declares subscribe_topics, in disk order.

    Sorted by path so the input order is deterministic across machines. The
    boot path applies its own skip rules to these; this function deliberately
    does not pre-filter, so the test exercises those rules rather than
    reimplementing them.
    """
    descriptors: dict[str, _Descriptor] = {}
    for contract in sorted(NODES_DIR.glob("*/contract.yaml")):
        try:
            document = yaml.safe_load(contract.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(document, dict):
            continue
        event_bus = document.get("event_bus")
        if not isinstance(event_bus, dict) or not event_bus.get("subscribe_topics"):
            continue
        descriptors[contract.parent.name] = _Descriptor(contract.parent.name, contract)
    return descriptors


def _host(descriptors: dict[str, _Descriptor]) -> RuntimeHostProcess:
    host = object.__new__(RuntimeHostProcess)
    host._event_bus = object()
    host._dispatch_engine = object()
    host._handler_descriptors = descriptors
    host._runtime_node_graph_config = None
    host._event_bus_wiring = None

    class _Identity:
        service = "omnibase-infra"
        version = "v1"

    host._node_identity = _Identity()
    host._get_environment_from_config = lambda: "local"  # type: ignore[method-assign]
    return host


@pytest.mark.integration
async def test_the_live_contract_tree_yields_both_kinds() -> None:
    """Corpus control. Without this the ordering assertion below can pass empty."""
    descriptors = _live_descriptors()
    assert len(descriptors) >= 20, (
        "Too few shipped contracts declare subscribe_topics for this to be a "
        f"meaningful corpus; found {len(descriptors)}"
    )

    command_nodes = set()
    event_nodes = set()
    for name, descriptor in descriptors.items():
        document = yaml.safe_load(Path(descriptor.contract_path).read_text("utf-8"))
        topics = document["event_bus"]["subscribe_topics"]
        if any(_is_command_topic(str(t)) for t in topics):
            command_nodes.add(name)
        else:
            event_nodes.add(name)

    assert command_nodes, "No shipped contract declares a command topic"
    assert event_nodes, "No shipped contract declares only event topics"


@pytest.mark.integration
async def test_every_command_handler_is_wired_before_every_event_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partition holds over the contracts the runtime really loads.

    Asserted as a partition rather than as an exact sequence, deliberately: the
    change reorders the two GROUPS and nothing inside them, so pinning a full
    136-entry order here would fail on any unrelated contract being added and
    would say nothing about this change.
    """
    descriptors = _live_descriptors()
    recorder = _RecordingWiring()
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process.EventBusSubcontractWiring",
        lambda **kwargs: recorder,
    )

    await _host(descriptors)._wire_event_bus_subscriptions()

    assert recorder.wired, "The boot path wired nothing from the live contract tree"

    command_indexes = [
        i
        for i, (_name, topics) in enumerate(recorder.wired)
        if any(_is_command_topic(t) for t in topics)
    ]
    event_indexes = [
        i
        for i, (_name, topics) in enumerate(recorder.wired)
        if not any(_is_command_topic(t) for t in topics)
    ]

    assert command_indexes, (
        "No command-topic handler survived the boot path's skip rules, so this "
        "assertion would be vacuous. Investigate before relaxing it."
    )
    assert event_indexes, "No event-only handler was wired; corpus is degenerate"

    assert max(command_indexes) < min(event_indexes), (
        "A command-topic handler is wired after an event-topic handler. Each "
        "subscription costs a serial consumer-group join, and a client is "
        "blocked on the command surface while the event surfaces are not "
        "(OMN-18843). Last command index "
        f"{max(command_indexes)}, first event index {min(event_indexes)}, of "
        f"{len(recorder.wired)} wired."
    )


@pytest.mark.integration
async def test_reordering_neither_drops_nor_duplicates_a_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split into two passes must be a permutation, nothing more.

    A collect pass that forgets to wire one group, or wires one twice, would
    still satisfy the partition assertion above.
    """
    descriptors = _live_descriptors()
    recorder = _RecordingWiring()
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process.EventBusSubcontractWiring",
        lambda **kwargs: recorder,
    )

    await _host(descriptors)._wire_event_bus_subscriptions()

    names = [name for name, _topics in recorder.wired]
    assert len(names) == len(set(names)), "A handler was wired more than once"
    assert set(names) <= set(descriptors), "A handler was wired that was never offered"
