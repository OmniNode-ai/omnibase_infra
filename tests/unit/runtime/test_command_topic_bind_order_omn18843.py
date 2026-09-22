# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command-topic consumers bind before event-topic consumers (OMN-18843).

Why bind ORDER is a correctness-adjacent property and not a preference:

Every subscription the runtime wires costs a full Kafka consumer-group join, and
the joins are serial. Measured inside ``omninode-runtime-effects`` on the .201
dev lane on 2026-09-21: the container started at 18:53:47Z, the kernel began
initialising at 18:54:00Z, and 228 groups joined one after another until
18:57:26Z -- about 3.5 s each. In plain descriptor order the delegate-skill
COMMAND topic joined at 18:54:48Z, 61 s after the container started, and for
every one of those 61 s ``onex delegate`` refused before publishing anything,
because its preflight found the consumer group Empty.

A command topic and an event topic are not symmetric in what a late bind costs:

* a COMMAND topic is a request surface. A client is blocked on it right now, it
  has a timeout, and an unbound group reads to it as "there is no deployed
  orchestrator" -- a refusal, not a delay.
* an EVENT topic is a fan-out with durable offsets. A consumer that binds a
  minute later reads the same messages from its committed offset. The cost is
  latency, and nothing is lost.

So the command topics go first. This does not make the runtime faster and is not
claimed to: the last event consumer binds at the same moment either way. It
moves the surface a human is waiting on to the front of a queue that already
existed.
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

COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"  # onex-topic-allow: OMN-18843 bind-order fixture
EVENT_TOPIC_A = "onex.evt.omniclaude.context-injected.v1"  # onex-topic-allow: OMN-18843 bind-order fixture
EVENT_TOPIC_B = "onex.evt.omnibase-infra.runtime-error.v1"  # onex-topic-allow: OMN-18843 bind-order fixture


class _RecordingWiring:
    """Stands in for ``EventBusSubcontractWiring`` and records wiring order."""

    def __init__(self, **_kwargs: Any) -> None:
        self.wired: list[str] = []

    async def wire_subscriptions(self, *, subcontract: Any, node_name: str) -> None:
        self.wired.append(node_name)


class _Descriptor:
    def __init__(self, name: str, contract_path: Path) -> None:
        self.name = name
        self.contract_path = str(contract_path)


def _write_contract(directory: Path, node_name: str, topics: list[str]) -> Path:
    """Write the smallest contract the boot wiring path accepts.

    ``runtime_profiles`` is omitted deliberately: a contract that claims a
    profile is skipped as profile-owned, and this test is about ordering among
    the contracts that DO get wired.
    """
    path = directory / f"{node_name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": node_name,
                "event_bus": {
                    "version": {"major": 1, "minor": 0, "patch": 0},
                    "subscribe_topics": topics,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _host_with_descriptors(descriptors: dict[str, _Descriptor]) -> RuntimeHostProcess:
    """A RuntimeHostProcess carrying only what the boot wiring path reads.

    Built without ``__init__`` on purpose. The method under test reads six
    attributes and calls one helper; constructing a real host would drag in a
    container, a broker and a dispatch engine, none of which this behaviour
    depends on, and would make an ordering test fail for unrelated reasons.
    """
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


@pytest.mark.unit
def test_command_topic_is_recognised_behind_an_optional_namespace() -> None:
    """The classifier reads the kind segment, not a fixed leading string.

    A transport-applied ``KAFKA_TOPIC_NAMESPACE`` prefix must not turn a command
    topic into an event topic by shifting the segment offsets.
    """
    assert _is_command_topic(COMMAND_TOPIC)
    assert _is_command_topic(f"lab.{COMMAND_TOPIC}")
    assert not _is_command_topic(EVENT_TOPIC_A)
    assert not _is_command_topic(f"lab.{EVENT_TOPIC_A}")
    # Negative controls: nothing that merely contains the token counts.
    assert not _is_command_topic(
        "onex.evt.omnimarket.cmd-audit.v1"
    )  # onex-topic-allow: OMN-18843 negative control
    assert not _is_command_topic("cmd.something.else")
    assert not _is_command_topic("")


@pytest.mark.unit
async def test_command_topic_handlers_are_wired_before_event_topic_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OMN-18843 ordering: the command surface leaves the queue first.

    The command contract is deliberately placed LAST in descriptor order, which
    is dict insertion order, so a pass that preserved the old behaviour would
    wire it last and fail here.
    """
    early = _write_contract(tmp_path, "event-consumer-early", [EVENT_TOPIC_A])
    late = _write_contract(tmp_path, "event-consumer-late", [EVENT_TOPIC_B])
    command = _write_contract(tmp_path, "delegate-orchestrator", [COMMAND_TOPIC])

    descriptors = {
        "event-consumer-early": _Descriptor("event-consumer-early", early),
        "event-consumer-late": _Descriptor("event-consumer-late", late),
        "delegate-orchestrator": _Descriptor("delegate-orchestrator", command),
    }

    recorder = _RecordingWiring()
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process.EventBusSubcontractWiring",
        lambda **kwargs: recorder,
    )

    host = _host_with_descriptors(descriptors)
    await host._wire_event_bus_subscriptions()

    assert recorder.wired[0] == "delegate-orchestrator", (
        "The command-topic handler must be wired first; each subscription costs "
        "a serial consumer-group join, and a client is blocked on the command "
        "surface while the event surfaces are not (OMN-18843). Order was "
        f"{recorder.wired}"
    )
    assert set(recorder.wired) == set(descriptors), (
        f"Reordering must not drop or duplicate a handler. Order was {recorder.wired}"
    )


@pytest.mark.unit
async def test_relative_order_within_each_group_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the two GROUPS are reordered; order inside a group is untouched.

    Without this, the change would be a licence to reshuffle event-consumer bind
    order as well, and a handler that today happens to bind before another would
    silently stop doing so.
    """
    first = _write_contract(tmp_path, "event-a", [EVENT_TOPIC_A])
    second = _write_contract(tmp_path, "event-b", [EVENT_TOPIC_B])
    command = _write_contract(tmp_path, "command-a", [COMMAND_TOPIC])

    descriptors = {
        "event-a": _Descriptor("event-a", first),
        "command-a": _Descriptor("command-a", command),
        "event-b": _Descriptor("event-b", second),
    }

    recorder = _RecordingWiring()
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process.EventBusSubcontractWiring",
        lambda **kwargs: recorder,
    )

    host = _host_with_descriptors(descriptors)
    await host._wire_event_bus_subscriptions()

    assert recorder.wired == ["command-a", "event-a", "event-b"], recorder.wired
