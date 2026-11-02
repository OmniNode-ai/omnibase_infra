# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21996: the delegate runtime's own bus factory keeps the group-less read."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_core.protocols.runtime.protocol_local_runtime_bus import (
    ProtocolLocalRuntimeUngroupedBus,
    UnsubscribeCallback,
)
from omnibase_core.protocols.runtime.protocol_local_runtime_message import (
    ProtocolLocalRuntimeMessage,
)
from omnibase_infra.cli.receipt_mode import (
    DelegatePhaseStopwatch,
    DelegatePhaseTimedRuntime,
)
from omnibase_infra.models.delegation.model_delegate_phase_timed_bus import (
    DelegatePhaseTimedBus,
    DelegatePhaseTimedUngroupedBus,
)

pytestmark = pytest.mark.integration

Callback = Callable[[ProtocolLocalRuntimeMessage], Awaitable[None]]
TOPIC = "onex.evt.omnimarket.delegate-skill-failed.v1"


class _UngroupedInmemoryBus(EventBusInmemory):
    """The real in-memory bus plus the group-less read the Kafka bus offers."""

    def __init__(self) -> None:
        super().__init__()
        self.ungrouped_topics: list[str] = []

    async def subscribe_ungrouped(
        self, topic: str, on_message: Callback
    ) -> UnsubscribeCallback:
        self.ungrouped_topics.append(topic)

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe


async def _noop(_: ProtocolLocalRuntimeMessage) -> None:
    return None


def _runtime(tmp_path: Path) -> DelegatePhaseTimedRuntime:
    return DelegatePhaseTimedRuntime(
        workflow_path=tmp_path / "contract.yaml",
        state_root=tmp_path / "state",
        phase_stopwatch=DelegatePhaseStopwatch(),
    )


@pytest.mark.asyncio
async def test_runtime_factory_wraps_ungrouped_bus_and_reads_without_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inner = _UngroupedInmemoryBus()
    monkeypatch.setattr(
        "omnibase_core.runtime.runtime_local.RuntimeLocal._create_event_bus",
        lambda self: inner,
    )
    runtime = _runtime(tmp_path)
    runtime._expected_correlation_id = uuid.uuid4()

    bus = runtime._create_event_bus()
    assert isinstance(bus, DelegatePhaseTimedUngroupedBus)
    assert isinstance(bus, ProtocolLocalRuntimeUngroupedBus)

    await runtime._subscribe_run_scoped(bus, TOPIC, _noop, "runtime_local_terminal_t1")

    assert inner.ungrouped_topics == [TOPIC]


def test_runtime_factory_keeps_plain_timed_bus_for_a_grouped_only_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "omnibase_core.runtime.runtime_local.RuntimeLocal._create_event_bus",
        lambda self: EventBusInmemory(),
    )

    bus = _runtime(tmp_path)._create_event_bus()

    assert type(bus) is DelegatePhaseTimedBus
