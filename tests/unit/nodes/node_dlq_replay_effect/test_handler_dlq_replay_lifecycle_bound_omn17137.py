# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-17137 second pass -- the bound held over the RECORD LOOP and nowhere else.

The first pass (``test_handler_dlq_replay_idle_wait_omn17137.py``) proved that
``run()`` stops waiting for a record once ``max_run_duration_seconds`` is spent.
That fix shipped in ``omnibase_infra#3021`` (``671310e2f``), is present in the
image the stability lane was running, and the lane parked anyway.

The reason is that ``max_run_duration_seconds`` bounds ONE SEGMENT of the
dispatch. The thing that must actually be bounded is the dispatch itself,
because the auto-wired outer trigger consumer's ``_consume_loop`` awaits it
serially -- while it is awaiting, that consumer is not polling. Three awaits sat
outside the budget and every one of them was unbounded:

* taking the per-consumer run mutex (``_run_lock_for``),
* ``_ensure_runtime_dependencies_started`` -> ``consumer.start()``,
* ``_stop_runtime_dependencies`` -> ``consumer.stop()``, in a ``finally``.

Live measurement, .201 dev lane, 2026-09-16 (read-only), which is what this file
encodes as tests. The node declares three DLQ subscribe topics; since OMN-18119
(``omnibase_infra#3388``) it holds one ``DLQConsumer`` per topic, and all three
sit in the SINGLE Kafka group ``onex-dlq-replay``. The handler starts and stops
one of them on EVERY trigger message, so each dispatch forces two group-wide
rebalances::

    14:21:59  Joined group 'onex-dlq-replay' (generation 227658)
    14:22:29  Heartbeat failed for group onex-dlq-replay because it is rebalancing
    14:22:29  Joined group 'onex-dlq-replay' (generation 227675)
    14:27:23  Revoking previously assigned partitions
                  frozenset({TopicPartition('onex.dlq.omnibase-infra.events.v1', 0)})
                  for group onex-dlq-replay
    <nothing further on that group, while the container logged normally for
     another ten minutes>

Generation 227,675 eight minutes after a cold boot is the churn. The revocation
at 14:27:23 with no rejoin after it is a ``stop()`` that never returned. The
dispatch therefore never returned either, so the outer trigger consumer stopped
polling and aiokafka evicted it at ``KAFKA_MAX_POLL_INTERVAL_MS=1800000`` --
after which nothing rejoined it, because a rejoin only ever happens on the next
poll and the loop was still inside that same ``await``.

Corroborating state on the stability lane at 2026-09-16T14:24Z, same shape::

    ...__t.onex.dlq.omnibase-infra.commands.v1   Empty   0 members   lag 23,357
    ...__t.onex.dlq.omnibase-infra.events.v1     Empty   0 members   lag 24,682
    ...__t.onex.dlq.omnibase-infra.intents.v1    Stable  1 member    lag 0

``intents.v1`` survives because it carries no DLQ traffic, so its outer loop
never enters ``run()`` at all -- the same per-topic asymmetry the first pass
recorded, now with two casualties instead of one because OMN-18119 gave
``commands.v1`` a drain of its own that can wedge.

``/ready`` on that lane reported ``consume_tasks_alive`` TRUE for both dead
topics, which is the fact that rules out "the consume loop raised and exited":
the task is alive and inside the dispatch, not gone.

Ticket: OMN-17137
Evidence-Ticket: OMN-17137
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from omnibase_infra.errors import DlqDependencyLifecycleTimeoutError
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
    _run_lock_for,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

pytestmark = pytest.mark.unit

# Each run() is wrapped so a regression fails LOUDLY instead of hanging the
# suite forever. Far above the run's own budget, far below "forever".
_OUTER_TIMEOUT_SECONDS = 5.0

# The lifecycle bound used throughout. Small enough that a wedged start/stop is
# observed in well under the outer timeout.
_LIFECYCLE_BUDGET_SECONDS = 0.3


def _config(topic: str, **overrides: object) -> ModelDlqReplayEngineConfig:
    base: dict[str, object] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": topic,
        "max_replay_count": 5,
        "max_records_per_run": 1000,
        "max_run_duration_seconds": 0.3,
        "commit_every_n_records": 1000,
        "dependency_lifecycle_timeout_seconds": _LIFECYCLE_BUDGET_SECONDS,
    }
    base.update(overrides)
    return ModelDlqReplayEngineConfig(**base)  # type: ignore[arg-type]


def _message() -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic="dev.orders.command.v1",
        original_key="k",
        original_value='{"hello": "world"}',
        original_offset="10",
        original_partition=0,
        failure_reason="boom",
        failure_timestamp="2026-06-02T00:00:00Z",
        correlation_id=uuid4(),
        retry_count=0,
        error_type="InfraConnectionError",  # retryable -> replay-eligible
        dlq_offset=42,
        dlq_partition=1,
        raw_payload={"original_topic": "dev.orders.command.v1"},
    )


class _WedgingConsumer:
    """A consumer whose ``stop()`` never returns -- the measured live shape.

    ``AIOKafkaConsumer.stop()`` awaits a coordinator close. Issued into the
    permanent rebalance the shared ``onex-dlq-replay`` group is in, that close
    does not settle. Nothing raises and nothing times out on its own, which is
    exactly why an unbounded ``await stop()`` is terminal for the outer
    consumer rather than merely slow.
    """

    def __init__(
        self,
        config: ModelDlqReplayEngineConfig,
        *,
        burst: int = 2,
        wedge_start: bool = False,
        wedge_stop: bool = True,
    ) -> None:
        self.config = config
        self._burst = burst
        self._wedge_start = wedge_start
        self._wedge_stop = wedge_stop
        self._started = False
        self.commits = 0
        self.yielded = 0
        self.stop_entered = False
        self.start_entered = False

    async def start(self) -> None:
        self.start_entered = True
        if self._wedge_start:
            await asyncio.Event().wait()
        self._started = True

    async def stop(self) -> None:
        self.stop_entered = True
        if self._wedge_stop:
            await asyncio.Event().wait()
        self._started = False

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        for _ in range(self._burst):
            self.yielded += 1
            yield _message()
        await asyncio.Event().wait()

    async def commit(self) -> None:
        self.commits += 1

    async def commit_offsets(self, offsets: object) -> None:
        self.commits += 1
        self.committed_offsets = dict(offsets)  # type: ignore[arg-type]


class _NoopEffect:
    """Fake producer / quarantine-producer: succeeds without publishing."""

    def __init__(self) -> None:
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def replay_message(
        self, message: object, replay_correlation_id: object
    ) -> None:
        return None

    async def quarantine_message(
        self, message: object, reason: str, quarantine_correlation_id: object
    ) -> None:
        return None


def _handler(*consumers: object) -> HandlerDlqReplay:
    return HandlerDlqReplay(
        consumers={c.config.dlq_topic: c for c in consumers},  # type: ignore[attr-defined,misc]
        producer=_NoopEffect(),  # type: ignore[arg-type]
        quarantine_producer=_NoopEffect(),  # type: ignore[arg-type]
        tracking=None,
    )


async def test_run_returns_when_consumer_stop_never_completes() -> None:
    """THE regression test for the second pass.

    Pre-fix this hangs in ``_stop_runtime_dependencies``' unbounded
    ``await stop()`` until the outer ``wait_for`` kills it -- which is the live
    wedge, reproduced. Post-fix ``run()`` returns, having abandoned the stop.
    """
    consumer = _WedgingConsumer(_config("onex.dlq.omnibase-infra.events.v1"))
    handler = _handler(consumer)

    started = time.monotonic()
    result = await asyncio.wait_for(handler.run(), timeout=_OUTER_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - started

    assert consumer.stop_entered, "the teardown under test was never reached"
    assert elapsed < _OUTER_TIMEOUT_SECONDS, (
        "run() did not return on its own; the dispatch is still unbounded and "
        "the outer trigger consumer would be evicted at max_poll_interval_ms"
    )
    # The batch that DID drain is still reported. A teardown timeout must not
    # discard committed work -- that would redeliver every record in it.
    assert result.total_processed == consumer.yielded == 2


async def test_run_returns_when_consumer_start_never_completes() -> None:
    """``AIOKafkaConsumer.start()`` joins a group; a join into a permanent
    rebalance does not return. The run must end, not park."""
    consumer = _WedgingConsumer(
        _config("onex.dlq.omnibase-infra.events.v1"),
        wedge_start=True,
        wedge_stop=False,
    )
    handler = _handler(consumer)

    with pytest.raises(DlqDependencyLifecycleTimeoutError) as excinfo:
        await asyncio.wait_for(handler.run(), timeout=_OUTER_TIMEOUT_SECONDS)

    assert consumer.start_entered
    assert "start()" in str(excinfo.value)


async def test_run_returns_when_a_peer_holds_the_run_mutex_forever() -> None:
    """A wedged peer must cost this run its topic, not its liveness.

    This is the propagation step the live lane showed: one stuck consumer made
    every later run queue behind the same mutex, so a single wedge took out
    every trafficked topic rather than one.
    """
    consumer = _WedgingConsumer(
        _config("onex.dlq.omnibase-infra.events.v1"), wedge_stop=False
    )
    handler = _handler(consumer)

    lock = _run_lock_for(consumer)
    await lock.acquire()  # stand in for the peer run that never finishes
    try:
        started = time.monotonic()
        result = await asyncio.wait_for(handler.run(), timeout=_OUTER_TIMEOUT_SECONDS)
        elapsed = time.monotonic() - started
    finally:
        lock.release()

    assert elapsed < _OUTER_TIMEOUT_SECONDS, (
        "run() parked on the run mutex; a peer's wedge propagates to every "
        "later dispatch and takes the whole node's trigger groups down"
    )
    assert result.total_processed == 0
    assert consumer.yielded == 0, "the topic must be skipped, not half-drained"


async def test_a_wedged_topic_does_not_permanently_starve_its_declared_peers() -> None:
    """The blast radius this closes, stated as what the design guarantees.

    A wedged teardown spends the run's shared wall clock, so the topic BEHIND
    it in this run's order is skipped -- that is OMN-18119's declared shared
    clock working, not a defect, and the rotation is what makes it survivable:
    the skipped topic leads the order on the next run.

    What the bound changes is that there IS a next run. Unbounded, the first
    dispatch never returned at all, so the peer was not merely skipped this
    pass, it was unreachable for the life of the container -- which is why
    ``commands.v1`` and ``events.v1`` went Empty together on the stability lane
    while untrafficked ``intents.v1`` stayed Stable.
    """
    wedged = _WedgingConsumer(_config("onex.dlq.omnibase-infra.events.v1"))
    healthy = _WedgingConsumer(
        _config("onex.dlq.omnibase-infra.commands.v1"), wedge_stop=False
    )
    handler = _handler(wedged, healthy)

    first = await asyncio.wait_for(handler.run(), timeout=_OUTER_TIMEOUT_SECONDS)
    assert wedged.stop_entered
    assert first.total_processed == wedged.yielded == 2
    assert healthy.yielded == 0, (
        "the shared wall clock was already spent; the peer is expected to be "
        "skipped on THIS pass (OMN-18119)"
    )

    # The rotation puts the skipped topic at the head of the next run's order.
    second = await asyncio.wait_for(handler.run(), timeout=_OUTER_TIMEOUT_SECONDS)
    assert healthy.yielded == 2, (
        "the peer never got its turn across two runs; the wedge is starving "
        "the rotation rather than merely costing it one pass"
    )
    assert second.total_processed == 2


async def test_lifecycle_bound_is_declared_not_hardcoded() -> None:
    """The bound is a contract-declared config field, so a lane can tune it and
    a reader can see what it is -- never a literal buried in the teardown."""
    config = _config("onex.dlq.omnibase-infra.events.v1")
    assert config.dependency_lifecycle_timeout_seconds == _LIFECYCLE_BUDGET_SECONDS
    assert (
        ModelDlqReplayEngineConfig(
            bootstrap_servers="localhost:9092",
            dlq_topic="onex.dlq.omnibase-infra.events.v1",
        ).dependency_lifecycle_timeout_seconds
        > 0.0
    )
