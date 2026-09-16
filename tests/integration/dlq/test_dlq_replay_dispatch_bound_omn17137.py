# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-17137 -- the DISPATCH the runtime awaits is bounded, end to end.

The unit suite
(``tests/unit/nodes/node_dlq_replay_effect/test_handler_dlq_replay_lifecycle_bound_omn17137.py``)
bounds ``run()``. This file asserts the property that actually keeps a consumer
group alive: that the call the auto-wired boundary awaits -- ``handle()``,
entered through the runtime's real materialized dispatch shape, with the real
engine config model and the real typed error -- returns even when the
underlying Kafka dependency lifecycle never completes.

Why that distinction is the whole bug. The outer trigger consumer's
``_consume_loop`` is a serial ``async for msg in consumer:`` that awaits each
dispatch. It is not polling while it awaits. aiokafka evicts a consumer that has
not polled within ``max_poll_interval_ms`` (1,800,000 on the .201 lanes) and a
rejoin only ever happens on the NEXT poll -- so one dispatch that never returns
takes the group to zero members permanently, for the life of the container.
Measured on the .201 dev lane 2026-09-16: the last line on the shared
``onex-dlq-replay`` group is a partition revocation at 14:27:23Z with no rejoin
after it, while the container logged normally for another ten minutes.

``run()`` returning is necessary but is not the property the runtime depends on;
``handle()`` returning is. Bounding one and asserting the other is what let the
first pass ship and the lane park anyway, so this file asserts the outer one.

No external infrastructure is required and this test never skips: the failure
being reproduced is a dependency whose ``stop()`` never returns, which is
expressed exactly by a fake that never returns. A real broker is neither
necessary to express it nor sufficient to make it deterministic.

Ticket: OMN-17137
Evidence-Ticket: OMN-17137
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

pytestmark = pytest.mark.integration

# A dispatch that has not returned by here would, in the runtime, be on its way
# to a max_poll_interval_ms eviction. Far above the handler's own budgets.
_DISPATCH_DEADLINE_SECONDS = 10.0

_LIFECYCLE_BUDGET_SECONDS = 0.3


def _config(topic: str) -> ModelDlqReplayEngineConfig:
    return ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=topic,
        max_replay_count=5,
        max_records_per_run=1000,
        max_run_duration_seconds=0.3,
        commit_every_n_records=1000,
        dependency_lifecycle_timeout_seconds=_LIFECYCLE_BUDGET_SECONDS,
    )


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


def _materialized_dispatch_dict() -> dict[str, object]:
    """The shape ``MessageDispatchEngine._execute_dispatcher`` hands a dispatcher.

    ``HandlerDlqReplay.handle``'s first parameter is named ``envelope``, so the
    auto-wiring boundary classifies it as envelope-accepting and, for an
    ``operation_match`` handler, delivers this raw dict rather than a hydrated
    ``ModelEventEnvelope`` (OMN-15021). Using it here is what makes this an
    integration assertion rather than a second unit test.
    """
    correlation_id = str(uuid4())
    return {
        "payload": {
            "original_topic": "onex.evt.omnibase-infra.runtime-booted.v1",
            "failure_reason": "unroutable operation",
            "correlation_id": correlation_id,
        },
        "__bindings": {},
        "__debug_trace": {
            "event_type": None,
            "correlation_id": correlation_id,
            "trace_id": None,
            "causation_id": None,
            "topic": "onex.dlq.omnibase-infra.events.v1",
            "timestamp": "2026-09-16T14:27:23Z",
        },
    }


class _StopNeverReturnsConsumer:
    """The measured live shape: ``stop()`` enters and never comes back.

    ``AIOKafkaConsumer.stop()`` awaits a coordinator close. Issued into the
    permanent rebalance the shared ``onex-dlq-replay`` group is in -- generation
    227,675 eight minutes after a cold boot on the dev lane -- that close does
    not settle. Nothing raises, nothing times out on its own.
    """

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config
        self._started = False
        self.yielded = 0
        self.commits = 0
        self.stop_entered = False

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self.stop_entered = True
        await asyncio.Event().wait()

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        for _ in range(2):
            self.yielded += 1
            yield _message()
        await asyncio.Event().wait()

    async def commit(self) -> None:
        self.commits += 1

    async def commit_offsets(self, offsets: object) -> None:
        self.commits += 1


class _NoopEffect:
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


async def test_dispatch_returns_when_the_dependency_teardown_never_completes() -> None:
    """The property the outer consumer's liveness actually rests on.

    Pre-fix this hangs in ``_stop_runtime_dependencies``' unbounded
    ``await stop()``, which in the runtime is a consumer group that goes Empty
    at ``max_poll_interval_ms`` and never rejoins.
    """
    consumer = _StopNeverReturnsConsumer(_config("onex.dlq.omnibase-infra.events.v1"))
    handler = HandlerDlqReplay(
        consumers={consumer.config.dlq_topic: consumer},  # type: ignore[dict-item]
        producer=_NoopEffect(),  # type: ignore[arg-type]
        quarantine_producer=_NoopEffect(),  # type: ignore[arg-type]
        tracking=None,
    )

    started = time.monotonic()
    output = await asyncio.wait_for(
        handler.handle(_materialized_dispatch_dict()),  # type: ignore[arg-type]
        timeout=_DISPATCH_DEADLINE_SECONDS,
    )
    elapsed = time.monotonic() - started

    assert consumer.stop_entered, "the teardown under test was never reached"
    assert elapsed < _DISPATCH_DEADLINE_SECONDS, (
        "handle() did not return; in the runtime the outer trigger consumer "
        "would now be on its way to a max_poll_interval_ms eviction it can "
        "never rejoin from"
    )
    assert output.result is not None
    assert output.result.total_processed == consumer.yielded == 2


async def test_repeated_dispatches_stay_bounded_so_the_loop_keeps_polling() -> None:
    """One wedge must not accumulate across deliveries.

    The node is auto-wired as a PER-MESSAGE trigger on the topics it drains, so
    a backlog delivers many dispatches back to back. Each must return on its own
    budget; a bound that only holds for the first is not a bound.
    """
    consumer = _StopNeverReturnsConsumer(_config("onex.dlq.omnibase-infra.events.v1"))
    handler = HandlerDlqReplay(
        consumers={consumer.config.dlq_topic: consumer},  # type: ignore[dict-item]
        producer=_NoopEffect(),  # type: ignore[arg-type]
        quarantine_producer=_NoopEffect(),  # type: ignore[arg-type]
        tracking=None,
    )

    for delivery in range(3):
        started = time.monotonic()
        await asyncio.wait_for(
            handler.handle(_materialized_dispatch_dict()),  # type: ignore[arg-type]
            timeout=_DISPATCH_DEADLINE_SECONDS,
        )
        assert time.monotonic() - started < _DISPATCH_DEADLINE_SECONDS, (
            f"dispatch {delivery} parked; the wedge accumulates across "
            "deliveries and the poll loop stops"
        )
