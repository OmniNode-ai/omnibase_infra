# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-then-GREEN proof for OMN-14498 / OMN-15029 — the auto-wired sync
event_publisher's fire-and-forget publish task must not silently discard a
downstream publish failure.

``_make_sync_event_publisher`` adapts a legacy sync handler's ``publish()``
call to the async event bus by scheduling the actual publish as a detached
asyncio Task/Future (``_await_event_bus_publish``) and attaching
``_log_publish_failure`` as its ``add_done_callback``. Before this fix, that
done_callback did exactly one thing on failure: ``logger.error(...)``. The
handler had already returned by the time the publish failed, so there is
nothing to re-raise into -- but the failed payload itself vanished with
nothing durable: no DLQ, no metric, no redelivery. This is the OMN-14498
"auto-wired consume path swallows publish exceptions" defect confirmed still
live on ``origin/dev`` by the OMN-15029 false-Done reopen (2026-07-24
proof-debt sweep): ``git show origin/dev:.../handler_wiring.py`` showed
``_log_publish_failure`` still only logging, never routing to a DLQ.

``test_publish_failure_still_only_logs_no_dlq_pre_fix`` is the RED half,
pinned as a permanent regression guard for the specific "no event_bus DLQ
capability" degenerate case. ``test_publish_failure_routes_to_dlq_instead_of_
vanishing`` is the GREEN half proving the failure becomes durably observable
at a real surface: the SAME best-effort ``_publish_raw_to_dlq`` duck-typed
contract ``_make_event_bus_callback._route_swallowed_exception`` already uses
for the main consume boundary (OMN-14507), driven end-to-end through the real
``_make_sync_event_publisher`` production function -- never a stubbed
``publisher`` surrogate.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_sync_event_publisher,
)

_TARGET_TOPIC = "onex.evt.platform.context-roi.v1"


def _failing_dlq_capable_event_bus(exc: Exception) -> MagicMock:
    """A bus mock spec'd to EventBusKafka (only real attributes settable,
    satisfying the transport-mock-lint gate, OMN-13026) whose ``publish()``
    raises and whose ``_publish_raw_to_dlq`` (MixinKafkaDlq) is a real,
    observable AsyncMock."""
    bus = MagicMock(spec=EventBusKafka)
    bus.publish = AsyncMock(side_effect=exc)
    bus._publish_raw_to_dlq = AsyncMock(return_value=True)
    return bus


async def _drain_scheduled_tasks(
    condition: Callable[[], bool], timeout: float = 5.0
) -> None:
    """Yield control repeatedly until ``condition()`` is True or timeout.

    The publish failure is handled by a chain of two scheduled tasks (the
    publish task itself, then the DLQ-routing task its done_callback
    schedules) -- a single ``asyncio.sleep(0)`` is not guaranteed to drain
    both hops.
    """
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_publish_failure_still_only_logs_no_dlq_pre_fix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RED regression guard: with no DLQ-capable event bus, the failure is
    still only logged -- there is genuinely nothing durable to route to, and
    this must not raise into the caller or crash the loop."""
    boom = RuntimeError("kafka producer send failed")
    event_bus = MagicMock(spec=EventBusKafka)
    event_bus.publish = AsyncMock(side_effect=boom)
    del event_bus._publish_raw_to_dlq  # no DLQ capability at all on this bus

    publisher = _make_sync_event_publisher(
        event_bus=event_bus, handler_name="HandlerContextRoiRunner"
    )

    with caplog.at_level("ERROR"):
        publisher(_TARGET_TOPIC, b'{"ok":true}')
        await _drain_scheduled_tasks(
            lambda: "Auto-wired event_publisher publish failed" in caplog.text
        )

    assert "Auto-wired event_publisher publish failed" in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_publish_failure_routes_to_dlq_instead_of_vanishing() -> None:
    """GREEN: a real publish failure through the real
    ``_make_sync_event_publisher`` production path lands in the DLQ instead
    of vanishing with only a log line -- the observable, durable surface a
    real operator/consumer can read back from, not merely "an exception was
    raised somewhere"."""
    boom = RuntimeError("kafka producer send failed")
    event_bus = _failing_dlq_capable_event_bus(boom)

    publisher = _make_sync_event_publisher(
        event_bus=event_bus, handler_name="HandlerContextRoiRunner"
    )

    publisher(_TARGET_TOPIC, b'{"ok":true}')
    await _drain_scheduled_tasks(lambda: event_bus._publish_raw_to_dlq.await_count > 0)

    event_bus._publish_raw_to_dlq.assert_awaited_once()
    call_kwargs = event_bus._publish_raw_to_dlq.call_args.kwargs
    assert call_kwargs["original_topic"] == _TARGET_TOPIC
    assert call_kwargs["error"] is boom
    assert call_kwargs["raw_msg"] is not None
    # The failed outgoing payload must be the thing preserved in the DLQ.
    assert getattr(call_kwargs["raw_msg"], "value", None) == b'{"ok":true}'
