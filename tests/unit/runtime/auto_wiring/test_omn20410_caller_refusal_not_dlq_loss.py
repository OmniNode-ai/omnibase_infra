# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20410 -- an answered caller refusal is not a projection's DLQ loss.

On the .201 dev lane, one tenant with no registered provider key sent cloud
delegations from 2026-10-03T14Z on, and every one was refused at the routing
terminus (``CustomerKeyRefusedError``, ``ONEX_MARKET_CUSTOMER_PROVIDER_KEY_ABSENT``).
The consume boundary answered each with a typed, non-retryable terminal and
parked the record on the DLQ. It also counted it as ``messages_dlq`` against
``node_delegation_routing_reducer``, so whenever that tenant's refusals were the
only routing traffic in the flow windows, ``projection_dlq_saturation`` read a
100% total loss, the runtime went DEGRADED, and ``--degraded-policy fail`` took
the omninode-runtime container unhealthy every few monitor ticks. C28 run
37146175477 graded INDETERMINATE on exactly that container state.

A refusal of this shape is an answer, not a loss: the caller holds a typed
terminal naming the remediation, and no replay of the record can ever succeed.
It is counted as a handler error and stays out of ``messages_dlq``. Every other
failure keeps counting as DLQ flow -- the positive control below.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.models.observability.model_consumer_flow_delta import (
    ModelConsumerFlowDelta,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _BOUNDARY_DLQ_ENV,
    _make_dispatch_callback,
    _make_event_bus_callback,
)
from omnibase_infra.runtime.boundary_failure_terminal import (
    is_answered_caller_refusal,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.observability import (
    get_consumer_flow_counters,
    reset_consumer_flow_counters,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

_IN_TOPIC = "onex.evt.platform.node-heartbeat.v1"  # onex-topic-allow: test fixture topic shared with the OMN-16777 seam test
_OUT_TOPIC = "onex.cmd.omnibase-infra.gateway-link-health-upsert.v1"  # onex-topic-allow: test fixture topic shared with the OMN-16777 seam test
_GROUP = "onex-dev.omnimarket.delegation-routing-reducer.consume"
_KEY_ABSENT = "ONEX_MARKET_CUSTOMER_PROVIDER_KEY_ABSENT"
_REFUSAL_MESSAGE = (
    f"[{_KEY_ABSENT}] delegation.customer_provider_key.absent: delegation "
    "refused for tenant 'omn20410-probe' (task_type='document', surface=cloud): "
    "no provider key is registered for this tenant. Register a provider key "
    "for this tenant and retry."
)


class CustomerKeyRefusedError(Exception):
    """Mirror of omnimarket's class: same NAME and ``error_code`` (layering)."""

    def __init__(self, message: str, code: str = _KEY_ABSENT) -> None:
        super().__init__(message)
        self.error_code = code


class _ModelIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    node_id: UUID


class _HandlerRaises:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def handle(self, request: _ModelIn) -> None:
        raise self._exc


@pytest.fixture(autouse=True)
def _clean_counters(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "true")
    reset_consumer_flow_counters()
    yield
    reset_consumer_flow_counters()


def _bus() -> MagicMock:
    bus = MagicMock(spec=EventBusKafka)
    bus.publish_envelope = AsyncMock()
    bus._publish_raw_to_dlq = AsyncMock(return_value=True)
    return bus


def _engine_for(handler: _HandlerRaises) -> MessageDispatchEngine:
    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id="omn20410-dispatcher",
        dispatcher=_make_dispatch_callback(handler, None),
        category=EnumMessageCategory.EVENT,
        message_types=None,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="omn20410-route",
            topic_pattern="*.evt.platform.node-heartbeat.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="omn20410-dispatcher",
        )
    )
    engine.freeze()
    return engine


async def _one_window(
    exc: Exception, count: int
) -> tuple[ModelConsumerFlowDelta, MagicMock]:
    t0 = datetime(2026, 10, 3, 18, 58, 0, tzinfo=UTC)
    counters = get_consumer_flow_counters()
    carrier = uuid4()
    counters.drain(node_id=carrier, now=t0)
    bus = _bus()
    callback = _make_event_bus_callback(
        _IN_TOPIC,
        _engine_for(_HandlerRaises(exc)),
        DispatchResultApplier(event_bus=bus, output_topic=_OUT_TOPIC),
        event_bus=bus,
        allowed_dispatcher_ids={"omn20410-dispatcher"},
        consumer_group=_GROUP,
    )
    for _ in range(count):
        message = MagicMock()
        message.value = json.dumps({"node_id": str(uuid4())}).encode("utf-8")
        await callback(message)
    window = counters.drain(node_id=carrier, now=t0 + timedelta(seconds=60))
    assert window is not None
    rows = {(d.consumer_group, d.topic): d for d in window.consumer_deltas}
    return rows[(_GROUP, _IN_TOPIC)], bus


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_keyless_tenants_refusals_are_not_counted_as_dlq_loss() -> None:
    row, bus = await _one_window(CustomerKeyRefusedError(_REFUSAL_MESSAGE), 4)

    assert row.messages_in == 4
    assert row.messages_dlq == 0, (
        "an answered caller refusal was counted as projection DLQ loss; four "
        "of them alone read as a 100% total loss and take the runtime unhealthy"
    )
    assert row.handler_errors == 4
    # The record is still parked: the change is to the accounting, not the route.
    assert bus._publish_raw_to_dlq.await_count == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_any_other_failure_still_counts_as_dlq_loss() -> None:
    """Positive control: a real handler failure keeps saturating the dimension."""
    row, bus = await _one_window(ValueError("projection row rejected"), 4)

    assert row.messages_in == 4
    assert row.messages_dlq == 4
    assert bus._publish_raw_to_dlq.await_count == 4


@pytest.mark.unit
def test_only_the_key_absent_refusal_is_an_answered_caller_refusal() -> None:
    """The class alone is not enough: its other code is OUR routing defect."""
    assert is_answered_caller_refusal(CustomerKeyRefusedError(_REFUSAL_MESSAGE))
    platform_key = CustomerKeyRefusedError(
        "[ONEX_MARKET_CUSTOMER_ROUTE_PLATFORM_KEY] customer work resolved to a "
        "platform-owned key",
        code="ONEX_MARKET_CUSTOMER_ROUTE_PLATFORM_KEY",
    )
    assert not is_answered_caller_refusal(platform_key)
    assert not is_answered_caller_refusal(ValueError(_REFUSAL_MESSAGE))
    # The engine-flattened shape the boundary actually holds (no __cause__).
    flattened = RuntimeError(
        "dispatch to topic=onex.cmd.omnibase-infra.delegation-routing-request.v1 "
        "returned status=handler_error with no terminal output (dispatcher_id=): "
        f"Dispatcher 'routing' failed: CustomerKeyRefusedError: {_REFUSAL_MESSAGE}"
    )
    assert is_answered_caller_refusal(flattened)
