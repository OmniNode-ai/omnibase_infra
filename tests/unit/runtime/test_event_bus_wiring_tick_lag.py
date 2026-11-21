# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The subscription wiring tells the loop sampler about ticks (OMN-26249)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_core.models.contracts.subcontracts import ModelEventBusSubcontract
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.util_loop_lag_sampler import LoopLagSampler
from omnibase_infra.topics import SUFFIX_RUNTIME_TICK

_OTHER_TOPIC = "onex.evt.omnibase-infra.runtime-booted.v1"


def _wiring(sampler: LoopLagSampler) -> tuple[EventBusSubcontractWiring, Any]:
    bus = MagicMock()
    bus.subscribe = AsyncMock(return_value=AsyncMock())
    wiring = EventBusSubcontractWiring(
        event_bus=bus,
        dispatch_engine=MagicMock(),
        environment="dev",
        node_name="n",
        service="svc",
        version="v1",
        tick_observer=sampler,
    )
    return wiring, bus


def _subcontract(*topics: str) -> ModelEventBusSubcontract:
    return ModelEventBusSubcontract(
        version=ModelSemVer(major=1, minor=0, patch=0),
        subscribe_topics=list(topics),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subscribing_to_the_tick_marks_the_member_a_tick_subscriber() -> None:
    sampler = LoopLagSampler()
    wiring, _ = _wiring(sampler)
    await wiring.wire_subscriptions(_subcontract(SUFFIX_RUNTIME_TICK), "n")
    assert sampler.tick_lag() == (None, "no_tick_consumed_yet")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_member_that_never_subscribes_to_the_tick_stays_not_applicable() -> None:
    sampler = LoopLagSampler()
    wiring, _ = _wiring(sampler)
    await wiring.wire_subscriptions(_subcontract(_OTHER_TOPIC), "n")
    assert sampler.tick_lag() == (None, "not_a_tick_subscriber")


@pytest.mark.unit
def test_a_consumed_tick_records_its_lag_from_scheduled_at() -> None:
    sampler = LoopLagSampler()
    wiring, _ = _wiring(sampler)
    scheduled = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    wiring._observe_tick(SUFFIX_RUNTIME_TICK, {"scheduled_at": scheduled.isoformat()})
    lag, reason = sampler.tick_lag()
    assert reason is None
    assert lag is not None
    assert lag > 0.0


@pytest.mark.unit
def test_ticks_on_other_topics_and_unreadable_payloads_record_nothing() -> None:
    sampler = LoopLagSampler()
    wiring, _ = _wiring(sampler)
    wiring._observe_tick(_OTHER_TOPIC, {"scheduled_at": "2026-11-20T12:00:00+00:00"})
    wiring._observe_tick(SUFFIX_RUNTIME_TICK, {"scheduled_at": "not a time"})
    wiring._observe_tick(SUFFIX_RUNTIME_TICK, {})
    assert sampler.tick_lag() == (None, "not_a_tick_subscriber")
