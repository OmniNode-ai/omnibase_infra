# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18084 — a handler failure on a DLQ topic must not be re-dead-lettered.

MEASURED on the .201 dev lane 2026-09-09T16:27:34Z: the events DLQ took 93,027
records in 480 s (193.8 msg/s = 16.74M/day, ~151 GB/day) and every one of a
2,000-record sample carried ``original_topic:
onex.dlq.omnibase-infra.events.v1`` -- the topic it was consumed FROM. The DLQ
replay handler is subscribed to that topic, fails on every message, and the
boundary answers the failure by writing it back to the same topic. Amplification
is 1:1 and unbounded; ``/data`` on that host had 590 GB free against ~1.05 TB of
projected steady state.

THE GUARD ALREADY EXISTED AND WAS WIRED TO TWO OF THREE LEGS.
``_is_dead_letter_source_topic`` (OMN-16798) is consulted on the
``NO_DISPATCHER`` leg and on the no-result-applier leg. The ``HANDLER_ERROR``
leg -- ``_raise_if_silent_dispatch_failure`` raising into
``_route_swallowed_exception``, which resolves its sink with
``get_dlq_topic_for_original``, a FIXED POINT on ``onex.dlq.*`` names -- had no
guard at all. OMN-18013 did not create the amplifier: it moved DLQ traffic off
the guarded ``NO_DISPATCHER`` leg onto this unguarded one.

These tests drive the real ``_make_event_bus_callback`` boundary with a real
``ModelDispatchResult``, because the defect is a missing branch at that seam and
an isolation test that called the handler directly would pass with the loop
running.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_event_bus_callback,
)

# The exact topic the live loop ran on, and the exact failure it carried.
_DLQ_EVENTS_TOPIC = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: verbatim from the live amplification trace
_DLQ_COMMANDS_TOPIC = "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: verbatim from the live amplification trace
_ORDINARY_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"  # onex-topic-allow: a non-DLQ control topic
_CORRELATION = "8a4ba739-3000-4000-8000-000000000001"
_LIVE_FAILURE = "Consumer not started"


class _RecordingDlqBus:
    """Minimal duck-typed bus exposing the boundary's DLQ contract."""

    def __init__(self, *, persisted: bool = True) -> None:
        self.calls: list[dict[str, object]] = []
        self._persisted = persisted

    async def _publish_raw_to_dlq(self, **kwargs: object) -> bool:
        self.calls.append(kwargs)
        return self._persisted


def _handler_error_result(topic: str) -> ModelDispatchResult:
    """The live shape: HANDLER_ERROR with no terminal output of any kind.

    This is precisely what ``_raise_if_silent_dispatch_failure`` converts into a
    ``HandlerDispatchFailureError``, which the callback's catch-all then routes
    to ``_route_swallowed_exception``.
    """
    now = datetime.now(UTC)
    return ModelDispatchResult(
        status=EnumDispatchStatus.HANDLER_ERROR,
        topic=topic,
        started_at=now,
        completed_at=now,
        output_count=0,
        output_events=[],
        correlation_id=UUID(_CORRELATION),
        error_message=f"RuntimeError: {_LIVE_FAILURE}",
        error_details={"failure_class": "RuntimeError"},
        dispatcher_id="dispatcher.auto.node_dlq_replay_effect.HandlerDlqReplay.replay_dlq_8a4ba739",
    )


def _boundary_message(topic: str) -> ModelEventMessage:
    envelope = ModelEventEnvelope[object](
        payload={"dlq_topic": topic},
        correlation_id=UUID(_CORRELATION),
        event_type="omnibase-infra.dlq-replay",
    )
    return ModelEventMessage(
        topic=topic,
        key=None,
        value=envelope.model_dump_json().encode("utf-8"),
        headers=ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="omn-18084-test",
            event_type="omnibase-infra.dlq-replay",
            correlation_id=UUID(_CORRELATION),
        ),
    )


async def _drive_boundary(topic: str, dlq_bus: _RecordingDlqBus) -> None:
    engine = AsyncMock()
    engine.dispatch_scoped.return_value = _handler_error_result(topic)
    callback = _make_event_bus_callback(
        topic,
        cast("object", engine),  # type: ignore[arg-type]
        result_applier=None,
        event_bus=dlq_bus,
        allowed_dispatcher_ids=("dispatcher.auto.mirror",),
    )
    await callback(_boundary_message(topic))


@pytest.mark.unit
class TestHandlerErrorOnADeadLetterTopicIsNotRepublished:
    """AC1 — the ``HANDLER_ERROR`` leg gets the loop-breaker the others have."""

    @pytest.mark.asyncio
    async def test_events_dlq_handler_failure_writes_nothing_back_to_its_own_topic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RED at the parent commit: one DLQ write per failure, forever.

        At ``a4393794`` this asserts ``dlq_bus.calls == []`` and gets a single
        call whose ``dlq_topic`` is ``onex.dlq.omnibase-infra.events.v1`` -- the
        same topic the record was consumed from. That one write, at 193.8/s, is
        the 151 GB/day.
        """
        monkeypatch.setenv("ONEX_BOUNDARY_DLQ_ENABLED", "true")

        dlq_bus = _RecordingDlqBus()
        await _drive_boundary(_DLQ_EVENTS_TOPIC, dlq_bus)

        assert dlq_bus.calls == [], (
            "a handler failure on a record consumed FROM "
            f"{_DLQ_EVENTS_TOPIC} was re-dead-lettered; "
            f"sinks written: {[c.get('dlq_topic') for c in dlq_bus.calls]}"
        )

    @pytest.mark.asyncio
    async def test_the_republished_record_would_name_its_own_source_topic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fixed point stated as an assertion, not as prose.

        Whatever the boundary writes for a DLQ-source failure, it must not be
        the source topic itself. At the parent commit the single recorded call
        has ``original_topic == dlq_topic == onex.dlq.omnibase-infra.events.v1``.
        """
        monkeypatch.setenv("ONEX_BOUNDARY_DLQ_ENABLED", "true")

        dlq_bus = _RecordingDlqBus()
        await _drive_boundary(_DLQ_EVENTS_TOPIC, dlq_bus)

        self_referential = [
            call
            for call in dlq_bus.calls
            if call.get("dlq_topic") == call.get("original_topic")
        ]
        assert self_referential == [], (
            "the boundary wrote a record back onto the topic it read it from — "
            "a fixed point that amplifies 1:1 with no bound"
        )

    @pytest.mark.asyncio
    async def test_the_commands_dlq_leg_is_guarded_on_the_same_terms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The commands DLQ looped too, three orders of magnitude smaller.

        267 of 300 sampled records on ``onex.dlq.omnibase-infra.commands.v1``
        carried the same self-referential ``original_topic`` and the same
        dispatcher. The guard must key on the DLQ shape, not on one topic name.
        """
        monkeypatch.setenv("ONEX_BOUNDARY_DLQ_ENABLED", "true")

        dlq_bus = _RecordingDlqBus()
        await _drive_boundary(_DLQ_COMMANDS_TOPIC, dlq_bus)

        assert dlq_bus.calls == [], (
            "a handler failure on a record consumed FROM "
            f"{_DLQ_COMMANDS_TOPIC} was re-dead-lettered"
        )

    @pytest.mark.asyncio
    async def test_an_ordinary_topic_still_reaches_the_dlq(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control — the guard must not disarm the DLQ generally.

        Without this, a guard that returned early for EVERY topic would satisfy
        the three assertions above while silently deleting the boundary's whole
        reason to exist. This test passes at the parent commit and must keep
        passing at the fix.
        """
        monkeypatch.setenv("ONEX_BOUNDARY_DLQ_ENABLED", "true")

        dlq_bus = _RecordingDlqBus()
        await _drive_boundary(_ORDINARY_TOPIC, dlq_bus)

        assert len(dlq_bus.calls) == 1, (
            "a handler failure on an ordinary topic must still be dead-lettered"
        )
        assert dlq_bus.calls[0]["original_topic"] == _ORDINARY_TOPIC
        assert dlq_bus.calls[0]["dlq_topic"] != _ORDINARY_TOPIC
