# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18084 — ``get_dlq_topic_for_original`` must refuse a DLQ topic.

THE FUNCTION IS A FIXED POINT ON ITS OWN OUTPUT. It resolves a message category
from the topic's own segments, so ``onex.dlq.omnibase-infra.events.v1`` contains
the segment ``events``, classifies as EVENT, and rebuilds the identical name.
``f(x) == x`` for every DLQ topic. Any caller that hands it a topic it just
consumed from — and the auto-wiring boundary did exactly that on the
``HANDLER_ERROR`` leg — gets back the instruction to write the record onto the
topic it came from.

A per-call-site guard closes one hole. Making the resolver itself refuse closes
the CLASS: a future call site cannot rebuild the fixed point by accident, and
the refusal is typed rather than a silently-plausible string.

The two boundary helpers exercised at the bottom are the other two places on
this seam that resolve a sink from a consumed topic without owning the
``HANDLER_ERROR`` leg's guard. Both are documented as never-raising, so a bare
typed refusal reaching them would convert a logged loss into a crash — they must
pre-guard, not catch.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from omnibase_infra.errors import DlqTopicFixedPointError
from omnibase_infra.event_bus.topic_constants import (
    derive_dlq_topic_for_event_type,
    get_dlq_topic_for_original,
    is_dlq_topic,
)

# Every DLQ sink the .201 dev lane actually carries.
_DLQ_TOPICS = (
    "onex.dlq.omnibase-infra.events.v1",  # onex-topic-allow: verbatim from the live amplification trace
    "onex.dlq.omnibase-infra.commands.v1",  # onex-topic-allow: verbatim from the live amplification trace
    "onex.dlq.omnibase-infra.intents.v1",  # onex-topic-allow: the third topic the replay contract subscribes
    "onex.dlq.omnibase-infra.quarantine.v1",  # onex-topic-allow: the platform quarantine sink
)
_ORDINARY_TOPIC = (
    "onex.evt.platform.node-registered.v1"  # onex-topic-allow: a non-DLQ control topic
)


class _RecordingDlqBus:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def _publish_raw_to_dlq(self, **kwargs: object) -> bool:
        self.calls.append(kwargs)
        return True


@pytest.mark.unit
class TestDlqTopicIsRefusedAsAnOriginalTopic:
    """AC2 — the resolver refuses to rebuild its own fixed point."""

    @pytest.mark.parametrize("topic", _DLQ_TOPICS)
    def test_resolving_a_dlq_topic_raises_instead_of_returning_it(
        self, topic: str
    ) -> None:
        """RED at the parent commit: returns ``topic`` unchanged.

        At ``a4393794`` this call returns the input string, which is why every
        one of 2,000 sampled records on the events DLQ carried
        ``original_topic`` equal to its own topic.
        """
        with pytest.raises(DlqTopicFixedPointError):
            get_dlq_topic_for_original(topic)

    @pytest.mark.parametrize("topic", _DLQ_TOPICS)
    def test_the_refusal_names_the_topic_it_refused(self, topic: str) -> None:
        """A refusal nobody can attribute is only marginally better than a loop."""
        with pytest.raises(DlqTopicFixedPointError) as excinfo:
            get_dlq_topic_for_original(topic)
        assert topic in str(excinfo.value)

    def test_an_ordinary_topic_still_resolves(self) -> None:
        """Positive control — a refusal that refused everything would pass above."""
        assert is_dlq_topic(_ORDINARY_TOPIC) is False
        assert (
            get_dlq_topic_for_original(_ORDINARY_TOPIC)
            == "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: the resolver's own documented output
        )

    def test_the_legacy_event_type_path_refuses_on_the_same_terms(self) -> None:
        """``derive_dlq_topic_for_event_type`` delegates here when event_type is absent.

        Leaving that path unrefused would keep one laundering route open into
        the same fixed point.
        """
        with pytest.raises(DlqTopicFixedPointError):
            derive_dlq_topic_for_event_type(None, _DLQ_TOPICS[0])


@pytest.mark.unit
class TestBoundaryHelpersPreGuardRatherThanCrash:
    """AC2 — the never-raising boundary helpers take the guard, not the raise."""

    @pytest.mark.asyncio
    async def test_apply_publish_failure_on_a_dlq_topic_neither_publishes_nor_raises(
        self,
    ) -> None:
        """The record is already on a dead-letter sink: durably captured.

        Raising here would withhold the offset and redeliver the same record
        forever; republishing would rebuild the loop. Log and return.
        """
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _route_apply_publish_failure,
        )

        bus = _RecordingDlqBus()
        await _route_apply_publish_failure(
            RuntimeError("Consumer not started"),
            event_bus=bus,
            topic=_DLQ_TOPICS[0],
            message=SimpleNamespace(value=b"{}", key=None, offset=1, partition=0),
            correlation_id=uuid4(),
        )
        assert bus.calls == [], (
            "an apply-publish failure on a dead-letter source topic was "
            "re-dead-lettered onto that same topic"
        )

    @pytest.mark.asyncio
    async def test_sync_publisher_failure_on_a_dlq_topic_neither_publishes_nor_raises(
        self,
    ) -> None:
        """The quarantine DLQ is a declared publish topic of the replay contract.

        So this helper genuinely can be handed a DLQ topic, and its docstring
        promises it never raises.
        """
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _route_sync_publisher_failure,
        )

        bus = _RecordingDlqBus()
        await _route_sync_publisher_failure(
            RuntimeError("Consumer not started"),
            event_bus=bus,
            handler_name="HandlerDlqReplay",
            topic=_DLQ_TOPICS[3],
            payload=b"{}",
        )
        assert bus.calls == [], (
            "a sync-publisher failure targeting the quarantine dead-letter sink "
            "was re-dead-lettered onto that same sink"
        )
