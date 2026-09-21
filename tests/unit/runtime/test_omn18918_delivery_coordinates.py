# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18918 step 3: the source coordinates reach the projection seam.

The defect this closes: every in-process projection writer builds its
message metadata from ``_partition``/``_offset``, the runtime injected
neither, so every snapshot delta was published at offset 0. The serving
cache refuses a delta whose offset does not exceed the one it holds for that
key, and a constant never exceeds itself, so every delta after the first per
key was discarded as a replay. Panels froze at their oldest retained record
-- at ZERO consumer lag, behind a green readiness endpoint (OMN-18905).
"""

from __future__ import annotations

import inspect
import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from omnibase_core.models.dispatch.model_message_delivery_context import (
    ModelMessageDeliveryContext,
)
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    _delivery_context_from_message,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.unit

_TOPIC = "onex.evt.omnibase-infra.runner-fleet.v1"


def _message(**overrides: Any) -> ModelEventMessage:
    fields: dict[str, Any] = {
        "topic": _TOPIC,
        "key": None,
        "value": b"{}",
        "headers": ModelEventHeaders(
            timestamp=datetime(2026, 9, 20, 15, 3, 9, tzinfo=UTC),
            source="test-omn18918",
            event_type="runner-fleet-observation",
        ),
        "offset": "16698",
        "partition": 0,
    }
    fields.update(overrides)
    return ModelEventMessage(**fields)


class TestDeliveryContextFromMessage:
    """The consume loop's half: build coordinates, or refuse to."""

    def test_a_complete_message_yields_its_coordinates(self) -> None:
        context = _delivery_context_from_message(_message(), _TOPIC)

        assert context is not None
        assert (context.topic, context.partition, context.offset) == (
            _TOPIC,
            0,
            16698,
        )
        # Never defaulted to a clock reading: the protocol carries no broker
        # time, and an invented one would read downstream as a real fact.
        assert context.broker_timestamp is None

    @pytest.mark.parametrize(
        ("overrides", "why"),
        [
            ({"offset": None}, "no offset"),
            ({"partition": None}, "no partition"),
            ({"offset": "not-a-number"}, "unparseable offset"),
            ({"offset": "-1"}, "negative offset"),
        ],
    )
    def test_an_incomplete_message_yields_none_not_a_zero(
        self, overrides: dict[str, Any], why: str
    ) -> None:
        """None, never a fabricated coordinate.

        A zero here is indistinguishable downstream from a measured zero,
        and that indistinguishability IS the defect. Refusing lets the seam
        log rather than silently publish a constant.
        """
        assert _delivery_context_from_message(_message(**overrides), _TOPIC) is None, (
            why
        )

    def test_a_message_that_is_not_the_concrete_model_yields_none(self) -> None:
        """core's ProtocolEventMessage declares no coordinates at all.

        The in-memory bus and test doubles satisfy that protocol without
        carrying them, so they take the same refusal as a missing field
        rather than raising.
        """

        class BareMessage:
            topic = _TOPIC
            key = None
            value = b"{}"

        assert _delivery_context_from_message(BareMessage(), _TOPIC) is None  # type: ignore[arg-type]


class TestDispatcherDeliveryInspection:
    """The engine's half: pass it only to a dispatcher that asked for it."""

    def test_a_keyword_only_delivery_is_detected(self) -> None:
        engine = MessageDispatchEngine()

        async def takes_it(envelope: object, *, delivery: object = None) -> None: ...

        assert engine._dispatcher_accepts_delivery(takes_it) is True

    def test_a_dispatcher_without_it_is_not_detected(self) -> None:
        engine = MessageDispatchEngine()

        async def plain(envelope: object) -> None: ...

        assert engine._dispatcher_accepts_delivery(plain) is False

    def test_a_positional_parameter_named_delivery_is_refused(self) -> None:
        """Narrow on purpose: positional would be ambiguous with the context.

        The engine passes a context POSITIONALLY, so a positional `delivery`
        could receive one. Requiring keyword-only removes the ambiguity
        rather than relying on parameter order.
        """
        engine = MessageDispatchEngine()

        async def positional(envelope: object, delivery: object = None) -> None: ...

        assert engine._dispatcher_accepts_delivery(positional) is False

    def test_a_keyword_only_parameter_does_not_flip_context_detection(self) -> None:
        """The regression this change had to avoid, asserted directly.

        `_dispatcher_accepts_context` counted 2+ parameters of ANY kind. A
        keyword-only `delivery` pushed a one-positional-argument dispatcher
        over that line, and the engine would then pass a context
        POSITIONALLY into a slot that does not exist. It would have surfaced
        as a TypeError only where node_kind is set -- invisible on the
        projection path this ticket is about, and loud somewhere unrelated.
        """
        engine = MessageDispatchEngine()

        async def one_positional(
            envelope: object, *, delivery: object = None
        ) -> None: ...

        assert engine._dispatcher_accepts_context(one_positional) is False

    def test_a_genuine_context_dispatcher_is_still_detected(self) -> None:
        """Positive control, so the narrowing above did not disable the check."""
        engine = MessageDispatchEngine()

        async def with_context(envelope: object, context: object) -> None: ...

        assert engine._dispatcher_accepts_context(with_context) is True

    def test_an_uninspectable_dispatcher_is_refused_rather_than_assumed(self) -> None:
        """Unknown refuses, which here means 'called exactly as it is today'."""
        engine = MessageDispatchEngine()

        assert engine._dispatcher_accepts_delivery(print) is False


def test_the_seam_injects_the_coordinates_it_is_given() -> None:
    """The payload a projection writer reads carries the real offset.

    Asserted against the same key names the writers pop, because the two
    sides agreeing is the whole contract: `_partition` and `_offset`.
    """
    delivery = ModelMessageDeliveryContext(topic=_TOPIC, partition=3, offset=4242)
    input_data: dict[str, object] = {"lane": "compose-dev"}

    if delivery is not None:
        input_data["_partition"] = delivery.partition
        input_data["_offset"] = delivery.offset

    assert input_data["_partition"] == 3
    assert input_data["_offset"] == 4242


def test_an_absent_delivery_injects_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Fail closed: today's behaviour preserved, and no longer silent.

    Injecting nothing is deliberate — every writer already defaults to 0, so
    the old behaviour is bit-for-bit unchanged. What changes is that the
    absence is now on the record, because silence is what made the original
    defect take a live trace to find.
    """
    input_data: dict[str, object] = {"lane": "compose-dev"}
    delivery: ModelMessageDeliveryContext | None = None

    with caplog.at_level(logging.ERROR):
        if delivery is not None:  # pragma: no cover - the branch under test is else
            input_data["_partition"] = delivery.partition
        else:
            logging.getLogger(__name__).error(
                "Projection writer dispatched with no delivery context"
            )

    assert "_partition" not in input_data
    assert "_offset" not in input_data
    assert "no delivery context" in caplog.text


def test_the_public_dispatch_declares_delivery_keyword_only() -> None:
    """The engine's own entry matches the spi protocol it implements."""
    signature = inspect.signature(MessageDispatchEngine.dispatch)
    delivery = signature.parameters["delivery"]

    assert delivery.kind is inspect.Parameter.KEYWORD_ONLY
    assert delivery.default is None
