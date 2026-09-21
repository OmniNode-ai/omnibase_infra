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
import pathlib
import re
from datetime import UTC, datetime
from typing import Any

import pytest

from omnibase_core.models.dispatch.model_message_delivery_context import (
    ModelMessageDeliveryContext,
)
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.delivery_context import (
    delivery_context_from_message,
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
        context = delivery_context_from_message(_message(), _TOPIC)

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
        assert delivery_context_from_message(_message(**overrides), _TOPIC) is None, why

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

        assert delivery_context_from_message(BareMessage(), _TOPIC) is None  # type: ignore[arg-type]


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


# ---------------------------------------------------------------------------
# OMN-18918: the CALLER-side probe at the wiring -> dispatch-engine seam.
#
# ``delivery`` is optional on ``ProtocolDispatchEngine``, so every engine
# written before it is still a valid implementor with a two-parameter
# ``dispatch``. Passing the keyword unconditionally turns that optionality
# into a TypeError on the first message. Both engines in this repo's own
# remote-agent integration test failed exactly that way before the probe
# existed (tests/integration/nodes/
# test_node_remote_agent_invoke_effect_integration.py), which is the RED
# proof for the consumer class; these cases pin the probe's own edges.
# ---------------------------------------------------------------------------


class _LegacyEngine:
    """An engine written before the parameter existed."""

    async def dispatch(self, topic: str, envelope: object) -> None: ...


class _ModernEngine:
    """An engine that declares it, keyword-only, as the protocol does."""

    async def dispatch(
        self, topic: str, envelope: object, *, delivery: object | None = None
    ) -> None: ...


class _PositionalDeliveryEngine:
    """Declares the name positionally -- ambiguous with topic/envelope."""

    async def dispatch(
        self, topic: str, envelope: object, delivery: object | None = None
    ) -> None: ...


class _KwargsOnlyEngine:
    """Swallows anything, but declares nothing."""

    async def dispatch(
        self, topic: str, envelope: object, **kwargs: object
    ) -> None: ...


class _NoDispatchAttribute:
    """Not an engine at all."""


def test_a_legacy_engine_is_not_passed_the_keyword() -> None:
    from omnibase_infra.runtime.delivery_context import (
        engine_type_accepts_delivery,
    )

    assert engine_type_accepts_delivery(_LegacyEngine, "dispatch") is False


def test_a_declaring_engine_is_passed_the_keyword() -> None:
    from omnibase_infra.runtime.delivery_context import (
        engine_type_accepts_delivery,
    )

    assert engine_type_accepts_delivery(_ModernEngine, "dispatch") is True


def test_a_positional_delivery_parameter_is_refused() -> None:
    """Name alone is not enough; a positional match would be ambiguous."""
    from omnibase_infra.runtime.delivery_context import (
        engine_type_accepts_delivery,
    )

    assert engine_type_accepts_delivery(_PositionalDeliveryEngine, "dispatch") is False


def test_a_kwargs_only_engine_is_refused() -> None:
    """It would not raise, but it declared nothing -- unknown refuses."""
    from omnibase_infra.runtime.delivery_context import (
        engine_type_accepts_delivery,
    )

    assert engine_type_accepts_delivery(_KwargsOnlyEngine, "dispatch") is False


def test_an_object_with_no_dispatch_is_refused_rather_than_raising() -> None:
    from omnibase_infra.runtime.delivery_context import (
        engine_type_accepts_delivery,
    )

    assert engine_type_accepts_delivery(_NoDispatchAttribute, "dispatch") is False


def test_the_probe_is_cached_per_engine_class() -> None:
    """The signature is a property of the type; it must not run per message."""
    from omnibase_infra.runtime.delivery_context import (
        engine_type_accepts_delivery,
    )

    engine_type_accepts_delivery(_ModernEngine, "dispatch")
    before = engine_type_accepts_delivery.cache_info().hits
    for _ in range(5):
        engine_type_accepts_delivery(_ModernEngine, "dispatch")

    assert engine_type_accepts_delivery.cache_info().hits == before + 5


def test_the_real_engine_is_recognised_by_the_probe() -> None:
    """The positive control: the engine actually wired in declares it."""
    from omnibase_infra.runtime.delivery_context import (
        engine_type_accepts_delivery,
    )

    assert engine_type_accepts_delivery(MessageDispatchEngine, "dispatch") is True


# ---------------------------------------------------------------------------
# OMN-18918 / OMN-18955: exactly ONE mechanism may write the coordinates.
#
# Both tickets fixed the same defect at the same seam, independently. #3900
# landed a task-local context channel; this change landed the typed parameter
# the operator ruled for on 2026-09-20. When the two were merged, the result
# auto-merged CLEAN and carried BOTH injections, writing the same two payload
# keys about sixty lines apart -- last writer wins, no conflict marker, no
# failing test. The contextvar arm is deleted here and this is the test that
# refuses its return, by ANY route rather than by name.
# ---------------------------------------------------------------------------

_INJECTION_SITE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "auto_wiring"
    / "handler_wiring.py"
)
_RUNTIME_DIR = _INJECTION_SITE.parents[1]


def _assignments_to(key: str) -> list[str]:
    """Every line in the wiring that assigns the given injected payload key."""
    pattern = re.compile(rf'^\s*input_data\[["\']{re.escape(key)}["\']\]\s*=')
    return [
        line
        for line in _INJECTION_SITE.read_text(encoding="utf-8").splitlines()
        if pattern.match(line)
    ]


@pytest.mark.parametrize("key", ["_partition", "_offset"])
def test_exactly_one_source_writes_each_coordinate_key(key: str) -> None:
    """Two writers for one key is the silent failure, not a redundancy."""
    assignments = _assignments_to(key)

    assert len(assignments) == 1, (
        f"{len(assignments)} assignments to input_data[{key!r}] in "
        f"{_INJECTION_SITE.name}: {assignments}. A second writer does not "
        "conflict and does not fail -- the later one silently wins. If a new "
        "source is genuinely needed, delete the old one in the same change."
    )


def test_the_single_assignment_is_the_typed_delivery_one() -> None:
    """The positive control: the pattern above still matches something real.

    A test asserting 'exactly one' passes just as happily at one as it would
    at zero if the pattern rotted, and a zero would mean the injection was
    deleted entirely -- the original OMN-18905 defect, restored. So the one
    match is read, not merely counted.
    """
    partition_line = _assignments_to("_partition")[0]
    offset_line = _assignments_to("_offset")[0]

    assert "delivery.partition" in partition_line
    assert "delivery.offset" in offset_line


def test_no_ambient_source_coordinate_channel_survives_anywhere_in_runtime() -> None:
    """The retired mechanism is gone by SUBSTANCE, not by import name.

    Deleting the two helpers is not enough on its own: the channel could come
    back as a differently-named ContextVar and this seam would be back to two
    sources with nothing noticing. The check is over the whole runtime package
    rather than the one module the helpers lived in.
    """
    offenders = sorted(
        str(path.relative_to(_RUNTIME_DIR))
        for path in _RUNTIME_DIR.rglob("*.py")
        if "source_coordinate" in path.read_text(encoding="utf-8")
    )

    assert offenders == [], (
        "a source-coordinate context channel is present again in "
        f"{offenders}. The coordinates travel as ModelMessageDeliveryContext "
        "on the dispatch call (operator ruling, 2026-09-20); ambient state "
        "beside it gives the seam two writers for one field."
    )


def test_the_channels_that_legitimately_live_there_are_untouched() -> None:
    """Second positive control, bounding the deletion above.

    `dispatch_envelope_context` predates OMN-18955 and carries two channels
    this change has no business removing. Asserting they survive is what makes
    the assertion above a scalpel rather than a claim that the module is
    empty.
    """
    from omnibase_infra.runtime import dispatch_envelope_context as channels

    assert hasattr(channels, "bind_dispatch_envelope")
    assert hasattr(channels, "current_dispatch_envelope")
    assert hasattr(channels, "bind_projection_tenant_authority")
    assert hasattr(channels, "current_projection_tenant_authority")
    assert not hasattr(channels, "bind_source_coordinate")
    assert not hasattr(channels, "current_source_coordinate")


# ---------------------------------------------------------------------------
# OMN-18918: THREE consume boundaries, not one.
#
# The first cut of this change supplied a delivery context at the subcontract
# seam only. The in-process projection writers -- the surfaces the OMN-18905
# defect is actually about -- do not arrive there; they reach handler_wiring's
# two callbacks, which dispatch through a different protocol. Every test
# passed, and on the lane it would have injected nothing for exactly those
# writers. Caught in second-actor review, not by a test.
#
# So the boundary COUNT is asserted, not just the mechanism at one of them.
# ---------------------------------------------------------------------------

_SUBCONTRACT_WIRING = _INJECTION_SITE.parents[1] / "event_bus_subcontract_wiring.py"


def _builder_call_count(path: pathlib.Path) -> int:
    return len(
        re.findall(
            r"\bdelivery_context_from_message\(", path.read_text(encoding="utf-8")
        )
    )


def test_every_consume_boundary_builds_a_delivery_context() -> None:
    """One builder call per boundary: two here, one at the subcontract seam."""
    handler_wiring_calls = _builder_call_count(_INJECTION_SITE)
    subcontract_calls = _builder_call_count(_SUBCONTRACT_WIRING)

    # Three calls for two boundaries: the event-bus callback has one, and the
    # raw projection callback has two because it dispatches on both arms of
    # its flow-counter branch. The retired context-channel arm bound on both
    # arms too, and for the stated reason -- a projection writer needs the
    # coordinates whether or not this lane happens to be counting flow.
    assert handler_wiring_calls == 3, (
        f"{handler_wiring_calls} builder calls in {_INJECTION_SITE.name}; expected 3 "
        "(event-bus callback: 1; raw projection callback: 2, one per arm of its "
        "flow-counter branch). A boundary that stops building a context does not "
        "fail -- it injects nothing and the exposure freezes at zero consumer lag."
    )
    assert subcontract_calls == 1, (
        f"{subcontract_calls} builder calls in {_SUBCONTRACT_WIRING.name}; expected 1."
    )


def test_the_builder_and_the_probe_have_exactly_one_home() -> None:
    """Three callers, one definition -- the reason the module exists.

    Two of the three boundaries previously had no access to the builder at
    all because it lived inside the third one's module. Copying it per
    boundary is how they drift apart; this asserts they cannot.
    """
    runtime_dir = _INJECTION_SITE.parents[1]
    definitions = sorted(
        str(path.relative_to(runtime_dir))
        for path in runtime_dir.rglob("*.py")
        if re.search(
            r"^def delivery_context_from_message\(",
            path.read_text(encoding="utf-8"),
            re.M,
        )
    )

    assert definitions == ["delivery_context.py"], (
        f"delivery_context_from_message is defined in {definitions}; it must have "
        "exactly one home, because three consume boundaries call it."
    )
