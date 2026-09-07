# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17862: the offset may advance only on a CONFIRMED quarantine publish.

## The defect

``_route_projection_error_to_dlq`` is declared ``-> bool`` and returns ``False``
on three separate failures -- no publishable event bus bound, the bound bus's
``publish`` attribute not callable, and the publish itself raising -- each logged
at ERROR. Its own docstring calls the whole function "best-effort ... a DLQ
publish failure never propagates, so it cannot wedge the consumer".

**Its one call site discarded that boolean.** Inside the
``if _is_projection_content_failure(exc):`` arm of
``_make_projection_dispatch_callback._callback``, the call was a bare ``await``
expression statement -- nothing assigned, nothing tested. ``write_path_failure``
stayed ``None``, the guard did not fire, the callback returned normally, and the
consume boundary read that as success. So on a broker refusal, a wedged
connection, or a lane brought up with no publishable bus, a refused record was
**neither projected nor quarantined and its offset advanced** -- the silent drop,
reached through the arm this design has been calling the safe one. The active-error
counter fired regardless, so the COUNTER moved while the RECORD was gone.

It also discarded the ``ModelPublishReceipt``. ``EventBus.publish``'s own
docstring states canonical invariant 7 -- *a publish return is not durability* --
and names the seam that makes the distinction mechanical.

## Why binding the boolean into ``write_path_failure`` would REINSTATE it

``write_path_failure = False`` satisfies the ``is not None`` guard, and the raise
below it then evaluates ``raise ProjectionNotMaterializedError(...) from False``,
which is a ``TypeError`` (*exception causes must derive from BaseException*). A
``TypeError`` is not a ``ProjectionNotMaterializedError``, so the offset-withholding
arm does not catch it; it falls to the bounded-retry loop's generic handler and
then to the boundary catch-all, which routes it to the swallowed-exception path
and **returns normally -- an ACK**. The same silent drop, one exception type
further along. ``test_failed_quarantine_raises_the_withholding_type`` is the fence:
it asserts the RAISED TYPE, so a literal boolean implementation goes red here.

## The positive control is not optional

"Withhold the offset unconditionally" satisfies every failure assertion in this
file on its own -- and that is the permanent partition stall R3 exists to end,
wearing this test as cover. ``test_confirmed_quarantine_still_advances_the_offset``
is what makes the pair meaningful.

## Blast radius

This is shared auto-wiring used by EVERY projection handler, not one node's code.
Every one of them acked on a failed DLQ publish before this change.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDispatchSinks,
    _make_projection_dispatch_callback,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)

_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"
_TEST_DSN = "postgresql://user:pass@host:5432/omnidash_analytics"
_TOPIC = "onex.evt.omniclaude.tool-executed.v1"


@pytest.fixture(autouse=True)
def _configured_projection_dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch, url=_TEST_DSN)


class _Strict(BaseModel):
    """Stands in for the refusal R3's own parse now raises."""

    emitted_at: datetime


class _RefusingHandler:
    """Raises the exact class the R3 refusal produces: a pydantic error."""

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        _Strict.model_validate({})
        raise AssertionError("unreachable -- validation must fail first")


def _receipt(topic: str) -> ModelPublishReceipt:
    """A real coordinate, as BOTH shipped buses return from ``publish``."""
    return ModelPublishReceipt(
        topic=topic,
        partition=0,
        offset=1,
        cluster="test-cluster",
        produced_at=datetime.now(UTC),
        transport=EnumInfraTransportType.INMEMORY,
    )


class _AcceptingBus:
    """Returns a coordinate, exactly as ``EventBusKafka``/``EventBusInmemory`` do."""

    def __init__(self) -> None:
        self.published: list[tuple[str, object, bytes]] = []

    async def publish(
        self, topic: str, key: object, value: bytes
    ) -> ModelPublishReceipt:
        self.published.append((topic, key, value))
        return _receipt(topic)


class _RaisingBus:
    """A broker refusal / wedged connection."""

    def __init__(self) -> None:
        self.published: list[tuple[str, object, bytes]] = []

    async def publish(
        self, topic: str, key: object, value: bytes
    ) -> ModelPublishReceipt:
        raise ConnectionError("broker unavailable")


class _CoordinateLessBus:
    """Accepts the produce and reports NO coordinate.

    A transport that cannot report a coordinate has told us nothing at all, and
    "nothing at all" is ``UNKNOWN``, which fails closed. Even the deliberately
    weakest shipped strategy refuses to confirm a ``None`` receipt.
    """

    def __init__(self) -> None:
        self.published: list[tuple[str, object, bytes]] = []

    async def publish(self, topic: str, key: object, value: bytes) -> None:
        self.published.append((topic, key, value))


class _NonCallablePublish:
    """The bound bus's ``publish`` attribute is not callable."""

    publish = "not-callable"


def _envelope() -> MagicMock:
    envelope = MagicMock()
    envelope.topic = _TOPIC
    envelope.payload = {"session_id": "s-omn-17862"}
    envelope.correlation_id = "omn-17862"
    return envelope


def _run(bus: object) -> object:
    callback = _make_projection_dispatch_callback(
        _RefusingHandler(),
        projection_database_target(
            "session_replay_snapshots", schema="omninode_internal"
        ),
        (_TOPIC,),
        sinks=ProjectionDispatchSinks(event_bus=bus),
    )
    with patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            return asyncio.run(callback(_envelope()))


# ---------------------------------------------------------------------------
# The positive control -- FIRST, because the failure assertions below are
# meaningless without it.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confirmed_quarantine_still_advances_the_offset() -> None:
    """A refused record whose quarantine IS confirmed acks, exactly as before.

    Redelivering identical refused bytes reproduces the identical refusal
    forever, so withholding here would wedge the partition on one poison record.
    The DLQ-and-advance contract is preserved; this change separates confirmed
    from unconfirmed, it does not make every refusal fatal. Without this
    assertion, "withhold unconditionally" passes every other test in the file.
    """
    bus = _AcceptingBus()

    assert _run(bus) is None, "a confirmed quarantine ACKs; the offset must advance"
    assert len(bus.published) == 1
    assert "dlq" in bus.published[0][0]


# ---------------------------------------------------------------------------
# The three arms that return False
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_failed_quarantine_publish_withholds_the_offset() -> None:
    """The publish raises: nothing is durable, so the offset must NOT advance.

    RED on ``origin/dev``, where this returns ``None`` -- an ACK -- with zero
    records published anywhere.
    """
    bus = _RaisingBus()

    with pytest.raises(ProjectionNotMaterializedError):
        _run(bus)

    assert bus.published == [], "nothing reached any topic"


@pytest.mark.unit
def test_absent_event_bus_withholds_the_offset() -> None:
    """A lane wired with no publishable bus is the volume case.

    This is the arm that produces the drop silently and at scale: no broker
    error, no log storm, just an ack per record.
    """
    with pytest.raises(ProjectionNotMaterializedError):
        _run(None)


@pytest.mark.unit
def test_non_callable_publish_withholds_the_offset() -> None:
    """A bound object whose ``publish`` is not callable publishes nothing."""
    with pytest.raises(ProjectionNotMaterializedError):
        _run(_NonCallablePublish())


@pytest.mark.unit
def test_coordinate_less_publish_withholds_the_offset() -> None:
    """Canonical invariant 7: a publish RETURN is not durability.

    The produce call did not raise, and that is not a durability claim. Both
    shipped buses return a ``ModelPublishReceipt``; a transport that returns no
    coordinate cannot support any durable claim, so it fails closed.
    """
    bus = _CoordinateLessBus()

    with pytest.raises(ProjectionNotMaterializedError):
        _run(bus)

    assert len(bus.published) == 1, "the produce was attempted"


# ---------------------------------------------------------------------------
# The type fence -- this is what a literal boolean implementation fails
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_failed_quarantine_raises_the_withholding_type() -> None:
    """The raised type must be ``ProjectionNotMaterializedError``, not ``TypeError``.

    ``write_path_failure = False`` passes the ``is not None`` guard and turns the
    raise into ``raise ... from False`` -- a ``TypeError``, which the
    offset-withholding arm does not catch and the boundary catch-all ACKs. This
    assertion is the only thing separating the two implementations from the
    outside.
    """
    with pytest.raises(ProjectionNotMaterializedError) as raised:
        _run(_RaisingBus())

    assert not isinstance(raised.value, TypeError)
    assert "quarantine" in str(raised.value).lower(), (
        "the message must name the QUARANTINE failure that withheld the offset, "
        "not only the parse failure that triggered it -- otherwise the log line "
        "points an operator at the wrong seam"
    )


@pytest.mark.unit
def test_the_parse_failure_is_chained_under_the_quarantine_failure() -> None:
    """Both facts are recoverable: what was refused, and why it was not durable."""
    with pytest.raises(ProjectionNotMaterializedError) as raised:
        _run(_RaisingBus())

    chain: list[BaseException] = []
    exc: BaseException | None = raised.value
    while exc is not None:
        chain.append(exc)
        exc = exc.__cause__

    assert any(isinstance(item, ValidationError) for item in chain), (
        f"the original refusal must stay reachable through __cause__; got {chain!r}"
    )


# ---------------------------------------------------------------------------
# The classifier is NOT widened -- the prohibition, asserted
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_content_failure_allowlist_is_unchanged() -> None:
    """This repair needs no change to the keep-or-ack allowlist, and makes none.

    Widening ``_is_projection_content_failure`` converts *withhold* into
    *DLQ-and-ack*, which is the acknowledgement class that cost
    ``pr_merged_events`` 230 merged PRs. The refusal R3 adds arrives as a
    ``ValidationError`` the allowlist ALREADY accepts, so nothing there moves.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _is_projection_content_failure,
    )

    class _UniqueViolationError(Exception):
        """Stands in for ``psycopg2.errors.UniqueViolation``."""

    assert _is_projection_content_failure(_UniqueViolationError("dup key")) is False
    assert _is_projection_content_failure(ValueError("refused")) is False

    class _RefusalValueError(ValueError):
        pass

    assert _is_projection_content_failure(_RefusalValueError("refused")) is False
    try:
        _Strict.model_validate({})
    except ValidationError as exc:
        assert _is_projection_content_failure(exc) is True
