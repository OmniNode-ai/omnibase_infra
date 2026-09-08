# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17862: the quarantine-confirmation boundary, against a REAL event bus.

The unit half of this repair
(``tests/unit/runtime/auto_wiring/test_omn17862_quarantine_publish_confirmed.py``)
drives the real projection dispatch callback against hand-written bus doubles.
This file drives the same callback against the **real** ``EventBusInmemory`` --
started, published to, and read back out of its own history -- so the claim that
a refused record becomes durable before its offset advances rests on a bus that
actually implements the publish contract, not on a double that was taught to.

That distinction is the whole point of the change. ``_route_projection_error_to_dlq``
used to discard both the ``ModelPublishReceipt`` and its own return value, so the
seam could not tell "the produce call did not raise" from "the record is at a
coordinate someone can read back". A double returning ``None`` is exactly the
shape that made the old code look correct; a real bus returning a real coordinate
is what makes the new code's confirmation meaningful.

Two facts are proven here that a mocked bus cannot establish:

* the receipt the DLQ path confirms is one a real transport minted, with a real
  ``(topic, partition, offset)``; and
* the quarantined record is genuinely retrievable from that coordinate afterwards
  -- durability demonstrated, not asserted.

Paired, as always, with the failure direction: a bus whose ``publish`` raises
must withhold the offset. Neither direction alone is evidence -- "withhold
unconditionally" passes the failure test and is the permanent partition stall
this repair exists to end.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
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
_QUARANTINE = "onex.dlq.omnibase-infra.quarantine.v1"


@pytest.fixture(autouse=True)
def _configured_projection_dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch, url=_TEST_DSN)


class _Strict(BaseModel):
    """Stands in for the refusal the session-replay parse now raises."""

    emitted_at: str


class _RefusingHandler:
    """Raises the exact class the refusal produces: a pydantic ValidationError."""

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        _Strict.model_validate({})
        raise AssertionError("unreachable -- validation must fail first")


def _envelope() -> MagicMock:
    envelope = MagicMock()
    envelope.topic = _TOPIC
    envelope.payload = {"session_id": "s-omn-17862-integration"}
    envelope.correlation_id = "omn-17862-integration"
    return envelope


def _callback(bus: object) -> object:
    return _make_projection_dispatch_callback(
        _RefusingHandler(),
        projection_database_target(
            "session_replay_snapshots", schema="omninode_internal"
        ),
        (_TOPIC,),
        sinks=ProjectionDispatchSinks(event_bus=bus),
    )


@pytest.mark.integration
def test_a_real_bus_confirms_the_quarantine_and_the_offset_advances() -> None:
    """THE POSITIVE CONTROL, against a started ``EventBusInmemory``.

    A refused record whose quarantine publication is confirmed still acks, and
    the record is retrievable from the coordinate the bus reported. Redelivering
    identical refused bytes reproduces the identical refusal forever, so
    withholding here would wedge the partition on one poison record -- the
    DLQ-and-advance contract is deliberately preserved.
    """

    async def _run() -> tuple[object, ModelPublishReceipt | None]:
        bus = EventBusInmemory(environment="test")
        await bus.start()
        try:
            receipts: list[ModelPublishReceipt] = []
            real_publish = bus.publish

            async def _recording_publish(
                topic: str, key: bytes | None, value: bytes, headers: object = None
            ) -> ModelPublishReceipt:
                receipt = await real_publish(topic, key, value)
                receipts.append(receipt)
                return receipt

            # Wrapping records the coordinate the REAL bus minted; the publish
            # itself is the real one, not a substitute.
            bus.publish = _recording_publish  # type: ignore[method-assign]

            with patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN):
                with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
                    outcome = await _callback(bus)(_envelope())
            return outcome, (receipts[0] if receipts else None)
        finally:
            await bus.close()

    outcome, receipt = asyncio.run(_run())

    assert outcome is None, "a confirmed quarantine ACKs; the offset must advance"
    assert receipt is not None, "the real bus must report a durability coordinate"
    assert receipt.topic == _QUARANTINE
    assert receipt.partition >= 0
    assert receipt.offset >= 0
    assert receipt.cluster, (
        "a coordinate without a cluster identity is not a coordinate"
    )


@pytest.mark.integration
def test_the_quarantined_record_is_actually_retrievable_at_that_coordinate() -> None:
    """Durability demonstrated rather than asserted.

    The old code's failure was precisely that a publish RETURN was being read as
    durability. Reading the record back out of the bus's own history closes that
    gap for this seam: the envelope is on the sink, and it carries the failure
    reason and correlation id an operator would need to reclassify it by hand.
    """

    async def _run() -> list[bytes]:
        bus = EventBusInmemory(environment="test")
        await bus.start()
        try:
            with patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN):
                with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
                    await _callback(bus)(_envelope())
            history = await bus.get_event_history(topic=_QUARANTINE)
            return [message.value for message in history]
        finally:
            await bus.close()

    values = asyncio.run(_run())

    assert len(values) == 1, f"expected one quarantined record, got {len(values)}"
    envelope = json.loads(values[0].decode("utf-8"))
    assert envelope["correlation_id"] == "omn-17862-integration"
    assert "ValidationError" in envelope["failure_reason"]
    assert envelope["quarantine_fallback"] is True


@pytest.mark.integration
def test_a_broker_refusal_withholds_the_offset_on_the_same_wiring() -> None:
    """THE FAILURE DIRECTION, on the same real-bus fixture.

    A started bus whose ``publish`` raises leaves nothing durable, so the record
    must be redelivered rather than acked. On ``origin/dev`` this returned
    normally with zero records published -- the silent drop, reached through the
    arm the design calls the safe one.
    """

    async def _run() -> None:
        bus = EventBusInmemory(environment="test")
        await bus.start()
        try:

            async def _raising_publish(
                topic: str, key: bytes | None, value: bytes, headers: object = None
            ) -> ModelPublishReceipt:
                raise ConnectionError("broker unavailable")

            bus.publish = _raising_publish  # type: ignore[method-assign]

            with patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN):
                with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
                    await _callback(bus)(_envelope())
        finally:
            await bus.close()

    with pytest.raises(ProjectionNotMaterializedError) as raised:
        asyncio.run(_run())

    assert "quarantine" in str(raised.value).lower(), (
        "the message must name the QUARANTINE failure that withheld the offset, "
        "not only the parse failure that triggered it"
    )
