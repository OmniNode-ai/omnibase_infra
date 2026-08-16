# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Cross-boundary seam proof: producer -> receipt -> readback -> ack (OMN-15861).

This is the regression test the seam doctrine (OMN-14208) requires: it drives
the **actual** seam end to end with real objects on both ends, rather than
asserting the producer's side in one unit suite and the consumer's side in
another. Two individually-green unit suites either side of a boundary are the
exact shape that produced the OMN-14208 near-miss -- a pair of PRs that were a
silent 100% runtime no-op.

What is real here, and why each has to be:

* ``EventBusInmemory`` -- the real infra bus, started, assigning real offsets.
  A fake bus would let a hand-written offset satisfy a hand-written readback.
* ``publish()`` return -- the real ``ModelPublishReceipt``. Nothing in this file
  constructs a receipt by hand.
* ``InmemoryReadbackSource`` -- reads the coordinate back out of the bus's own
  history, independently of the value ``publish`` returned.
* ``BrokerReadbackStrategy`` -- the real verdict logic, including fail-closed.
* ``_SeamOutbox`` -- a minimal durable-outbox *driver*, not a mock. It is the
  consumer end of the seam: it holds records, and it truncates one only when a
  confirmation says ``is_durable``. It exists so this test asserts a real ack
  DECISION, not just a verdict object. The production outbox it stands in for
  (``omnimarket`` ``node_emit_daemon``) is a separate repo and lands in the
  paired PR; this driver proves the infra half of the contract those records
  will be acked against.

Zero infrastructure, zero LAN: no broker, no container, no network. That is the
falsifiable-proof surface the brief scopes the first version to.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from omnibase_infra.enums import EnumConfirmationState, EnumInfraTransportType
from omnibase_infra.event_bus.confirmation import (
    BrokerReadbackStrategy,
    InmemoryReadbackSource,
    PublishReturnOnlyStrategy,
)
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelPublishReceipt

pytestmark = pytest.mark.integration

SEAM_TOPIC = "onex.evt.seam.durable.v1"
SEAM_ENVIRONMENT = "seamtest"
SEAM_CLUSTER = f"inmemory.{SEAM_ENVIRONMENT}"


class _SeamOutbox:
    """A durable outbox reduced to the one decision under test.

    ``append`` -> ``flush`` -> the record survives unless a confirmation is
    durable. Deliberately has no notion of "publish succeeded": the only input
    to truncation is a ``ModelDurabilityConfirmation``.
    """

    def __init__(self) -> None:
        self.pending: list[str] = []
        self.acked: list[str] = []
        self.applied_effects: list[str] = []

    def append(self, idempotency_key: str) -> None:
        self.pending.append(idempotency_key)

    async def flush_one(
        self,
        bus: EventBusInmemory,
        strategy: BrokerReadbackStrategy | PublishReturnOnlyStrategy,
    ) -> ModelPublishReceipt | None:
        """Publish the oldest pending record and ack ONLY on confirmation."""
        if not self.pending:
            return None
        key = self.pending[0]
        headers = ModelEventHeaders(
            source="seam-test",
            event_type=SEAM_TOPIC,
            timestamp=datetime.now(UTC),
            idempotency_key=key,
        )
        receipt = await bus.publish(
            topic=SEAM_TOPIC,
            key=None,
            value=key.encode("utf-8"),
            headers=headers,
        )
        confirmation = await strategy.confirm(receipt)
        if confirmation.is_durable:
            self.pending.remove(key)
            self.acked.append(key)
            self.applied_effects.append(key)
        return receipt


@pytest.fixture
async def bus() -> EventBusInmemory:
    """A started, real in-memory bus."""
    instance = EventBusInmemory(environment=SEAM_ENVIRONMENT, group="seam")
    await instance.start()
    try:
        yield instance
    finally:
        await instance.close()


@pytest.fixture
def strategy(bus: EventBusInmemory) -> BrokerReadbackStrategy:
    """Real readback strategy bound to the same bus's history."""
    return BrokerReadbackStrategy(
        InmemoryReadbackSource(bus, cluster=SEAM_CLUSTER),
        readback_deadline_seconds=1.0,
    )


@pytest.mark.asyncio
async def test_publish_returns_the_offset_the_bus_actually_assigned(
    bus: EventBusInmemory,
) -> None:
    """The producer half of the seam: the receipt is not decorative.

    Publishes three records and asserts the returned offsets are exactly the
    monotonic sequence the bus assigned -- checked against the bus's own
    ``get_topic_offset`` rather than against a constant, so a receipt that
    invented its offset would fail here.
    """
    receipts: list[ModelPublishReceipt] = []
    for index in range(3):
        headers = ModelEventHeaders(
            source="seam-test",
            event_type=SEAM_TOPIC,
            timestamp=datetime.now(UTC),
            idempotency_key=f"evt-{index}",
        )
        receipts.append(
            await bus.publish(
                topic=SEAM_TOPIC,
                key=None,
                value=f"payload-{index}".encode(),
                headers=headers,
            )
        )

    assert [r.offset for r in receipts] == [0, 1, 2]
    assert await bus.get_topic_offset(SEAM_TOPIC) == 3
    assert {r.transport for r in receipts} == {EnumInfraTransportType.INMEMORY}
    assert {r.cluster for r in receipts} == {SEAM_CLUSTER}
    assert [r.idempotency_key for r in receipts] == ["evt-0", "evt-1", "evt-2"]


@pytest.mark.asyncio
async def test_receipt_confirms_by_independent_readback(
    bus: EventBusInmemory, strategy: BrokerReadbackStrategy
) -> None:
    """The full seam: publish -> receipt -> read the coordinate back -> CONFIRMED.

    The readback consults the bus's history by coordinate; it does not trust the
    receipt's own word. A receipt whose coordinate did not correspond to a
    stored record would come back UNCONFIRMED.
    """
    headers = ModelEventHeaders(
        source="seam-test",
        event_type=SEAM_TOPIC,
        timestamp=datetime.now(UTC),
        idempotency_key="confirm-me",
    )
    receipt = await bus.publish(
        topic=SEAM_TOPIC, key=None, value=b"payload", headers=headers
    )

    confirmation = await strategy.confirm(receipt)

    assert confirmation.state is EnumConfirmationState.CONFIRMED
    assert confirmation.is_durable is True
    assert confirmation.receipt == receipt
    assert confirmation.strategy == "broker_readback"


@pytest.mark.asyncio
async def test_outbox_acks_only_after_confirmation(
    bus: EventBusInmemory, strategy: BrokerReadbackStrategy
) -> None:
    """The consumer half: a confirmed record leaves the outbox exactly once."""
    outbox = _SeamOutbox()
    outbox.append("logical-1")

    await outbox.flush_one(bus, strategy)

    assert outbox.pending == []
    assert outbox.acked == ["logical-1"]
    assert outbox.applied_effects == ["logical-1"]


@pytest.mark.asyncio
async def test_publish_succeeds_but_confirmation_fails_keeps_the_record(
    bus: EventBusInmemory,
) -> None:
    """The zero-false-durable-claims half of the falsifiable proof.

    The publish genuinely succeeds -- a real record with a real offset lands in
    the bus. Only the *confirmation* surface is unavailable. Under the old
    ack-on-publish-return behaviour this record would have been truncated; under
    the seam it must survive, unacked, with no effect applied.
    """

    class _UnreachableSource:
        @property
        def transport(self) -> EnumInfraTransportType:
            return EnumInfraTransportType.INMEMORY

        async def observe(
            self, receipt: ModelPublishReceipt, *, deadline_seconds: float
        ) -> bool:
            raise ConnectionResetError("projection store unavailable")

    outbox = _SeamOutbox()
    outbox.append("logical-2")
    failing_strategy = BrokerReadbackStrategy(
        _UnreachableSource(), readback_deadline_seconds=0.2
    )

    receipt = await outbox.flush_one(bus, failing_strategy)

    # The publish itself succeeded and produced a real coordinate...
    assert receipt is not None
    assert receipt.offset == 0
    # ...and the record is genuinely on the bus.
    history = await bus.get_event_history(topic=SEAM_TOPIC)
    assert len(history) == 1
    # ...but nothing was acked and no effect was applied.
    assert outbox.pending == ["logical-2"]
    assert outbox.acked == []
    assert outbox.applied_effects == []


@pytest.mark.asyncio
async def test_unconfirmed_record_is_reserved_until_it_confirms(
    bus: EventBusInmemory, strategy: BrokerReadbackStrategy
) -> None:
    """A retained record is re-served and acked once, not dropped and not doubled.

    Drives the same logical event through a failing confirm and then a working
    one, and asserts the effect is applied exactly once across both attempts --
    the zero-duplicate-side-effects half.
    """

    class _FlakySource:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def transport(self) -> EnumInfraTransportType:
            return EnumInfraTransportType.INMEMORY

        async def observe(
            self, receipt: ModelPublishReceipt, *, deadline_seconds: float
        ) -> bool:
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("readback timed out")
            return True

    outbox = _SeamOutbox()
    outbox.append("logical-3")
    flaky = _FlakySource()
    flaky_strategy = BrokerReadbackStrategy(flaky, readback_deadline_seconds=0.5)

    first = await outbox.flush_one(bus, flaky_strategy)
    assert outbox.pending == ["logical-3"]
    assert outbox.applied_effects == []

    second = await outbox.flush_one(bus, flaky_strategy)
    assert outbox.pending == []
    assert outbox.acked == ["logical-3"]

    # Republished on retry, so two distinct coordinates exist for one logical
    # event -- which is precisely why the idempotency key, not the offset, is
    # the identity a downstream dedupe must key on.
    assert first is not None
    assert second is not None
    assert first.offset != second.offset
    assert first.idempotency_key == second.idempotency_key == "logical-3"
    # The effect was applied exactly once despite two publishes.
    assert outbox.applied_effects == ["logical-3"]


@pytest.mark.asyncio
async def test_publish_return_only_would_have_acked_the_unconfirmed_record(
    bus: EventBusInmemory,
) -> None:
    """Names the old behaviour and pins it as the weaker, attributable choice.

    This is the control for the test above: with ``PublishReturnOnlyStrategy``
    the same record IS acked without any readback. Keeping this green and
    explicitly labelled is what stops the weak path from silently becoming the
    default again -- and the ``strategy`` field on the confirmation makes any
    such ack auditable after the fact.
    """
    outbox = _SeamOutbox()
    outbox.append("logical-4")

    await outbox.flush_one(bus, PublishReturnOnlyStrategy())

    assert outbox.acked == ["logical-4"]


@pytest.mark.asyncio
async def test_concurrent_publishes_get_distinct_coordinates(
    bus: EventBusInmemory, strategy: BrokerReadbackStrategy
) -> None:
    """Receipts must not collide under concurrency, or dedupe breaks silently."""
    headers = [
        ModelEventHeaders(
            source="seam-test",
            event_type=SEAM_TOPIC,
            timestamp=datetime.now(UTC),
            idempotency_key=f"concurrent-{i}",
        )
        for i in range(10)
    ]
    receipts = await asyncio.gather(
        *(
            bus.publish(topic=SEAM_TOPIC, key=None, value=f"c{i}".encode(), headers=h)
            for i, h in enumerate(headers)
        )
    )

    offsets = [r.offset for r in receipts]
    assert len(set(offsets)) == 10
    assert sorted(offsets) == list(range(10))

    confirmations = await asyncio.gather(*(strategy.confirm(r) for r in receipts))
    assert all(c.is_durable for c in confirmations)
