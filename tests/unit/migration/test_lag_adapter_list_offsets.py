# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression: lag observation against the real aiokafka 0.13.0 admin surface (OMN-12632).

The merged OMN-12623 unit tests stubbed ``list_offsets`` onto a fake admin, which
masked the production crash: ``aiokafka==0.13.0``'s ``AIOKafkaAdminClient`` has
**no** ``list_offsets`` method, so ``ServiceConsumerLagObserver.observe`` raised
``AttributeError`` against the real runtime admin (found by the live .201 proof of
OMN-12623).

These tests deliberately use a fake admin that mirrors the 0.13.0 surface —
``list_consumer_group_offsets`` exists, ``list_offsets`` does **not** — so the
missing method cannot be hidden. ``test_observer_crashes_on_raw_0_13_0_admin``
pins the buggy behavior; the remaining tests prove :class:`AdapterKafkaAdminLag`
restores lag observation by serving log-end offsets through the consumer's
``end_offsets`` (which 0.13.0 *does* provide).
"""

from __future__ import annotations

import pytest

from omnibase_infra.migration.adapter_kafka_admin_lag import AdapterKafkaAdminLag
from omnibase_infra.migration.service_consumer_lag_observer import (
    ServiceConsumerLagObserver,
)

pytestmark = pytest.mark.unit


class _FakeTopicPartition:
    """TopicPartition-like key with .topic/.partition, hashable."""

    def __init__(self, topic: str, partition: int) -> None:
        self.topic = topic
        self.partition = partition

    def __hash__(self) -> int:
        return hash((self.topic, self.partition))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _FakeTopicPartition)
            and other.topic == self.topic
            and other.partition == self.partition
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_FakeTopicPartition({self.topic!r}, {self.partition})"


class _FakeOffsetAndMetadata:
    """Committed-offset value exposing .offset, as the real admin returns."""

    def __init__(self, offset: int) -> None:
        self.offset = offset


class _FakeAiokafka0130Admin:
    """Mirrors the aiokafka 0.13.0 admin surface: it has NO ``list_offsets``.

    Crucially this class does not define ``list_offsets`` — attribute access for
    it raises ``AttributeError`` exactly like the pinned ``AIOKafkaAdminClient``,
    so any code that calls ``admin.list_offsets`` is provably broken.
    """

    def __init__(self, committed: dict[_FakeTopicPartition, int]) -> None:
        self._committed = committed

    async def start(self) -> None:  # pragma: no cover - protocol shape
        return None

    async def stop(self) -> None:  # pragma: no cover
        return None

    async def close(self) -> None:  # pragma: no cover
        return None

    async def list_consumer_groups(self, broker_ids=None):  # pragma: no cover
        return []

    async def describe_consumer_groups(self, group_ids, **kwargs):  # pragma: no cover
        return []

    async def list_consumer_group_offsets(self, group_id, **kwargs):
        return {tp: _FakeOffsetAndMetadata(off) for tp, off in self._committed.items()}


class _FakeAiokafka0130Consumer:
    """Mirrors the aiokafka 0.13.0 consumer surface: ``end_offsets`` returns ints."""

    def __init__(self, end: dict[_FakeTopicPartition, int]) -> None:
        self._end = end
        self.requested: list[_FakeTopicPartition] | None = None

    async def end_offsets(self, partitions):
        self.requested = list(partitions)
        return {tp: self._end[tp] for tp in self.requested}


@pytest.mark.asyncio
async def test_observer_crashes_on_raw_0_13_0_admin() -> None:
    """The pre-fix wiring (observer over a raw 0.13.0 admin) must crash.

    This is the regression the merged tests masked by stubbing ``list_offsets``.
    Without that stub, ``observe`` hits the missing method.
    """
    tp0 = _FakeTopicPartition("onex.evt.orders.order-placed.v1", 0)
    admin = _FakeAiokafka0130Admin(committed={tp0: 3})
    observer = ServiceConsumerLagObserver(admin)
    with pytest.raises(AttributeError, match="list_offsets"):
        await observer.observe("dev.orders.order-placed.consume.v1")


@pytest.mark.asyncio
async def test_adapter_serves_lag_via_consumer_end_offsets() -> None:
    """Observer wired to the adapter computes lag against the 0.13.0 surface."""
    tp0 = _FakeTopicPartition("onex.evt.orders.order-placed.v1", 0)
    tp1 = _FakeTopicPartition("onex.evt.orders.order-placed.v1", 1)
    admin = _FakeAiokafka0130Admin(committed={tp0: 10, tp1: 7})
    consumer = _FakeAiokafka0130Consumer(end={tp0: 10, tp1: 12})

    adapter = AdapterKafkaAdminLag(admin, consumer)
    observer = ServiceConsumerLagObserver(adapter)
    lag = await observer.observe("dev.orders.order-placed.consume.v1")

    assert lag.total_lag == 5
    assert lag.lag_for_topic("onex.evt.orders.order-placed.v1") == 5
    # The adapter must query log-end for exactly the committed partitions.
    assert consumer.requested is not None
    assert set(consumer.requested) == {tp0, tp1}


@pytest.mark.asyncio
async def test_adapter_committed_offsets_delegate_to_real_admin() -> None:
    """Committed offsets stay on the admin's ``list_consumer_group_offsets``."""
    tp0 = _FakeTopicPartition("onex.evt.orders.order-placed.v1", 0)
    admin = _FakeAiokafka0130Admin(committed={tp0: 4})
    consumer = _FakeAiokafka0130Consumer(end={tp0: 9})

    adapter = AdapterKafkaAdminLag(admin, consumer)
    committed = await adapter.list_consumer_group_offsets("g")
    assert committed[tp0].offset == 4


@pytest.mark.asyncio
async def test_adapter_rejects_non_latest_offset_request() -> None:
    """Only the LATEST (log-end) request shape is serviceable via end_offsets."""
    tp0 = _FakeTopicPartition("t.v1", 0)
    admin = _FakeAiokafka0130Admin(committed={tp0: 0})
    consumer = _FakeAiokafka0130Consumer(end={tp0: 0})

    adapter = AdapterKafkaAdminLag(admin, consumer)
    with pytest.raises(ValueError, match="LATEST"):
        await adapter.list_offsets({tp0: 12345})
