# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the publish-receipt / confirmation seam (OMN-15861).

Scope note: these are the *unit* halves -- model invariants, strategy verdict
logic, and the Kafka readback loop against an injected consumer. The
cross-boundary proof that a real bus's real ``publish`` return drives a real
confirmation lives in
``tests/integration/event_bus/test_publish_confirmation_seam.py``; two unit
suites either side of a seam are exactly what the seam doctrine forbids as
sufficient evidence.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumConfirmationState, EnumInfraTransportType
from omnibase_infra.event_bus.confirmation import (
    BrokerReadbackStrategy,
    InmemoryReadbackSource,
    KafkaReadbackSource,
    PublishReturnOnlyStrategy,
)
from omnibase_infra.event_bus.models import (
    ModelDurabilityConfirmation,
    ModelPublishReceipt,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

pytestmark = pytest.mark.unit

TEST_TOPIC = "onex.evt.test.receipt.v1"
TEST_CLUSTER = "test-broker:9092"


def _receipt(
    *,
    offset: int = 7,
    partition: int = 0,
    cluster: str = TEST_CLUSTER,
    transport: EnumInfraTransportType = EnumInfraTransportType.KAFKA,
    idempotency_key: str | None = "logical-1",
) -> ModelPublishReceipt:
    return ModelPublishReceipt(
        topic=TEST_TOPIC,
        partition=partition,
        offset=offset,
        cluster=cluster,
        produced_at=datetime.now(UTC),
        transport=transport,
        idempotency_key=idempotency_key,
    )


class TestModelPublishReceipt:
    """The coordinate model's own invariants."""

    def test_coordinate_renders_topic_partition_offset(self) -> None:
        assert _receipt(offset=41, partition=3).coordinate == f"{TEST_TOPIC}/3/41"

    def test_offset_zero_is_valid(self) -> None:
        """Offset 0 is a real coordinate, not a falsy 'missing' sentinel.

        Guards the classic bug where `if not offset:` silently discards the very
        first record on a partition.
        """
        assert _receipt(offset=0).offset == 0

    def test_negative_offset_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelPublishReceipt(
                topic=TEST_TOPIC,
                partition=0,
                offset=-1,
                cluster=TEST_CLUSTER,
                produced_at=datetime.now(UTC),
                transport=EnumInfraTransportType.KAFKA,
            )

    def test_naive_produced_at_rejected(self) -> None:
        """An ambiguous instant is not durable evidence."""
        with pytest.raises(ValidationError):
            ModelPublishReceipt(
                topic=TEST_TOPIC,
                partition=0,
                offset=1,
                cluster=TEST_CLUSTER,
                produced_at=datetime(2026, 8, 13),
                transport=EnumInfraTransportType.KAFKA,
            )

    def test_receipt_is_frozen(self) -> None:
        """A durability coordinate must not be editable after the fact."""
        receipt = _receipt()
        with pytest.raises(ValidationError):
            receipt.offset = 99  # type: ignore[misc]


class TestModelDurabilityConfirmation:
    """Verdict-model invariants, including the fail-closed reading."""

    def test_unknown_is_not_durable(self) -> None:
        """UNKNOWN fails closed. This is the whole point of the tri-state."""
        outcome = ModelDurabilityConfirmation(
            state=EnumConfirmationState.UNKNOWN,
            strategy="broker_readback",
            receipt=_receipt(),
            checked_at=datetime.now(UTC),
            detail="broker unreachable",
        )
        assert outcome.is_durable is False

    def test_unconfirmed_is_not_durable(self) -> None:
        outcome = ModelDurabilityConfirmation(
            state=EnumConfirmationState.UNCONFIRMED,
            strategy="broker_readback",
            receipt=_receipt(),
            checked_at=datetime.now(UTC),
            detail="not observed in budget",
        )
        assert outcome.is_durable is False

    def test_confirmed_is_durable(self) -> None:
        outcome = ModelDurabilityConfirmation(
            state=EnumConfirmationState.CONFIRMED,
            strategy="broker_readback",
            receipt=_receipt(),
            checked_at=datetime.now(UTC),
        )
        assert outcome.is_durable is True

    def test_non_confirmed_without_detail_rejected(self) -> None:
        """A stuck record must always carry its own explanation."""
        with pytest.raises(ValidationError):
            ModelDurabilityConfirmation(
                state=EnumConfirmationState.UNCONFIRMED,
                strategy="broker_readback",
                receipt=_receipt(),
                checked_at=datetime.now(UTC),
                detail="   ",
            )


class _RaisingSource:
    """A readback surface that cannot be consulted at all."""

    @property
    def transport(self) -> EnumInfraTransportType:
        return EnumInfraTransportType.KAFKA

    async def observe(
        self, receipt: ModelPublishReceipt, *, deadline_seconds: float
    ) -> bool:
        raise ConnectionRefusedError("broker down")


class _AbsentSource:
    """A reachable surface that does not have the record."""

    @property
    def transport(self) -> EnumInfraTransportType:
        return EnumInfraTransportType.KAFKA

    async def observe(
        self, receipt: ModelPublishReceipt, *, deadline_seconds: float
    ) -> bool:
        return False


class _PresentSource:
    """A reachable surface that has the record."""

    @property
    def transport(self) -> EnumInfraTransportType:
        return EnumInfraTransportType.KAFKA

    async def observe(
        self, receipt: ModelPublishReceipt, *, deadline_seconds: float
    ) -> bool:
        return True


class TestBrokerReadbackStrategy:
    """The three-way verdict, and that it never raises."""

    @pytest.mark.asyncio
    async def test_observed_record_confirms(self) -> None:
        outcome = await BrokerReadbackStrategy(_PresentSource()).confirm(_receipt())
        assert outcome.state is EnumConfirmationState.CONFIRMED
        assert outcome.is_durable is True

    @pytest.mark.asyncio
    async def test_absent_record_is_unconfirmed_not_unknown(self) -> None:
        """'The broker says no' is a different fact from 'the broker never answered'."""
        outcome = await BrokerReadbackStrategy(_AbsentSource()).confirm(_receipt())
        assert outcome.state is EnumConfirmationState.UNCONFIRMED
        assert outcome.is_durable is False

    @pytest.mark.asyncio
    async def test_unreachable_source_fails_closed_as_unknown(self) -> None:
        """An indeterminate check must never be read as success."""
        outcome = await BrokerReadbackStrategy(_RaisingSource()).confirm(_receipt())
        assert outcome.state is EnumConfirmationState.UNKNOWN
        assert outcome.is_durable is False
        assert "ConnectionRefusedError" in outcome.detail

    @pytest.mark.asyncio
    async def test_missing_receipt_is_unknown(self) -> None:
        outcome = await BrokerReadbackStrategy(_PresentSource()).confirm(None)
        assert outcome.state is EnumConfirmationState.UNKNOWN
        assert outcome.is_durable is False

    @pytest.mark.asyncio
    async def test_transport_mismatch_refuses_to_confirm(self) -> None:
        """An in-memory history cannot vouch for a Kafka produce.

        Without this guard a misconfigured wiring would happily 'confirm' every
        Kafka record against a local list -- a false durable claim that looks
        green in every test.
        """
        outcome = await BrokerReadbackStrategy(_PresentSource()).confirm(
            _receipt(transport=EnumInfraTransportType.INMEMORY)
        )
        assert outcome.state is EnumConfirmationState.UNKNOWN
        assert "does not match" in outcome.detail

    def test_non_positive_deadline_rejected(self) -> None:
        with pytest.raises(ValueError, match="readback_deadline_seconds"):
            BrokerReadbackStrategy(_PresentSource(), readback_deadline_seconds=0)


class TestPublishReturnOnlyStrategy:
    """The weak strategy is allowed, but it is named and it still fails closed."""

    @pytest.mark.asyncio
    async def test_confirms_on_receipt_presence(self) -> None:
        outcome = await PublishReturnOnlyStrategy().confirm(_receipt())
        assert outcome.state is EnumConfirmationState.CONFIRMED

    @pytest.mark.asyncio
    async def test_records_its_own_weakness_by_name(self) -> None:
        """Attribution is the point: a durable claim must say what backed it."""
        outcome = await PublishReturnOnlyStrategy().confirm(_receipt())
        assert outcome.strategy == "publish_return_only"

    @pytest.mark.asyncio
    async def test_no_receipt_is_unknown_even_here(self) -> None:
        outcome = await PublishReturnOnlyStrategy().confirm(None)
        assert outcome.state is EnumConfirmationState.UNKNOWN


class _FakeRecord:
    def __init__(self, topic: str, partition: int, offset: int) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset


class _FakeSeekableConsumer:
    """Records the assign/seek/fetch sequence a coordinate readback must perform."""

    def __init__(
        self,
        *,
        high_water_mark: int,
        record: _FakeRecord | None,
        hang_on_fetch: bool = False,
    ) -> None:
        self.high_water_mark = high_water_mark
        self.record = record
        self.hang_on_fetch = hang_on_fetch
        self.started = False
        self.stopped = False
        self.assigned: list[Any] = []
        self.sought: list[tuple[Any, int]] = []
        self.group_joined = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def assign(self, partitions: list[Any]) -> None:
        self.assigned = list(partitions)

    def seek(self, partition: Any, offset: int) -> None:
        self.sought.append((partition, offset))

    async def end_offsets(self, partitions: list[Any]) -> dict[Any, int]:
        return {partitions[0]: self.high_water_mark}

    async def getone(self) -> Any:
        if self.hang_on_fetch:
            await asyncio.sleep(3600)
        if self.record is None:
            raise AssertionError("getone called with no record staged")
        return self.record


def _kafka_config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(bootstrap_servers=TEST_CLUSTER, environment="test")


class TestKafkaReadbackSource:
    """The assign+seek loop, exercised without a broker."""

    @pytest.mark.asyncio
    async def test_observes_record_at_coordinate(self) -> None:
        consumer = _FakeSeekableConsumer(
            high_water_mark=8, record=_FakeRecord(TEST_TOPIC, 0, 7)
        )
        source = KafkaReadbackSource(_kafka_config(), consumer_factory=lambda: consumer)
        assert await source.observe(_receipt(offset=7), deadline_seconds=1.0) is True
        assert consumer.sought == [(consumer.assigned[0], 7)]
        assert consumer.stopped is True

    @pytest.mark.asyncio
    async def test_never_joins_a_consumer_group(self) -> None:
        """Confirmation must not mutate anyone's committed offsets.

        Asserted structurally: the source only ever `assign`s, so no group
        membership and no commit can occur as a side effect of confirming.
        """
        consumer = _FakeSeekableConsumer(
            high_water_mark=8, record=_FakeRecord(TEST_TOPIC, 0, 7)
        )
        source = KafkaReadbackSource(_kafka_config(), consumer_factory=lambda: consumer)
        await source.observe(_receipt(offset=7), deadline_seconds=1.0)
        assert consumer.assigned
        assert consumer.group_joined is False

    @pytest.mark.asyncio
    async def test_offset_beyond_high_water_mark_is_not_observed(self) -> None:
        """hwm == offset means the record has NOT been written yet."""
        consumer = _FakeSeekableConsumer(high_water_mark=7, record=None)
        source = KafkaReadbackSource(_kafka_config(), consumer_factory=lambda: consumer)
        assert await source.observe(_receipt(offset=7), deadline_seconds=0.05) is False
        assert consumer.stopped is True

    @pytest.mark.asyncio
    async def test_fetch_timeout_is_not_observed(self) -> None:
        consumer = _FakeSeekableConsumer(
            high_water_mark=8, record=None, hang_on_fetch=True
        )
        source = KafkaReadbackSource(_kafka_config(), consumer_factory=lambda: consumer)
        assert await source.observe(_receipt(offset=7), deadline_seconds=0.05) is False
        assert consumer.stopped is True

    @pytest.mark.asyncio
    async def test_foreign_cluster_is_not_observed(self) -> None:
        """Same (topic, partition, offset) on a different cluster is a different record."""
        consumer = _FakeSeekableConsumer(
            high_water_mark=8, record=_FakeRecord(TEST_TOPIC, 0, 7)
        )
        source = KafkaReadbackSource(_kafka_config(), consumer_factory=lambda: consumer)
        foreign = _receipt(offset=7, cluster="other-broker:9092")
        assert await source.observe(foreign, deadline_seconds=1.0) is False
        assert consumer.started is False

    @pytest.mark.asyncio
    async def test_broker_error_propagates_for_the_strategy_to_classify(self) -> None:
        """`observe` reports facts; only the strategy decides UNKNOWN vs UNCONFIRMED."""

        class _ExplodingConsumer(_FakeSeekableConsumer):
            async def end_offsets(self, partitions: list[Any]) -> dict[Any, int]:
                raise ConnectionResetError("broker went away")

        consumer = _ExplodingConsumer(high_water_mark=8, record=None)
        source = KafkaReadbackSource(_kafka_config(), consumer_factory=lambda: consumer)
        with pytest.raises(ConnectionResetError):
            await source.observe(_receipt(offset=7), deadline_seconds=1.0)
        assert consumer.stopped is True


class _FakeHistoryBus:
    """Minimal in-memory history stand-in for the readback source."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages

    async def get_event_history(
        self, limit: int = 100, topic: str | None = None
    ) -> list[object]:
        return [m for m in self._messages if topic is None or m.topic == topic]


class TestInmemoryReadbackSource:
    """Coordinate matching against the zero-infra surface."""

    @pytest.mark.asyncio
    async def test_matches_string_offset_from_bus_history(self) -> None:
        """The bus stores offset as `str`; the receipt carries `int`."""

        class _Msg:
            topic = TEST_TOPIC
            partition = 0
            offset = "7"

        source = InmemoryReadbackSource(
            _FakeHistoryBus([_Msg()]), cluster="inmemory.test"
        )
        receipt = _receipt(
            offset=7, cluster="inmemory.test", transport=EnumInfraTransportType.INMEMORY
        )
        assert await source.observe(receipt, deadline_seconds=0.2) is True

    @pytest.mark.asyncio
    async def test_absent_offset_is_not_observed(self) -> None:
        class _Msg:
            topic = TEST_TOPIC
            partition = 0
            offset = "6"

        source = InmemoryReadbackSource(
            _FakeHistoryBus([_Msg()]), cluster="inmemory.test"
        )
        receipt = _receipt(
            offset=7, cluster="inmemory.test", transport=EnumInfraTransportType.INMEMORY
        )
        assert await source.observe(receipt, deadline_seconds=0.05) is False

    @pytest.mark.asyncio
    async def test_receipt_from_a_different_bus_is_not_observed(self) -> None:
        class _Msg:
            topic = TEST_TOPIC
            partition = 0
            offset = "7"

        source = InmemoryReadbackSource(
            _FakeHistoryBus([_Msg()]), cluster="inmemory.test"
        )
        receipt = _receipt(
            offset=7,
            cluster="inmemory.other",
            transport=EnumInfraTransportType.INMEMORY,
        )
        assert await source.observe(receipt, deadline_seconds=0.05) is False
