# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for consumer-group lag models + observer + drain-proof gate (OMN-12623).

TDD: these tests assert the drain-proof gate behavior required by the ticket —
"cutover blocks while old lag remains; retirement only after drain proof".
"""

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_cutover_criterion import EnumCutoverCriterion
from omnibase_core.enums.enum_migration_phase import EnumMigrationPhase
from omnibase_core.models.contracts.model_topic_migration_contract import (
    ModelTopicMigrationContract,
)
from omnibase_core.models.contracts.model_topic_schema_binding import (
    ModelTopicSchemaBinding,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.migration.models.model_consumer_group_lag import (
    ModelConsumerGroupLag,
)
from omnibase_infra.migration.models.model_topic_partition_offset import (
    ModelTopicPartitionOffset,
)
from omnibase_infra.migration.service_consumer_lag_observer import (
    ServiceConsumerLagObserver,
)
from omnibase_infra.migration.service_drain_proof_gate import ServiceDrainProofGate

pytestmark = pytest.mark.unit


class _FakeTopicPartition:
    """Minimal TopicPartition with .topic/.partition for the fake admin."""

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


class _FakeOffset:
    """Minimal offset response with .offset."""

    def __init__(self, offset: int) -> None:
        self.offset = offset


class _FakeAdmin:
    """In-memory ProtocolKafkaAdminLike for committed/end offsets."""

    def __init__(
        self,
        committed: dict[_FakeTopicPartition, int],
        end: dict[_FakeTopicPartition, int],
    ) -> None:
        self._committed = committed
        self._end = end

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
        return {tp: _FakeOffset(off) for tp, off in self._committed.items()}

    async def list_offsets(self, topic_partitions):
        return {tp: _FakeOffset(self._end[tp]) for tp in topic_partitions}


def _binding(topic: str, major: int) -> ModelTopicSchemaBinding:
    return ModelTopicSchemaBinding(
        topic=topic,
        event_name="ORDER_PLACED",
        schema_version=ModelSemVer(major=major, minor=0, patch=0),
    )


def _migration_contract(drain_required: bool = True) -> ModelTopicMigrationContract:
    return ModelTopicMigrationContract(
        contract_version=ModelSemVer(major=1, minor=0, patch=0),
        ticket="OMN-12623",
        old_binding=_binding("onex.evt.orders.order-placed.v1", 1),
        new_binding=_binding("onex.evt.orders.order-placed.v2", 2),
        old_consumer_group="dev.orders.order-placed.consume.v1",
        new_consumer_group="dev.orders.order-placed.consume.v2",
        compatibility_window_hours=24,
        cutover_criteria=(EnumCutoverCriterion.OLD_TOPIC_DRAINED,),
        drain_proof_required=drain_required,
        phase=EnumMigrationPhase.DUAL_READ,
    )


# ---------------------------------------------------------------------------
# Lag model
# ---------------------------------------------------------------------------


def test_partition_offset_lag_is_difference() -> None:
    po = ModelTopicPartitionOffset(
        topic="onex.evt.orders.order-placed.v1",
        partition=0,
        committed_offset=10,
        log_end_offset=15,
    )
    assert po.lag == 5


def test_partition_offset_rejects_committed_beyond_end() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        ModelTopicPartitionOffset(
            topic="t.v1",
            partition=0,
            committed_offset=20,
            log_end_offset=10,
        )


def test_group_lag_drained_only_when_zero_total_lag() -> None:
    drained = ModelConsumerGroupLag(
        group_id="g",
        partition_offsets=(
            ModelTopicPartitionOffset(
                topic="t.v1", partition=0, committed_offset=5, log_end_offset=5
            ),
        ),
    )
    assert drained.is_drained is True
    assert drained.total_lag == 0


def test_group_lag_not_drained_with_residual() -> None:
    lagging = ModelConsumerGroupLag(
        group_id="g",
        partition_offsets=(
            ModelTopicPartitionOffset(
                topic="t.v1", partition=0, committed_offset=3, log_end_offset=5
            ),
        ),
    )
    assert lagging.is_drained is False
    assert lagging.total_lag == 2


def test_group_lag_empty_is_not_drained() -> None:
    empty = ModelConsumerGroupLag(group_id="g", partition_offsets=())
    assert empty.is_drained is False


# ---------------------------------------------------------------------------
# Lag observer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observer_computes_lag_from_committed_and_end() -> None:
    tp0 = _FakeTopicPartition("onex.evt.orders.order-placed.v1", 0)
    tp1 = _FakeTopicPartition("onex.evt.orders.order-placed.v1", 1)
    admin = _FakeAdmin(
        committed={tp0: 10, tp1: 7},
        end={tp0: 10, tp1: 12},
    )
    observer = ServiceConsumerLagObserver(admin)
    lag = await observer.observe("dev.orders.order-placed.consume.v1")
    assert lag.total_lag == 5
    assert lag.lag_for_topic("onex.evt.orders.order-placed.v1") == 5


@pytest.mark.asyncio
async def test_observer_normalizes_negative_committed_to_zero() -> None:
    tp0 = _FakeTopicPartition("onex.evt.orders.order-placed.v1", 0)
    admin = _FakeAdmin(committed={tp0: -1}, end={tp0: 4})
    observer = ServiceConsumerLagObserver(admin)
    lag = await observer.observe("g")
    assert lag.total_lag == 4


# ---------------------------------------------------------------------------
# Drain-proof gate — the core ticket requirement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_blocks_retirement_while_lag_remains() -> None:
    tp0 = _FakeTopicPartition("onex.evt.orders.order-placed.v1", 0)
    admin = _FakeAdmin(committed={tp0: 3}, end={tp0: 5})
    gate = ServiceDrainProofGate(ServiceConsumerLagObserver(admin))
    decision = await gate.evaluate(_migration_contract())
    assert decision.retirement_allowed is False
    assert decision.residual_lag == 2


@pytest.mark.asyncio
async def test_gate_allows_retirement_after_drain_proof() -> None:
    tp0 = _FakeTopicPartition("onex.evt.orders.order-placed.v1", 0)
    admin = _FakeAdmin(committed={tp0: 5}, end={tp0: 5})
    gate = ServiceDrainProofGate(ServiceConsumerLagObserver(admin))
    decision = await gate.evaluate(_migration_contract())
    assert decision.retirement_allowed is True
    assert decision.residual_lag == 0


@pytest.mark.asyncio
async def test_gate_blocks_when_no_offsets_observed() -> None:
    admin = _FakeAdmin(committed={}, end={})
    gate = ServiceDrainProofGate(ServiceConsumerLagObserver(admin))
    decision = await gate.evaluate(_migration_contract())
    # Absence of evidence is not proof of drain.
    assert decision.retirement_allowed is False


@pytest.mark.asyncio
async def test_gate_blocks_when_group_has_offsets_but_not_on_old_topic() -> None:
    # The group commits offsets on an UNRELATED topic but nothing on old_topic.
    # Global non-emptiness must not be mistaken for old-topic drain proof.
    other = _FakeTopicPartition("onex.evt.orders.order-shipped.v1", 0)
    admin = _FakeAdmin(committed={other: 5}, end={other: 5})
    gate = ServiceDrainProofGate(ServiceConsumerLagObserver(admin))
    decision = await gate.evaluate(_migration_contract())
    assert decision.retirement_allowed is False
    assert "no committed offsets" in decision.reason


@pytest.mark.asyncio
async def test_gate_allows_when_drain_proof_opted_out() -> None:
    admin = _FakeAdmin(committed={}, end={})
    gate = ServiceDrainProofGate(ServiceConsumerLagObserver(admin))
    # An opt-out contract still requires OLD_TOPIC_DRAINED cutover criterion only
    # when drain_proof_required is True; build a valid opt-out contract.
    contract = ModelTopicMigrationContract(
        contract_version=ModelSemVer(major=1, minor=0, patch=0),
        ticket="OMN-12623",
        old_binding=_binding("onex.evt.orders.order-placed.v1", 1),
        new_binding=_binding("onex.evt.orders.order-placed.v2", 2),
        old_consumer_group="g1",
        new_consumer_group="g2",
        compatibility_window_hours=24,
        cutover_criteria=(EnumCutoverCriterion.COMPATIBILITY_WINDOW_ELAPSED,),
        drain_proof_required=False,
        phase=EnumMigrationPhase.DUAL_READ,
    )
    decision = await gate.evaluate(contract)
    assert decision.retirement_allowed is True
