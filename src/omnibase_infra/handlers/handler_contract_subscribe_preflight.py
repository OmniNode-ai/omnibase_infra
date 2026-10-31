# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a contract's consumer needs from the bus before ``subscribe`` joins it.

Auto-wiring subscribes every topic a contract declares. Two contract
declarations must reach the bus strictly BEFORE that ``subscribe``, because
``subscribe`` starts the consume loop and joins the group, and each reads its
input once, at that moment:

* ``consume_concurrency`` (OMN-18852): the in-flight bound the consume loop
  reads when it starts. Absent -- every node in the fleet but a few -- it is 1,
  and 1 keeps the inline serial ``await``, so an undeclared node is unchanged.
* ``consumer_group_handover`` (OMN-21753, under OMN-21288): a writer folded
  into the runtime from its own container names that container's consumer
  (``ModelConsumerGroupHandover``), and the runtime's group starts from that
  group's committed offsets instead of the bus default (``latest``), which
  dropped every record published between the container stopping and the
  runtime joining.

Both are refused, never skipped: a malformed declaration raises out of its
loader, and a bus that cannot honour a declaration raises ``TypeError`` at
construction, before any topic of the contract is subscribed. An operator who
believes a bound or a handover is in force while the runtime runs serial, or
joins at ``latest``, is the position both tickets started from. The
concurrency half lived inline in ``_subscribe_contract_topics`` until OMN-21753
added the second declaration.

Which offset each partition of the successor group gets is one pure rule,
:func:`plan_offset_handover`, shared by every substrate:

1. A partition the successor group already committed is kept: the handover has
   happened (or the group has its own history), and a second start of the
   runtime resumes from its own position, never the retired one.
2. A partition the retired group committed, at or above the first retained
   offset, starts at that offset: the next record the retired consumer had not
   finished.
3. Any other partition (never committed, or committed below what retention
   kept) starts where the retired consumer would have started on it: the first
   retained offset for ``earliest``, the end for ``latest``. A record published
   during the window on a partition the retired consumer never saw is at or
   after the first retained offset, and the bus default would skip it.

Substrates:

* ``EventBusKafka``: two short-lived consumers, one per group, each manually
  assigned with auto-commit off. A manually assigned consumer never joins its
  group, so reading the retired group cannot disturb it, and the successor's
  offsets go in as a simple commit, which the broker accepts while that group
  has no members. It has none: the Kafka bus scopes every subscription to its
  own per-topic group, resolved here by the bus's own
  ``_resolve_effective_group_id`` so the offsets land on the group the consumer
  will join. (The bus module is baselined and only shrinks, so this lives here
  rather than as a bus method.)
* ``EventBusInmemory``: keeps no committed offset for any group and delivers a
  record only to the subscribers joined when it is published. No retired group
  on it has a position, so nothing is committed.
* A bus that implements ``ProtocolConsumerGroupOffsetHandover`` itself (the
  in-process group-offset substrate in the tests) is asked directly.

The retired consumer must already be stopped (the fold's deploy retires the
container before it recreates the runtime). A handover from a consumer still
running copies a position it then moves past: records projected twice, never
records lost, and the projection writers deduplicate on the event key.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition

from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.models.event_bus.model_consumer_group_handover import (
    ModelConsumerGroupHandover,
)
from omnibase_infra.models.event_bus.model_contract_subscribe_preflight_request import (
    ModelContractSubscribePreflightRequest,
)
from omnibase_infra.models.event_bus.model_contract_subscribe_preflight_result import (
    ModelContractSubscribePreflightResult,
)
from omnibase_infra.models.event_bus.model_offset_handover_result import (
    ModelOffsetHandoverResult,
)
from omnibase_infra.protocols.protocol_consume_concurrency_declarer import (
    ProtocolConsumeConcurrencyDeclarer,
)
from omnibase_infra.protocols.protocol_consumer_group_offset_handover import (
    ProtocolConsumerGroupOffsetHandover,
)
from omnibase_infra.runtime.auto_wiring.models.model_consume_concurrency import (
    load_consume_concurrency,
)

__all__ = [
    "CONSUMER_GROUP_HANDOVER_KEY",
    "HandlerContractSubscribePreflight",
    "hand_over_committed_offsets",
    "load_consumer_group_handover",
    "plan_offset_handover",
    "supports_offset_handover",
]

logger = logging.getLogger(__name__)

CONSUMER_GROUP_HANDOVER_KEY = "consumer_group_handover"


def load_consumer_group_handover(
    contract_path: Path,
) -> ModelConsumerGroupHandover | None:
    """Read ``consumer_group_handover`` from a node contract.

    A path that does not exist resolves to ``None`` for the reason
    ``load_consume_concurrency`` gives: synthetic contracts reach auto-wiring
    with a ``contract_path`` that points at nothing.

    Raises:
        ValueError: The file is not a mapping, the key is not a mapping, or the
            declaration fails validation.
    """
    if not contract_path.exists():
        return None
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{contract_path} must contain a mapping")
    if CONSUMER_GROUP_HANDOVER_KEY not in raw:
        return None
    declared = raw[CONSUMER_GROUP_HANDOVER_KEY]
    if not isinstance(declared, dict):
        raise ValueError(
            f"{contract_path} declares {CONSUMER_GROUP_HANDOVER_KEY} as "
            f"{type(declared).__name__}, which is not a mapping; a handover that "
            "cannot be read must not be treated as absent"
        )
    return ModelConsumerGroupHandover.model_validate(declared)


def plan_offset_handover(
    *,
    partitions: Iterable[int],
    successor_committed: Mapping[int, int | None],
    retired_committed: Mapping[int, int | None],
    first_offsets: Mapping[int, int],
    end_offsets: Mapping[int, int],
    retired_auto_offset_reset: Literal["earliest", "latest"],
) -> dict[int, int]:
    """Partition to offset for every partition the successor must be given.

    A partition absent from the result is one the successor already owns.
    """
    plan: dict[int, int] = {}
    for partition in sorted(partitions):
        if successor_committed.get(partition) is not None:
            continue
        retired = retired_committed.get(partition)
        if retired is not None and retired >= first_offsets[partition]:
            plan[partition] = retired
        elif retired_auto_offset_reset == "earliest":
            plan[partition] = first_offsets[partition]
        else:
            plan[partition] = end_offsets[partition]
    return plan


def supports_offset_handover(event_bus: object) -> bool:
    """Whether a consumer-group handover can be honoured on ``event_bus``."""
    return isinstance(
        event_bus,
        (ProtocolConsumerGroupOffsetHandover, EventBusKafka, EventBusInmemory),
    )


async def _committed(
    consumer: AIOKafkaConsumer, partitions: list[TopicPartition]
) -> dict[int, int | None]:
    committed: dict[int, int | None] = {}
    for tp in partitions:
        offset = await consumer.committed(tp)
        committed[tp.partition] = None if offset is None or offset < 0 else offset
    return committed


async def _hand_over_on_kafka(
    bus: EventBusKafka,
    *,
    topic: str,
    from_group: str,
    to_group: str,
    from_auto_offset_reset: Literal["earliest", "latest"],
) -> ModelOffsetHandoverResult:
    effective = bus._resolve_effective_group_id(
        to_group, topic, uuid4(), (topic, to_group)
    )
    binding = bus.resolve_bus_binding(topic, consumer_group=effective)
    successor_group = binding.consumer_group or effective
    physical_topic = binding.physical_topic
    client_kwargs: dict[str, object] = dict(build_aiokafka_auth_kwargs(bus.config))
    if bus.config.api_version is not None:
        client_kwargs["api_version"] = bus.config.api_version

    def _consumer(group: str) -> AIOKafkaConsumer:
        return AIOKafkaConsumer(
            bootstrap_servers=bus.bootstrap_servers,
            group_id=group,
            enable_auto_commit=False,
            **client_kwargs,
        )

    successor, retired = _consumer(successor_group), _consumer(from_group)
    await successor.start()
    try:
        await retired.start()
        try:
            # ``topics()`` and ``partitions_for_topic()`` read different
            # metadata objects on a consumer with no subscription; read the
            # partitions off the fetched cluster, as the DLQ depth monitor does.
            cluster = await successor._client.fetch_all_metadata()
            partition_ids = sorted(cluster.partitions_for_topic(physical_topic) or ())
            if not partition_ids:
                raise ValueError(
                    f"topic {physical_topic!r} has no partitions in broker "
                    "metadata; nothing can be handed over or consumed"
                )
            partitions = [TopicPartition(physical_topic, p) for p in partition_ids]
            successor.assign(partitions)
            retired.assign(partitions)
            successor_committed = await _committed(successor, partitions)
            retired_committed = await _committed(retired, partitions)
            first = await successor.beginning_offsets(partitions)
            end = await successor.end_offsets(partitions)
            plan = plan_offset_handover(
                partitions=partition_ids,
                successor_committed=successor_committed,
                retired_committed=retired_committed,
                first_offsets={tp.partition: off for tp, off in first.items()},
                end_offsets={tp.partition: off for tp, off in end.items()},
                retired_auto_offset_reset=from_auto_offset_reset,
            )
            if plan:
                await successor.commit(
                    {TopicPartition(physical_topic, p): o for p, o in plan.items()}
                )
        finally:
            await retired.stop()
    finally:
        await successor.stop()
    return ModelOffsetHandoverResult(
        topic=topic,
        from_group=from_group,
        to_group=successor_group,
        committed=plan,
        kept=tuple(p for p in partition_ids if successor_committed[p] is not None),
    )


async def hand_over_committed_offsets(
    event_bus: object,
    *,
    topic: str,
    from_group: str,
    to_group: str,
    from_auto_offset_reset: Literal["earliest", "latest"],
) -> ModelOffsetHandoverResult:
    """Seed ``to_group`` (the id passed to ``subscribe``) from ``from_group``.

    Raises:
        TypeError: ``event_bus`` cannot hand offsets over; see
            :func:`supports_offset_handover`.
    """
    if isinstance(event_bus, ProtocolConsumerGroupOffsetHandover):
        return await event_bus.hand_over_committed_offsets(
            topic=topic,
            from_group=from_group,
            to_group=to_group,
            from_auto_offset_reset=from_auto_offset_reset,
        )
    if isinstance(event_bus, EventBusKafka):
        return await _hand_over_on_kafka(
            event_bus,
            topic=topic,
            from_group=from_group,
            to_group=to_group,
            from_auto_offset_reset=from_auto_offset_reset,
        )
    if isinstance(event_bus, EventBusInmemory):
        return ModelOffsetHandoverResult(
            topic=topic, from_group=from_group, to_group=to_group
        )
    raise TypeError(
        f"{type(event_bus).__name__} cannot hand a retired consumer group's "
        "committed offsets over (OMN-21753)"
    )


class HandlerContractSubscribePreflight:
    """Apply a contract's pre-subscribe declarations to one bus, per topic."""

    def __init__(
        self,
        *,
        contract_name: str,
        contract_path: Path,
        event_bus: object,
        environment: str,
        consumer_group: str,
    ) -> None:
        """Read both declarations and refuse a bus that cannot honour them.

        Args:
            contract_name: The contract, for messages and logs.
            contract_path: The contract YAML the declarations are read from.
            event_bus: The bus the contract's topics will be subscribed on.
            environment: The runtime environment; the handover's retired group
                is derived in it.
            consumer_group: The group id ``subscribe`` resolves to for this
                contract (``compute_consumer_group_id`` of its identity).

        Raises:
            ValueError: A declaration is malformed.
            TypeError: The bus cannot honour a declaration.
        """
        self._contract_name = contract_name
        self._event_bus = event_bus
        self._consumer_group = consumer_group
        self._concurrency = load_consume_concurrency(contract_path)
        if not self._concurrency.is_serial and not isinstance(
            event_bus, ProtocolConsumeConcurrencyDeclarer
        ):
            raise TypeError(
                f"contract '{contract_name}' declares consume_concurrency."
                f"max_in_flight_records={self._concurrency.max_in_flight_records}, "
                f"but the wired event bus ({type(event_bus).__name__}) cannot "
                "bound in-flight records. Implement "
                "ProtocolConsumeConcurrencyDeclarer.declare_consume_concurrency "
                "on the bus and record the declared bound before subscribing. "
                "A declared bound that silently does "
                "nothing is the defect OMN-18852 exists to remove"
            )
        self._handover = load_consumer_group_handover(contract_path)
        if self._handover is not None and not supports_offset_handover(event_bus):
            raise TypeError(
                f"contract '{contract_name}' declares consumer_group_handover, "
                f"but the wired event bus ({type(event_bus).__name__}) cannot "
                "hand a retired group's committed offsets over; joining at the "
                "bus default would drop the handover window (OMN-21753)"
            )
        self._retired_group = (
            self._handover.source_group(environment) if self._handover else ""
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        """INFRA_HANDLER: configures the bus a node consumes through."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """EFFECT: a handover reads and writes broker group offsets."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelContractSubscribePreflightRequest
    ) -> ModelContractSubscribePreflightResult:
        """Apply both declarations for one topic; call strictly before ``subscribe``.

        Scoped to the topics the contract actually subscribes to, so a topic
        skipped as core-runtime owned never carries a bound nothing reads.
        """
        topic = request.topic
        bound = self._concurrency.max_in_flight_records
        if not self._concurrency.is_serial and isinstance(
            self._event_bus, ProtocolConsumeConcurrencyDeclarer
        ):
            self._event_bus.declare_consume_concurrency(
                topic=topic,
                group_id=self._consumer_group,
                max_in_flight_records=bound,
            )
            logger.info(
                "Auto-wired consume concurrency: topic=%s consumer_group=%s "
                "node=%s max_in_flight_records=%d -- this topic no longer "
                "preserves global partition ordering (OMN-18852)",
                topic,
                self._consumer_group,
                self._contract_name,
                bound,
            )
        handover: ModelOffsetHandoverResult | None = None
        if self._handover is not None:
            handover = await hand_over_committed_offsets(
                self._event_bus,
                topic=topic,
                from_group=self._retired_group,
                to_group=self._consumer_group,
                from_auto_offset_reset=self._handover.from_auto_offset_reset,
            )
            logger.info(
                "Consumer group handover: topic=%s from=%s to=%s committed=%s "
                "kept=%s node=%s (OMN-21753)",
                topic,
                handover.from_group,
                handover.to_group,
                handover.committed,
                list(handover.kept),
                self._contract_name,
            )
        return ModelContractSubscribePreflightResult(
            topic=topic, max_in_flight_records=bound, handover=handover
        )
