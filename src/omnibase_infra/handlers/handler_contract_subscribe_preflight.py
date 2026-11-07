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

That is the ``copy_if_uncommitted`` rule of the projection writers. A
declaration with ``handover_rule: last_ran`` (effects and the worker, one-runtime
plan T8) replaces rule 1: of the two commits still inside retention, the later
wins, because it belongs to the side that last ran. The successor's own commit
is kept only when it is that later one; otherwise the retired group's commit is
copied over it. Rule 3 applies when neither side holds a commit inside
retention.

Substrates:

* ``EventBusKafka``: two short-lived consumers, one per group, each manually
  assigned with auto-commit off. A manually assigned consumer never joins its
  group, so reading the retired group cannot disturb it, and the successor's
  offsets go in as a simple commit, which the broker accepts while that group
  has no members. It has none: the Kafka bus scopes every subscription to its
  own per-topic group, resolved here by the bus's own
  ``_resolve_effective_group_id`` so the offsets land on the group the consumer
  will join. (The bus module is baselined and only shrinks, so this lives here
  rather than as a bus method.) The consumers come from
  ``create_offset_consumer``: this module builds no raw Kafka client
  of its own, as the imperative contract guard requires.
* ``EventBusInmemory``: keeps no committed offset for any group and delivers a
  record only to the subscribers joined when it is published. No retired group
  on it has a position, so nothing is committed.
* A bus that implements ``ProtocolConsumerGroupOffsetHandover`` itself (the
  in-process group-offset substrate in the tests) is asked directly.

The retired consumer must already be stopped (the fold's deploy retires the
container before it recreates the runtime), and on Kafka the handover confirms
it: before reading any offset it describes the retired group and proceeds only
when the group has no members (the fence, OMN-22714). A member still listed is
waited for up to the declaration's ``fence.wait_seconds``, then refused with
``ConsumerGroupFenceError``, as is a group whose membership cannot be read. A
handover from a consumer still running would copy a position that consumer then
moves past, so a fold would run an effect twice or, after a rollback, lose the
retired side's later commit. The rule holds for every fold, writers included.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal
from uuid import uuid4

from aiokafka import AIOKafkaConsumer
from aiokafka.client import CoordinationType
from aiokafka.protocol.admin import DescribeGroupsRequest
from aiokafka.structs import TopicPartition

from omnibase_core.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.contracts.contract_yaml import read_contract_yaml
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_transport import create_offset_consumer
from omnibase_infra.models.event_bus.model_consumer_group_fence import (
    ModelConsumerGroupFence,
)
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
from omnibase_infra.models.model_fold_handover import ModelFoldHandover
from omnibase_infra.protocols.protocol_consume_concurrency_declarer import (
    ProtocolConsumeConcurrencyDeclarer,
)
from omnibase_infra.protocols.protocol_consumer_group_offset_handover import (
    ProtocolConsumerGroupOffsetHandover,
)
from omnibase_infra.runtime.auto_wiring.models.model_consume_concurrency import (
    ModelConsumeConcurrency,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.runtime_profile import resolve_runtime_folded_profiles

__all__ = [
    "CONSUMER_GROUP_HANDOVER_KEY",
    "ConsumerGroupFenceError",
    "HandlerContractSubscribePreflight",
    "hand_over_committed_offsets",
    "load_contract_handover",
    "resolve_fold_handover",
    "resolve_retired_group",
    "load_consumer_group_handover",
    "plan_offset_handover",
    "supports_offset_handover",
]

logger = logging.getLogger(__name__)

CONSUMER_GROUP_HANDOVER_KEY = "consumer_group_handover"

# DescribeGroups answers (error_code, group_id, state, protocol_type, protocol,
# members[, authorized_operations]) per group in every version aiokafka speaks.
_DESCRIBED_ERROR_CODE = 0
_DESCRIBED_GROUP_ID = 1
_DESCRIBED_MEMBERS = 5


class ConsumerGroupFenceError(RuntimeError):
    """The retired group still has members, or its membership cannot be read.

    Raised before any offset is read or committed, so a refused handover leaves
    both groups exactly as it found them (OMN-22714).
    """


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
    raw = read_contract_yaml(contract_path)
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


FOLD_HANDOVERS_ENV_VAR = "ONEX_RUNTIME_FOLD_HANDOVERS"


def load_contract_handover(
    contract: ModelDiscoveredContract,
) -> ModelConsumerGroupHandover | None:
    """The handover a discovered contract is wired with (OMN-22714).

    The contract's own ``consumer_group_handover`` wins; otherwise the
    fold-wide declaration of the folded profile that owns the contract
    (``ONEX_RUNTIME_FOLD_HANDOVERS``); otherwise none.
    """
    return load_consumer_group_handover(
        contract.contract_path
    ) or resolve_fold_handover(contract.runtime_profiles)


def resolve_fold_handover(
    contract_profiles: Sequence[str],
    environ: Mapping[str, str] | None = None,
) -> ModelConsumerGroupHandover | None:
    """The fold-wide handover of the first folded profile a contract belongs to.

    ``ONEX_RUNTIME_FOLD_HANDOVERS`` is a JSON object, profile to
    ``ModelFoldHandover``, rendered from the lane's runtime policy contract
    (OMN-22714). Unset or blank gives no profile a handover. A contract's own
    ``consumer_group_handover`` is read first by the caller and always wins.

    Raises:
        ValueError: The variable is not a JSON object of valid declarations, or
            names a profile this process does not fold; an unreadable fold
            declaration must not read as absent, which would start the folded
            groups at the bus default.
    """
    env = os.environ if environ is None else environ
    raw = env.get(FOLD_HANDOVERS_ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        declared = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{FOLD_HANDOVERS_ENV_VAR} is not valid JSON: {exc}") from exc
    if not isinstance(declared, dict):
        raise ValueError(f"{FOLD_HANDOVERS_ENV_VAR} must be a JSON object")
    folded = set(resolve_runtime_folded_profiles(env))
    if unfolded := sorted(set(declared) - folded):
        raise ValueError(
            f"{FOLD_HANDOVERS_ENV_VAR} names {unfolded}, which "
            "ONEX_RUNTIME_FOLDED_PROFILES does not fold"
        )
    for profile in contract_profiles:
        if profile in declared:
            return ModelFoldHandover.model_validate(
                declared[profile]
            ).to_contract_handover()
    return None


def plan_offset_handover(
    *,
    partitions: Iterable[int],
    successor_committed: Mapping[int, int | None],
    retired_committed: Mapping[int, int | None],
    first_offsets: Mapping[int, int],
    end_offsets: Mapping[int, int],
    retired_auto_offset_reset: Literal["earliest", "latest"],
    handover_rule: Literal["copy_if_uncommitted", "last_ran"] = "copy_if_uncommitted",
) -> dict[int, int]:
    """Partition to offset for every partition the successor must be given.

    A partition absent from the result is one the successor already owns.
    """
    plan: dict[int, int] = {}
    for partition in sorted(partitions):
        first = first_offsets[partition]
        successor = successor_committed.get(partition)
        retired = retired_committed.get(partition)
        if handover_rule == "last_ran":
            retained = [c for c in (successor, retired) if c is not None and c >= first]
            if retained:
                later = max(retained)
                if successor != later:
                    plan[partition] = later
                continue
        elif successor is not None:
            continue
        elif retired is not None and retired >= first:
            plan[partition] = retired
            continue
        if retired_auto_offset_reset == "earliest":
            plan[partition] = first
        else:
            plan[partition] = end_offsets[partition]
    return plan


def supports_offset_handover(event_bus: object) -> bool:
    """Whether a consumer-group handover can be honoured on ``event_bus``."""
    return isinstance(
        event_bus,
        (ProtocolConsumerGroupOffsetHandover, EventBusKafka, EventBusInmemory),
    )


async def _member_count(consumer: AIOKafkaConsumer, group: str) -> int:
    """Members of ``group`` now, from one DescribeGroups of that group alone.

    Sent to the group's coordinator over the started offset consumer's own
    connection, so the fence builds no client of its own. One group per
    describe, as ``backend_probe`` does: a batched describe has failed to decode
    against MSK.
    """
    client = consumer._client
    coordinator = await client.coordinator_lookup(CoordinationType.GROUP, group)
    response = await client.send(coordinator, DescribeGroupsRequest(groups=[group]))
    for described in getattr(response, "groups", []):
        if (
            len(described) <= _DESCRIBED_MEMBERS
            or str(described[_DESCRIBED_GROUP_ID]) != group
        ):
            continue
        if described[_DESCRIBED_ERROR_CODE] != 0:
            raise ConsumerGroupFenceError(
                f"describing retired consumer group {group!r} returned error "
                f"code {described[_DESCRIBED_ERROR_CODE]}; its membership is "
                "unknown, so its position is not taken over"
            )
        return len(described[_DESCRIBED_MEMBERS])
    raise ConsumerGroupFenceError(
        f"describing retired consumer group {group!r} returned no answer for "
        "it; its membership is unknown, so its position is not taken over"
    )


async def _fence_retired_group(
    consumer: AIOKafkaConsumer, *, group: str, fence: ModelConsumerGroupFence
) -> None:
    """Return once ``group`` has no members; refuse after ``fence.wait_seconds``."""
    deadline = time.monotonic() + fence.wait_seconds
    while True:
        members = await _member_count(consumer, group)
        if members == 0:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ConsumerGroupFenceError(
                f"retired consumer group {group!r} still has {members} "
                f"member(s) after {fence.wait_seconds}s; its consumer is still "
                "running, so its position is not taken over (OMN-22714). Stop "
                "the retired consumer, then restart."
            )
        await asyncio.sleep(min(fence.poll_seconds, remaining))


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
    handover_rule: Literal["copy_if_uncommitted", "last_ran"],
    fence: ModelConsumerGroupFence,
) -> ModelOffsetHandoverResult:
    effective = bus._resolve_effective_group_id(
        to_group, topic, uuid4(), (topic, to_group)
    )
    binding = bus.resolve_bus_binding(topic, consumer_group=effective)
    successor_group = binding.consumer_group or effective
    physical_topic = binding.physical_topic
    config = bus.config.model_copy(update={"bootstrap_servers": bus.bootstrap_servers})

    successor, retired = (
        create_offset_consumer(config, group_id=group)
        for group in (successor_group, from_group)
    )
    await successor.start()
    try:
        await retired.start()
        try:
            # The fence comes first: no offset of a group still consumed is read.
            await _fence_retired_group(retired, group=from_group, fence=fence)
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
                handover_rule=handover_rule,
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
        kept=tuple(
            p
            for p in partition_ids
            if p not in plan and successor_committed[p] is not None
        ),
        fenced_group=from_group,
    )


def resolve_retired_group(
    event_bus: object,
    handover: ModelConsumerGroupHandover,
    *,
    environment: str,
    to_group: str,
    topic: str,
) -> str:
    """The retired consumer group to read for ``topic`` (OMN-22714, T8 part 2).

    A static declaration names one group for every topic. ``from_instance_id``
    names a retired process that ran this same contract, whose group the bus
    scopes per topic and discriminates by instance, so it is derived here by the
    same resolution the successor's group goes through.

    Raises:
        ValueError: The declared instance is the bus's own, which would read
            the successor's commits as the retired side's.
        TypeError: The bus cannot derive a group for another instance.
    """
    if handover.from_instance_id is None:
        return handover.source_group(environment)
    if isinstance(event_bus, EventBusInmemory):
        return to_group
    if not isinstance(event_bus, EventBusKafka):
        raise TypeError(
            f"{type(event_bus).__name__} cannot derive the consumer group of "
            f"retired instance {handover.from_instance_id!r} (OMN-22714)"
        )
    if handover.from_instance_id == event_bus.config.instance_id:
        raise ValueError(
            f"handover from instance {handover.from_instance_id!r} names this "
            "bus's own instance; the retired group would be the successor's "
            "group itself (OMN-22714)"
        )
    # The retired process's group goes through the same resolution as the
    # successor's (topic scope, instance discriminator, length limit), by a bus
    # object that holds this bus's configuration under the retired instance id
    # and is never started.
    retired_bus = EventBusKafka(
        config=event_bus.config.model_copy(
            update={"instance_id": handover.from_instance_id}
        )
    )
    effective = retired_bus._resolve_effective_group_id(
        to_group, topic, uuid4(), (topic, to_group)
    )
    binding = event_bus.resolve_bus_binding(topic, consumer_group=effective)
    return binding.consumer_group or effective


async def hand_over_committed_offsets(
    event_bus: object,
    *,
    topic: str,
    from_group: str,
    to_group: str,
    from_auto_offset_reset: Literal["earliest", "latest"],
    handover_rule: Literal["copy_if_uncommitted", "last_ran"] = "copy_if_uncommitted",
    fence: ModelConsumerGroupFence | None = None,
) -> ModelOffsetHandoverResult:
    """Seed ``to_group`` (the id passed to ``subscribe``) from ``from_group``.

    Raises:
        TypeError: ``event_bus`` cannot hand offsets over; see
            :func:`supports_offset_handover`.
        ConsumerGroupFenceError: On Kafka, ``from_group`` still has members
            after ``fence.wait_seconds``, or its membership cannot be read.
    """
    if isinstance(event_bus, ProtocolConsumerGroupOffsetHandover):
        return await event_bus.hand_over_committed_offsets(
            topic=topic,
            from_group=from_group,
            to_group=to_group,
            from_auto_offset_reset=from_auto_offset_reset,
            handover_rule=handover_rule,
        )
    if isinstance(event_bus, EventBusKafka):
        return await _hand_over_on_kafka(
            event_bus,
            topic=topic,
            from_group=from_group,
            to_group=to_group,
            from_auto_offset_reset=from_auto_offset_reset,
            handover_rule=handover_rule,
            fence=fence if fence is not None else ModelConsumerGroupFence(),
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
        concurrency: ModelConsumeConcurrency,
        handover: ModelConsumerGroupHandover | None,
        event_bus: object,
        environment: str,
        consumer_group: str,
    ) -> None:
        """Validate injected declarations and refuse a bus that cannot honour them.

        Args:
            contract_name: The contract, for messages and logs.
            concurrency: The contract's parsed consume-concurrency declaration.
            handover: The contract's parsed consumer-group handover, if declared.
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
        self._concurrency = concurrency
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
        self._handover = handover
        if self._handover is not None and not supports_offset_handover(event_bus):
            raise TypeError(
                f"contract '{contract_name}' declares consumer_group_handover, "
                f"but the wired event bus ({type(event_bus).__name__}) cannot "
                "hand a retired group's committed offsets over; joining at the "
                "bus default would drop the handover window (OMN-21753)"
            )
        self._environment = environment
        # A static declaration names one retired group for every topic; one by
        # instance has a group per topic, resolved in ``handle`` (OMN-22714).
        self._retired_group = (
            self._handover.source_group(environment)
            if self._handover and self._handover.from_instance_id is None
            else ""
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
            retired_group = (
                resolve_retired_group(
                    self._event_bus,
                    self._handover,
                    environment=self._environment,
                    to_group=self._consumer_group,
                    topic=topic,
                )
                if self._handover.from_instance_id is not None
                else self._retired_group
            )
            handover = await hand_over_committed_offsets(
                self._event_bus,
                topic=topic,
                from_group=retired_group,
                to_group=self._consumer_group,
                from_auto_offset_reset=self._handover.from_auto_offset_reset,
                handover_rule=self._handover.handover_rule,
                fence=self._handover.fence,
            )
            logger.info(
                "Consumer group handover: topic=%s from=%s to=%s committed=%s "
                "kept=%s fenced=%s node=%s (OMN-21753)",
                topic,
                handover.from_group,
                handover.to_group,
                handover.committed,
                list(handover.kept),
                handover.fenced_group,
                self._contract_name,
            )
        return ModelContractSubscribePreflightResult(
            topic=topic, max_in_flight_records=bound, handover=handover
        )
