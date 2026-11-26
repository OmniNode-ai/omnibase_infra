# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The boot subscription split by demand (OMN-18087, OMN-25856).

Moved out of ``runtime.auto_wiring.handler_wiring``, whose baseline only shrinks.
Under ``gate_required`` a contract that subscribes no command topic stays off the
boot path even when its topic holds a retained record: the demand watch reads that
topic from its high watermark and starts the contract, from ``earliest``, when the
next record arrives.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Sequence
from typing import TYPE_CHECKING, Protocol

from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.models.operational_timeouts import TIMEOUTS

if TYPE_CHECKING:
    from omnibase_infra.event_bus.model_contract_attach_result import (
        ModelContractAttachResult,
    )
    from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract
    from omnibase_infra.runtime.auto_wiring.report import ModelContractWiringResult

logger = logging.getLogger(__name__)

__all__: list[str] = ["ProtocolDemandSource", "split_boot_set_by_demand"]


class ProtocolDemandSource(Protocol):
    """What the split needs of a demand source."""

    async def watch_idle_topics(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
        *,
        watch_retained: Collection[str] = (),
    ) -> tuple[frozenset[str], Callable[[], Awaitable[None]]]: ...


async def split_boot_set_by_demand(
    eligible: list[tuple[ModelContractWiringResult, ModelDiscoveredContract]],
    *,
    demand_source: ProtocolDemandSource,
    start: Callable[
        [ModelContractWiringResult, ModelDiscoveredContract],
        Awaitable[ModelContractAttachResult],
    ],
    exclude: Callable[[str, EnumContractAttachExclusionReason, str], None],
    watch_stops_out: list[Callable[[], Awaitable[None]]] | None,
    priority_contracts: Sequence[str] = (),
    defer_retained: bool = False,
    retry_backoff_seconds: float = 30.0,
) -> list[tuple[ModelContractWiringResult, ModelDiscoveredContract]]:
    """Boot priority contracts and contracts whose topics hold retained records.

    With *defer_retained* (OMN-25856) the retained topics of a contract
    that subscribes no command topic are watched too, so that contract starts on the
    next record instead of joining at boot.

    The rest are watched by one group-less reader, which costs
    no group join, and each is attached by *start* when a record first reaches
    one of its topics. Each start runs as its own task, so a slow join never
    holds up the watch, and a start that does not attach is retried with the
    unattached-contract reconciliation's bounded backoff.
    """
    # Clients can require a live first hop before publishing. Priority consumers
    # therefore take the normal provision->ready->attach path without waiting
    # for demand, and only nonpriority contracts need a first-record watch.
    topics_of = {
        contract.name: tuple(contract.event_bus.subscribe_topics)
        if contract.event_bus is not None
        else ()
        for _, contract in eligible
        if contract.name not in priority_contracts
    }
    contract_of = {contract.name: contract for _, contract in eligible}
    all_topics = {topic for topics in topics_of.values() for topic in topics}
    deferred_by_topic: dict[
        str, list[tuple[ModelContractWiringResult, ModelDiscoveredContract]]
    ] = {}
    started: set[str] = set()
    start_tasks: set[asyncio.Task[None]] = set()

    async def start_until_attached(
        topic: str,
        result: ModelContractWiringResult,
        contract: ModelDiscoveredContract,
    ) -> None:
        for attempt in range(1, TIMEOUTS.not_ready_retry_max_attempts + 1):
            outcome = await start(result, contract)
            if outcome.status is EnumContractAttachStatus.ATTACHED:
                logger.info(
                    "Demand start: contract '%s' attached on its first request "
                    "(topic=%s, topics=%s, attempt=%d) (OMN-18087)",
                    contract.name,
                    topic,
                    list(outcome.topics_subscribed),
                    attempt,
                )
                return
            logger.warning(
                "Demand start: contract '%s' did not attach on its first request "
                "(topic=%s, status=%s, detail=%s, attempt=%d of %d) (OMN-18087)",
                contract.name,
                topic,
                outcome.status.value,
                outcome.detail,
                attempt,
                TIMEOUTS.not_ready_retry_max_attempts,
            )
            await asyncio.sleep(retry_backoff_seconds)

    async def on_first_record(topic: str) -> None:
        for result, contract in deferred_by_topic.get(topic, ()):
            if contract.name in started:
                continue
            started.add(contract.name)
            task = asyncio.create_task(
                start_until_attached(topic, result, contract),
                name=f"demand-start-{contract.name}",
            )
            start_tasks.add(task)
            task.add_done_callback(start_tasks.discard)

    if defer_retained:
        from omnibase_infra.runtime.health.contract_attach_readiness_gate import (
            contract_subscribes_a_command_topic,
        )

        in_use, stop = await demand_source.watch_idle_topics(
            all_topics,
            on_first_record,
            watch_retained={
                topic
                for name, topics in topics_of.items()
                if not contract_subscribes_a_command_topic(contract_of[name])
                for topic in topics
            },
        )
    else:
        in_use, stop = await demand_source.watch_idle_topics(
            all_topics, on_first_record
        )
    if watch_stops_out is not None:
        watch_stops_out.append(stop)
    boot: list[tuple[ModelContractWiringResult, ModelDiscoveredContract]] = []
    for result, contract in eligible:
        topics = topics_of.get(contract.name, ())
        if contract.name in priority_contracts or in_use.intersection(topics):
            boot.append((result, contract))
            continue
        for topic in topics:
            deferred_by_topic.setdefault(topic, []).append((result, contract))
        exclude(
            contract.name,
            EnumContractAttachExclusionReason.DEMAND_DEFERRED,
            f"idle_topics={list(topics)}",
        )
    logger.info(
        "Boot subscription split: %d priority or in-use contract(s) attach at boot, %d "
        "start on their first request; %d of %d topic(s) are in use (hold a "
        "record, or cannot be read here); watched_topics=%s (OMN-18087)",
        len(boot),
        len(eligible) - len(boot),
        len(in_use),
        len(all_topics),
        sorted(deferred_by_topic),
    )
    return boot
