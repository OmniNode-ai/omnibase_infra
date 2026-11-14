# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A contract that is demand-deferred costs nothing at boot (OMN-21318).

The boot subscription split (OMN-18087) already kept an idle contract's consumer
groups from joining at boot, but only after the contract's handlers had been
imported, built and registered. Measured on the dev runtime on 2026-11-14:
``wired=932`` contracts took 22.96 s to wire for 158 attached.

This module moves the split one step earlier. Before ``wire_from_manifest``
runs, the broker is asked which of the deferrable contracts' subscribe topics
are in use; a contract with none is taken out of the manifest that is wired, so
its handler modules are never imported and nothing is registered for it. After
the dispatch engine froze, one group-less reader waits on its topics. The first
record wires the contract on the frozen engine through the same preparation and
commit the boot uses, then runs the provision -> confirm-ready -> attach
interleave from ``earliest``: the record that started the contract is the first
one its groups read, so it is handled once.

This is the runtime side of ``node_boot_wiring_deferral_compute``: the node's
handler decides which contracts to defer from the broker's reading, and this
module takes that reading, wires and attaches. It reaches the wiring engine
through the module, never by copied names, so a test that patches the engine's
seams patches this path too.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Collection, Mapping
from typing import TYPE_CHECKING

from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.handlers.boot_topic_provisioning import (
    provision_unattached_publish_topics,
)
from omnibase_infra.models.model_contract_hotload_lane_overlay import (
    lane_defers_topics_to_first_request,
)
from omnibase_infra.models.operational_timeouts import TIMEOUTS
from omnibase_infra.nodes.node_boot_wiring_deferral_compute.handler import (
    decide_deferred_contracts,
)
from omnibase_infra.nodes.node_boot_wiring_deferral_compute.models import (
    ModelBootWiringCandidate,
    ModelBootWiringDeferralRequest,
)
from omnibase_infra.runtime.auto_wiring import handler_wiring as _wiring
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )
    from omnibase_infra.event_bus.model_topic_readiness_config import (
        ModelTopicReadinessConfig,
    )
    from omnibase_infra.handlers.handler_kafka_topic_demand_source import (
        HandlerKafkaTopicDemandSource,
    )
    from omnibase_infra.protocols.protocol_dispatch_result_applier import (
        ProtocolDispatchResultApplier,
    )
    from omnibase_infra.protocols.protocol_topic_provisioner import (
        ProtocolTopicProvisioner,
    )

logger = logging.getLogger(__name__)

DEFERRED_WIRING_REASON = (
    "handlers neither imported nor registered at boot; wired when the "
    "contract's first request arrives (OMN-21318)"
)

DeferredContractWirer = Callable[
    [ModelDiscoveredContract], Awaitable[ModelContractWiringResult]
]


def select_demand_deferrable_contracts(
    manifest: ModelAutoWiringManifest,
    priority_contracts: Collection[str] = (),
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier]
    | None = None,
) -> tuple[ModelDiscoveredContract, ...]:
    """The contracts whose handlers may be left unwired until first demand.

    Decided from the contract alone, before any handler is imported: it declares
    handlers and subscribe topics, no plugin owns its subscription, a raw audit
    or projection consumer has the result applier the kernel opts in per contract
    (without one it never attaches, so there is nothing to defer), and it is not
    a boot priority contract (a client may publish to a priority consumer before
    its first record exists, OMN-18087).
    """
    return tuple(
        contract
        for contract in manifest.contracts
        if contract.handler_routing is not None
        and contract.event_bus is not None
        and contract.event_bus.subscribe_topics
        and not contract.event_bus.plugin_managed
        and (
            not _wiring._is_raw_event_projection_contract(contract)
            or _wiring._raw_event_projection_enabled(
                contract, result_appliers_by_contract
            )
        )
        and contract.name not in priority_contracts
    )


class _FirstRecordRelay:
    """Holds a first-record arrival until the dispatch engine froze and a start is bound.

    The broker is measured before wiring, so the demand watch starts before the
    engine can take a registration. A record that lands in between is kept here
    and replayed once ``bind`` hands over the start.
    """

    def __init__(self) -> None:
        self._start: Callable[[str], Awaitable[None]] | None = None
        self._held: list[str] = []

    async def on_first_record(self, topic: str) -> None:
        if self._start is None:
            self._held.append(topic)
            return
        await self._start(topic)

    async def bind(self, start: Callable[[str], Awaitable[None]]) -> None:
        self._start = start
        held, self._held = self._held, []
        for topic in held:
            await start(topic)


class BootWiringDeferral:
    """What boot decided to leave unwired, and the means to start each on demand."""

    def __init__(
        self,
        *,
        manifest: ModelAutoWiringManifest,
        deferred: tuple[ModelDiscoveredContract, ...],
        relay: _FirstRecordRelay,
        stop: Callable[[], Awaitable[None]],
        wire: DeferredContractWirer | None,
    ) -> None:
        names = frozenset(contract.name for contract in deferred)
        self._deferred = deferred
        self._relay = relay
        self._stop = stop
        self._wire = wire
        self.deferred_names = names
        self.wired_manifest = manifest.model_copy(
            update={
                "contracts": tuple(c for c in manifest.contracts if c.name not in names)
            }
        )

    def complete_report(self, report: ModelAutoWiringReport) -> ModelAutoWiringReport:
        """The wiring report with one row per deferred contract, so it stays total.

        A deferred row is ``SKIPPED`` with ``DEFERRED_WIRING_REASON``: the
        roster partition, the runtime manifest and the identity checks account
        for every contract of the full manifest, and a reader sees why.
        """
        rows = tuple(
            ModelContractWiringResult(
                contract_name=contract.name,
                package_name=contract.package_name,
                outcome=EnumWiringOutcome.SKIPPED,
                reason=DEFERRED_WIRING_REASON,
            )
            for contract in self._deferred
        )
        return report.model_copy(update={"results": (*report.results, *rows)})

    async def watch(
        self,
        *,
        dispatch_engine: object,
        event_bus: object,
        environment: str,
        provisioner: ProtocolTopicProvisioner | None,
        readiness_config: ModelTopicReadinessConfig,
        result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier] | None,
        exclusions_out: list[ModelContractAttachExclusion],
        stops_out: list[Callable[[], Awaitable[None]]],
    ) -> None:
        """Report every deferred contract as demand-deferred and arm its first-request start.

        Called after the dispatch engine froze and the booted contracts
        attached. The watch started at planning time keeps its stop in
        *stops_out*; the relay now hands each held and later arrival to the
        start of the contract(s) on that topic.
        """
        stops_out.append(self._stop)
        if not self._deferred:
            return
        topics_of = {
            contract.name: tuple(
                contract.event_bus.subscribe_topics if contract.event_bus else ()
            )
            for contract in self._deferred
        }
        for contract in self._deferred:
            exclusions_out.append(
                ModelContractAttachExclusion(
                    contract_name=contract.name,
                    reason=EnumContractAttachExclusionReason.DEMAND_DEFERRED,
                    detail=f"idle_topics={list(topics_of[contract.name])}",
                )
            )
        # OMN-23951: on a lane whose hot-load mode is not on_demand the publish
        # topics of a deferred contract are created at boot; on_demand leaves
        # them to its first request (OMN-23953).
        scheduled = (
            self.deferred_names
            if lane_defers_topics_to_first_request()
            else frozenset()
        )
        await provision_unattached_publish_topics(
            self.wired_manifest.model_copy(update={"contracts": self._deferred}),
            scheduled,
            provisioner,
            max_concurrent=readiness_config.max_concurrent_contract_attach,
        )
        if self._wire is None:
            return
        semaphore = asyncio.Semaphore(readiness_config.max_concurrent_contract_attach)
        wired: dict[str, ModelContractWiringResult] = {}
        started: set[str] = set()
        tasks: set[asyncio.Task[None]] = set()
        by_topic: dict[str, list[ModelDiscoveredContract]] = {}
        for contract in self._deferred:
            for topic in topics_of[contract.name]:
                by_topic.setdefault(topic, []).append(contract)

        async def attach(contract: ModelDiscoveredContract) -> bool:
            """Wire once, then attach from earliest. True: attached, or nothing to retry."""
            assert self._wire is not None
            wiring = wired.get(contract.name)
            if wiring is None:
                wiring = await self._wire(contract)
                wired[contract.name] = wiring
            ineligible = _first_demand_ineligibility(wiring)
            if ineligible is not None:
                logger.warning(
                    "Demand start: contract '%s' wired on its first request but "
                    "is not attached: %s (OMN-21318)",
                    contract.name,
                    ineligible,
                )
                return True
            async with semaphore:
                outcome = await _wiring._interleave_contract(
                    name=contract.name,
                    contract=contract,
                    dispatch_engine=dispatch_engine,
                    event_bus=event_bus,
                    environment=environment,
                    result_applier=(result_appliers_by_contract or {}).get(
                        contract.name
                    ),
                    allowed_dispatcher_ids=wiring.dispatchers_registered,
                    provisioner=provisioner,
                    readiness_config=readiness_config,
                    auto_offset_reset="earliest",
                )
            return await _report_attach(contract, outcome)

        async def start_until_attached(contract: ModelDiscoveredContract) -> None:
            for _attempt in range(TIMEOUTS.not_ready_retry_max_attempts):
                if await attach(contract):
                    return
                await asyncio.sleep(_wiring.DEFAULT_NOT_READY_RETRY_BACKOFF_SECONDS)

        def _finish_start(done: asyncio.Task[None]) -> None:
            """Forget a finished start; a start that raised is logged, never lost."""
            tasks.discard(done)
            if not done.cancelled() and done.exception() is not None:
                logger.error(
                    "Demand start: task '%s' ended with an exception (OMN-21318)",
                    done.get_name(),
                    exc_info=done.exception(),
                )

        async def start(topic: str) -> None:
            for contract in by_topic.get(topic, ()):
                if contract.name in started:
                    continue
                started.add(contract.name)
                task = asyncio.create_task(
                    start_until_attached(contract),
                    name=f"demand-start-{contract.name}",
                )
                tasks.add(task)
                task.add_done_callback(_finish_start)

        await self._relay.bind(start)
        logger.info(
            "Boot subscription split: %d deferred contract(s) wire and attach on "
            "their first request; watched_topics=%d (OMN-21318)",
            len(self._deferred),
            len(by_topic),
        )


def _first_demand_ineligibility(result: ModelContractWiringResult) -> str | None:
    """Why a contract wired on its first request must not attach, or None.

    The same structural filters ``subscribe_wired_contract_topics`` applies to a
    contract wired at boot, read from the row its first-request wiring produced.
    """
    if result.outcome is not EnumWiringOutcome.WIRED:
        return f"outcome={result.outcome.value} reason={result.reason}"
    if not result.dispatchers_registered:
        return "it owns zero dispatchers (OMN-15474)"
    if result.has_no_live_dispatcher:
        return "every handler entry wires a no-op dispatch (OMN-17562)"
    return None


async def _report_attach(
    contract: ModelDiscoveredContract, outcome: ModelContractAttachResult
) -> bool:
    """Log one attach attempt; True when the contract attached and needs no retry."""
    if outcome.status is EnumContractAttachStatus.ATTACHED:
        logger.info(
            "Demand start: contract '%s' wired and attached on its first request "
            "(topics=%s) (OMN-21318)",
            contract.name,
            list(outcome.topics_subscribed),
        )
        return True
    logger.warning(
        "Demand start: contract '%s' did not attach on its first request "
        "(status=%s, detail=%s) (OMN-21318)",
        contract.name,
        outcome.status.value,
        outcome.detail,
    )
    return False


def make_deferred_contract_wirer(
    *,
    manifest: ModelAutoWiringManifest,
    dispatch_engine: object,
    event_bus: object | None,
    environment: str,
    container: object | None,
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier] | None,
    materialized_explicit_dependencies: dict[str, dict[str, object]] | None,
    topology: ModelDeploymentTopology | None,
) -> DeferredContractWirer:
    """Wire one deferred contract on the frozen engine, exactly as boot would have.

    The same preparation and commit ``wire_from_manifest`` runs for each
    contract: handler preparation through the resolver (the boot manifest's names
    are the local owners), container pre-resolution (OMN-9410), the registration
    ID preflight and the commit, with the frozen-engine authorization the
    dynamic listener already uses. A preparation that fails is a ``FAILED`` row,
    never an exception into the running runtime.
    """
    owners = frozenset(contract.name for contract in manifest.contracts)

    async def wire(contract: ModelDiscoveredContract) -> ModelContractWiringResult:
        pre_resolved: dict[str, object] = {}
        if container is not None and contract.handler_routing is not None:
            for entry in contract.handler_routing.handlers:
                try:
                    instance = await _wiring._async_resolve_from_container(
                        container,
                        _wiring._import_handler_class(
                            entry.handler.module, entry.handler.name
                        ),
                    )
                except Exception:
                    # The same import failure surfaces again in preparation,
                    # which turns it into a FAILED row; log it here and go on.
                    logger.exception(
                        "Demand start: pre-resolving a handler of '%s' failed "
                        "(OMN-21318)",
                        contract.name,
                    )
                    continue
                if instance is not None:
                    pre_resolved[_wiring._derive_handler_entry_key(entry)] = instance
        try:
            prepared = _wiring._prepare_contract_wiring(
                contract=contract,
                dispatch_engine=dispatch_engine,
                resolver=ServiceHandlerResolver(),
                ownership_query=ServiceLocalHandlerOwnershipQuery(
                    local_node_names=owners
                ),
                event_bus=event_bus,
                environment=environment,
                container=container,
                pre_resolved_handlers=pre_resolved if container is not None else None,
                result_appliers_by_contract=result_appliers_by_contract,
                materialized_explicit_dependencies=materialized_explicit_dependencies,
                topology=topology,
                secret_resolver=await _wiring.build_topology_secret_resolver(container),
            )
        except Exception as exc:
            logger.exception(
                "Demand start: wiring contract '%s' raised %s (OMN-21318)",
                contract.name,
                type(exc).__name__,
            )
            return ModelContractWiringResult(
                contract_name=contract.name,
                package_name=contract.package_name,
                outcome=EnumWiringOutcome.FAILED,
                reason=f"{type(exc).__name__}: {exc}",
            )
        _wiring._preflight_prepared_registration_ids(
            (prepared,), dispatch_engine, dynamic_materialization_authorized=True
        )
        return await _wiring._commit_contract_wiring(
            prepared,
            dispatch_engine,
            event_bus,
            subscribe_immediately=False,
            result_applier=(result_appliers_by_contract or {}).get(contract.name),
            dynamic_materialization_authorized=True,
        )

    return wire


async def plan_boot_wiring_deferral(
    manifest: ModelAutoWiringManifest,
    *,
    demand_source: HandlerKafkaTopicDemandSource | None,
    priority_contracts: Collection[str],
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier] | None,
    wire: DeferredContractWirer | None,
) -> BootWiringDeferral:
    """Decide, before wiring, which contracts boot leaves unwired.

    With no *demand_source* (the policy is not ``in_use``, the bus is not Kafka,
    or core-runtime topics are allowlisted) nothing is deferred. Otherwise the
    broker is measured once and watched from that moment: a deferrable contract
    none of whose subscribe topics is in use is deferred. An unmeasurable broker
    reports every topic in use, so nothing is deferred.
    """
    relay = _FirstRecordRelay()
    candidates = (
        select_demand_deferrable_contracts(
            manifest, priority_contracts, result_appliers_by_contract
        )
        if demand_source is not None
        else ()
    )
    topics = {
        topic
        for contract in candidates
        if contract.event_bus is not None
        for topic in contract.event_bus.subscribe_topics
    }
    if demand_source is None or not topics:
        return BootWiringDeferral(
            manifest=manifest,
            deferred=(),
            relay=relay,
            stop=_nothing_to_stop,
            wire=wire,
        )
    in_use, stop = await demand_source.watch_idle_topics(topics, relay.on_first_record)
    decided = decide_deferred_contracts(
        ModelBootWiringDeferralRequest(
            candidates=tuple(
                ModelBootWiringCandidate(
                    contract_name=contract.name,
                    subscribe_topics=tuple(contract.event_bus.subscribe_topics),
                )
                for contract in candidates
                if contract.event_bus is not None
            ),
            in_use_topics=in_use,
        )
    )
    deferred = tuple(
        contract
        for contract in candidates
        if contract.name in decided.deferred_contract_names
    )
    logger.info(
        "Boot wiring split: %d of %d contract(s) wire at boot, %d wire on their "
        "first request; %d of %d topic(s) are in use (OMN-21318)",
        len(manifest.contracts) - len(deferred),
        len(manifest.contracts),
        len(deferred),
        len(in_use),
        len(topics),
    )
    return BootWiringDeferral(
        manifest=manifest,
        deferred=deferred,
        relay=relay,
        stop=stop,
        wire=wire,
    )


async def plan_runtime_boot_wiring_deferral(
    manifest: ModelAutoWiringManifest,
    *,
    policy: str,
    event_bus: object | None,
    container: ModelONEXContainer,
    priority_contracts: Collection[str],
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier] | None,
    dispatch_engine: object,
    environment: str,
    materialized_explicit_dependencies: dict[str, dict[str, object]] | None,
    topology: ModelDeploymentTopology | None,
) -> BootWiringDeferral:
    """The kernel's entry: defer only when the policy is ``in_use`` on a Kafka bus.

    The policy is the runtime config contract's ``boot_subscriptions.policy``.
    The in-memory bus has no broker to ask, so it wires and joins every contract
    as before; so does a lane that allowlists core-runtime topics
    (``ONEX_CORE_RUNTIME_TOPICS``, empty by default), because the core loop builds
    its routes from handlers that are already wired.
    """
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.handlers.handler_kafka_topic_demand_source import (
        HandlerKafkaTopicDemandSource,
    )
    from omnibase_infra.runtime.core_runtime.composition import (
        parse_core_runtime_topics,
    )

    deferring = (
        policy == "in_use"
        and isinstance(event_bus, EventBusKafka)
        and not parse_core_runtime_topics(os.environ)
    )
    logger.info(
        "Boot subscription policy=%s demand_start=%s (OMN-18087, OMN-21318)",
        policy,
        deferring,
    )
    return await plan_boot_wiring_deferral(
        manifest,
        demand_source=HandlerKafkaTopicDemandSource(container) if deferring else None,
        priority_contracts=priority_contracts,
        result_appliers_by_contract=result_appliers_by_contract,
        wire=make_deferred_contract_wirer(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment=environment,
            container=container,
            result_appliers_by_contract=result_appliers_by_contract,
            materialized_explicit_dependencies=materialized_explicit_dependencies,
            topology=topology,
        ),
    )


async def _nothing_to_stop() -> None:
    """The stop of a plan that left no watch running."""
    return


__all__: list[str] = [
    "DEFERRED_WIRING_REASON",
    "BootWiringDeferral",
    "DeferredContractWirer",
    "make_deferred_contract_wirer",
    "plan_boot_wiring_deferral",
    "plan_runtime_boot_wiring_deferral",
    "select_demand_deferrable_contracts",
]
