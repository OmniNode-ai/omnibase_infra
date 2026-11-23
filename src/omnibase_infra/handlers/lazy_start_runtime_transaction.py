# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime's start transaction for one dormant contract (OMN-21592).

Implements ``ProtocolLazyStartTransaction`` over the kernel's own public wiring
calls, the same two the boot uses, in the plan's order (section 3.11.5):

* ``build`` runs ``wire_from_manifest`` for the one contract into a dispatch
  engine of the node's own, with no subscription: its handler is imported and
  its dispatchers registered there. No route of the runtime's shared engine and
  no subscription reaches that engine yet, so nothing can be dispatched to it.
* ``attach`` freezes the node's engine and runs ``subscribe_wired_contract_topics``
  for the one contract: provision its topics, confirm their metadata, then attach
  its subscriptions (the per-contract interleave keeps that order). Every
  subscription handle the bus returns is recorded as it is returned; an outcome
  other than attached raises.
* ``rollback`` stops every recorded subscription (and any that completes after)
  and discards the node's engine, so no subscription and no dispatcher is left.
  It never deletes a topic.
* ``deliver`` dispatches the held request to the node's engine in this process;
  the request never rides the node's command topic. What the node returns is
  applied the way the node's own subscription applies it (the contract-derived
  result applier), so the next hop of a chain receives its message on the first
  request, not only on a later one (OMN-26396).

The node's own engine is what makes a rollback complete: the shared engine has
no dispatcher removal, so a dispatcher registered there could not be taken back
after a failed attach.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.protocols.protocol_lazy_start_bus import (
    ProtocolLazyStartBus,
)
from omnibase_infra.protocols.protocol_topic_provisioner import (
    ProtocolTopicProvisioner,
)
from omnibase_infra.runtime.auto_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.report import ModelAutoWiringReport
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
    build_contract_result_applier,
)

if TYPE_CHECKING:
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )

__all__: list[str] = ["LazyStartRuntimeTransaction"]

logger = logging.getLogger(__name__)


class RecordingBus:
    """The runtime bus, recording each subscription handle a subscribe returns.

    It exposes ``wrapped_event_bus`` (OMN-21753): the pre-subscribe
    preflight judges handover and consume-concurrency support on the bus named
    here, because a ``__getattr__`` pass-through is invisible to ``isinstance``.
    """

    def __init__(self, bus: ProtocolLazyStartBus) -> None:
        self._bus = bus
        self.handles: list[Callable[[], Awaitable[object]]] = []
        self.closed = False

    @property
    def wrapped_event_bus(self) -> object:
        """The runtime bus this recorder forwards to."""
        return self._bus

    async def subscribe(self, *args: object, **kwargs: object) -> object:
        handle = await self._bus.subscribe(*args, **kwargs)
        if callable(handle):
            if self.closed:
                # A sibling subscribe finished after the rollback: stop it now.
                await handle()
            else:
                self.handles.append(handle)
        return handle

    async def publish_envelope(self, *args: object, **kwargs: object) -> object:
        return await self._bus.publish_envelope(*args, **kwargs)

    async def publish(self, *args: object, **kwargs: object) -> object:
        return await self._bus.publish(*args, **kwargs)

    def get_consumer_groups(self) -> object:
        return self._bus.get_consumer_groups()

    def __getattr__(self, name: str) -> object:
        return getattr(self._bus, name)


@dataclass
class NodeStart:
    engine: MessageDispatchEngine
    manifest: ModelAutoWiringManifest
    report: ModelAutoWiringReport
    bus: RecordingBus
    applier: DispatchResultApplier | None = None
    dispatcher_ids: tuple[str, ...] = ()
    attach_results: list[ModelContractAttachResult] = field(default_factory=list)


class LazyStartRuntimeTransaction:
    """Build, attach, roll back and deliver for the dormant contracts of one boot."""

    def __init__(
        self,
        contracts: Mapping[str, ModelDiscoveredContract],
        bus: ProtocolLazyStartBus,
        environment: str,
        provisioner: ProtocolTopicProvisioner | None,
        container: object | None,
        readiness_config: ModelTopicReadinessConfig,
        topology: ModelDeploymentTopology | None = None,
    ) -> None:
        self._topology = topology
        self._contracts = dict(contracts)
        self._bus = bus
        self._environment = environment
        self._provisioner = provisioner
        self._container = container
        self._readiness_config = readiness_config
        self._starts: dict[str, NodeStart] = {}

    def node_engine(self, contract_name: str) -> MessageDispatchEngine | None:
        """The started node's own dispatch engine, or None."""
        start = self._starts.get(contract_name)
        return None if start is None else start.engine

    async def build(self, contract_name: str) -> None:
        manifest = ModelAutoWiringManifest(contracts=(self._contracts[contract_name],))
        engine = MessageDispatchEngine()
        report = await wire_from_manifest(
            manifest,
            engine,
            event_bus=self._bus,
            environment=self._environment,
            container=self._container,
            subscribe_immediately=False,
            topology=self._topology,
        )
        if report.total_wired != 1:
            reasons = [f"{r.outcome.value} {r.reason}" for r in report.results]
            msg = f"contract '{contract_name}' did not wire: {reasons}"
            raise RuntimeError(msg)
        self._starts[contract_name] = NodeStart(
            engine=engine,
            manifest=manifest,
            report=report,
            bus=RecordingBus(self._bus),
            applier=self._applier_of(self._contracts[contract_name]),
            dispatcher_ids=tuple(report.results[0].dispatchers_registered),
        )

    def _applier_of(
        self, contract: ModelDiscoveredContract
    ) -> DispatchResultApplier | None:
        """The contract-derived applier the node's own subscription would build."""
        if contract.event_bus is None or not contract.event_bus.publish_topics:
            return None
        return build_contract_result_applier(
            event_bus=cast("ProtocolEventBusLike", self._bus),
            contract_path=contract.contract_path,
            publish_topics=contract.event_bus.publish_topics,
            terminal_event=contract.terminal_event,
            output_topic_map=load_published_events_map(contract.contract_path),
            allowed_output_topics=contract.event_bus.publish_topics,
        )

    async def attach(self, contract_name: str) -> None:
        start = self._starts[contract_name]
        start.engine.freeze()
        await subscribe_wired_contract_topics(
            manifest=start.manifest,
            report=start.report,
            dispatch_engine=start.engine,
            event_bus=start.bus,
            environment=self._environment,
            provisioner=self._provisioner,
            readiness_config=self._readiness_config,
            attach_results_out=start.attach_results,
        )
        outcome = [r.status for r in start.attach_results]
        if outcome != [EnumContractAttachStatus.ATTACHED]:
            details = [r.detail for r in start.attach_results]
            msg = f"contract '{contract_name}' did not attach: {outcome} {details}"
            raise RuntimeError(msg)

    async def rollback(self, contract_name: str) -> None:
        start = self._starts.pop(contract_name, None)
        if start is None:
            return
        start.bus.closed = True
        for handle in start.bus.handles:
            try:
                await handle()
            except Exception:
                logger.exception(
                    "Lazy start rollback: a subscription of '%s' did not stop",
                    contract_name,
                )

    async def deliver(self, contract_name: str, topic: str, envelope: object) -> object:
        start = self._starts[contract_name]
        typed = (
            envelope
            if isinstance(envelope, ModelEventEnvelope)
            else ModelEventEnvelope[object].model_validate(envelope)
        )
        result = await start.engine.dispatch(
            topic, typed, allowed_dispatcher_ids=start.dispatcher_ids
        )
        if start.applier is not None and result is not None:
            with bind_dispatch_envelope(typed):
                await start.applier.apply(result, typed.correlation_id)
        return result
