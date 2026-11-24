# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The always-available ingress node's handler: the lazy-start front door (OMN-21592).

Bound from ``nodes/node_lazy_start_ingress/contract.yaml``. The ingress node is
in every lazy-start lane's boot set (the overlay's ``lazy_start.ingress_contract``),
so its command topic is consumed from boot. A request on it names a dormant
contract and carries the envelope for it; this handler starts the contract on
that first request through ``LazyStartFrontDoor`` over the runtime's start
transaction, and delivers the envelope to the node's own dispatcher. The
dormant contracts, the policy and the topic provisioner are the ones the
kernel's boot recorded (``StoreLazyStartDormant``); the bus is the runtime's
subscriber bus from the container.

The handler is also where a bus message wakes a dormant consumer (OMN-26396): it
puts the front door in front of the runtime bus's publish, so a message published
to the topic of a dormant contract starts that contract (and its closure) before
the message reaches the topic. A handler never touches the bus; the gate is the
runtime's own publish path, installed once by this always-wired node.

Two requests never reach either of those (OMN-26316): one on a dormant contract's
own HTTP mount, and a command a client writes straight to the broker. This node
records the front door's ``wake`` for the HTTP listener's mount middleware, and
arms a group-less watch on the topics dormant contracts subscribe to, so the first
message to one of them wakes its consumers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import cast

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_lazy_start_outcome import EnumLazyStartOutcome
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.handlers.lazy_start_boot_split import (
    StoreLazyStartDormant,
    StoreLazyStartWaker,
)
from omnibase_infra.handlers.lazy_start_dormant_topic_watch import (
    LazyStartDormantTopicWatch,
)
from omnibase_infra.handlers.lazy_start_front_door import LazyStartFrontDoor
from omnibase_infra.handlers.lazy_start_publish_gate import install_publish_gate
from omnibase_infra.handlers.lazy_start_runtime_transaction import (
    LazyStartRuntimeTransaction,
)
from omnibase_infra.models.lazy_start import (
    ModelLazyStartRequest,
    ModelLazyStartResult,
)
from omnibase_infra.protocols.protocol_lazy_start_bus import (
    ProtocolLazyStartBus,
)
from omnibase_infra.protocols.protocol_topic_provisioner import (
    ProtocolTopicProvisioner,
)

__all__: list[str] = ["HandlerLazyStartIngress"]

logger = logging.getLogger(__name__)


class HandlerLazyStartIngress:
    """Consume first requests for dormant contracts and start them."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container
        self._bus: object | None = None
        self._front_door: LazyStartFrontDoor | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._watch_stop: Callable[[], Awaitable[None]] | None = None
        StoreLazyStartWaker.record(self.wake_contract)
        try:
            self._front_door_of_this_boot()
        except Exception:
            # The bus may not be registered yet; the first request installs the gate.
            logger.exception(
                "Lazy start ingress: wake-on-publish gate not installed at wiring "
                "(OMN-26396)"
            )

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _front_door_of_this_boot(self) -> LazyStartFrontDoor | None:
        policy = StoreLazyStartDormant.policy()
        if self._front_door is None and policy is not None:
            if self._bus is None:
                # The registry uses the interface as a key, never as a constructor.
                # Resolved on first use: a boot with no policy needs no bus.
                key: type[object] = ProtocolEventBusSubscriber
                self._bus = self._container.get_service(key)
            contracts = StoreLazyStartDormant.contracts()
            provisioner = StoreLazyStartDormant.provisioner()
            self._front_door = LazyStartFrontDoor(
                LazyStartRuntimeTransaction(
                    contracts,
                    cast("ProtocolLazyStartBus", self._bus),
                    str(getattr(self._bus, "environment", "")),
                    provisioner
                    if isinstance(provisioner, ProtocolTopicProvisioner)
                    else None,
                    self._container,
                    ModelTopicReadinessConfig(),
                    topology=StoreLazyStartDormant.topology(),
                    handler_dependencies=StoreLazyStartDormant.handler_dependencies(),
                ),
                contracts,
                policy,
                StoreLazyStartDormant.wired_contracts(),
            )
            StoreLazyStartDormant.record_front_door(self._front_door)
            if not install_publish_gate(self._bus, self._front_door.wake_for_topic):
                logger.warning(
                    "Lazy start ingress: this bus has no publish, so a message to a "
                    "dormant consumer's topic does not wake it (OMN-26396)"
                )
            self._arm_topic_watch(self._front_door)
        return self._front_door

    def _arm_topic_watch(self, front_door: LazyStartFrontDoor) -> None:
        """Wake a dormant consumer on the first message a client writes to its topic."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "Lazy start ingress: no running loop at wiring, so a message a "
                "client writes straight to the broker does not wake a dormant "
                "consumer (OMN-26316)"
            )
            return
        self._watch_task = loop.create_task(
            self._watch_dormant_topics(front_door), name="lazy-start-arm-topic-watch"
        )

    async def _watch_dormant_topics(self, front_door: LazyStartFrontDoor) -> None:
        try:
            watch = LazyStartDormantTopicWatch(self._container)
            policy = StoreLazyStartDormant.policy()
            if policy is None:
                return
            self._watch_stop = await watch.watch(
                front_door.dormant_topics,
                front_door.wake_for_topic,
                policy.topic_request_chunk,
            )
        except TypeError:
            logger.info(
                "Lazy start ingress: this bus is not Kafka, so there is no broker "
                "to watch for a first message (OMN-26316)"
            )
        except Exception:
            logger.exception(
                "Lazy start ingress: the dormant-topic watch did not start; a "
                "message a client writes straight to the broker does not wake a "
                "dormant consumer (OMN-26316)"
            )

    async def wake_contract(self, name: str) -> None:
        """Wake one dormant contract's closure for a request on its HTTP mount."""
        front_door = self._front_door_of_this_boot()
        if front_door is None:
            msg = "this runtime booted with no lazy-start policy"
            raise RuntimeError(msg)
        await front_door.wake(name)

    async def shutdown(self) -> None:
        """Stop the dormant-topic watch and forget the recorded wake."""
        StoreLazyStartWaker.record(None)
        if self._watch_task is not None:
            await asyncio.gather(self._watch_task, return_exceptions=True)
        if self._watch_stop is not None:
            await self._watch_stop()
            self._watch_stop = None

    async def handle(self, request: ModelLazyStartRequest) -> ModelLazyStartResult:
        front_door = self._front_door_of_this_boot()
        if front_door is None:
            return ModelLazyStartResult(
                target_contract=request.target_contract,
                correlation_id=request.correlation_id,
                outcome=EnumLazyStartOutcome.CONTRACT_NOT_ANNOUNCED,
                detail="this runtime booted with no lazy-start policy",
            )
        return await front_door.handle(request)
