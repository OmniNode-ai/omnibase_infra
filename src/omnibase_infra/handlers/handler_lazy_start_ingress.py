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
"""

from __future__ import annotations

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
from omnibase_infra.handlers.lazy_start_boot_split import StoreLazyStartDormant
from omnibase_infra.handlers.lazy_start_front_door import LazyStartFrontDoor
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


class HandlerLazyStartIngress:
    """Consume first requests for dormant contracts and start them."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container
        self._bus: object | None = None
        self._front_door: LazyStartFrontDoor | None = None

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
                ),
                contracts,
                policy,
            )
        return self._front_door

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
