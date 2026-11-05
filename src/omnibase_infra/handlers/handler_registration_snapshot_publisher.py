# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registration snapshot producer lifecycle owned by the event-bus effect."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import cast

from omnibase_core.errors.error_service_resolution import ServiceResolutionError
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ContainerWiringError, ModelInfraErrorContext
from omnibase_infra.models.registration.model_registration_snapshot_producer import (
    ModelRegistrationSnapshotProducer,
)
from omnibase_infra.protocols.protocol_registration_snapshot_producer import (
    ProtocolRegistrationSnapshotProducer,
)
from omnibase_infra.runtime.protocol_domain_plugin import ModelDomainPluginConfig


class HandlerRegistrationSnapshotPublisher:
    """Acquire the snapshot producer; the plugin retains start and cleanup policy."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container

    @classmethod
    async def from_container(
        cls, container: ModelONEXContainer
    ) -> HandlerRegistrationSnapshotPublisher:
        """Resolve an injected handler, registering the default bootstrap binding on a miss."""
        registry = container.service_registry
        if registry is None:
            registry = container.initialize_service_registry()
        try:
            return await container.get_service_async(cls)
        except ServiceResolutionError:
            await registry.register_instance(interface=cls, instance=cls(container))
            return await container.get_service_async(cls)

    @staticmethod
    def resources() -> ProtocolRegistrationSnapshotProducer:
        """Resolve the effect provider through its registered public backend port."""
        providers = tuple(
            entry_points(group="onex.backends", name="registration_snapshot_producer")
        )
        if len(providers) != 1:
            raise ContainerWiringError(
                "Expected one onex.backends provider for registration_snapshot_producer",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="resolve_registration_snapshot_backend",
                ),
            )
        return cast("ProtocolRegistrationSnapshotProducer", providers[0].load())

    @property
    def handler_type(self) -> EnumHandlerType:
        """Classify bootstrap resource acquisition as infrastructure handling."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Resource acquisition belongs to the EFFECT layer."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelDomainPluginConfig
    ) -> ModelRegistrationSnapshotProducer:
        """Construct the snapshot producer with the existing Kafka auth boundary."""
        return ModelRegistrationSnapshotProducer(
            producer=self.resources().create(request.kafka_bootstrap_servers)
        )
