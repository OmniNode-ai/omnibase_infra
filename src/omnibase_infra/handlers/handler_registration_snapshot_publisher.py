# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registration snapshot producer lifecycle owned by the event-bus effect."""

from __future__ import annotations

from omnibase_core.errors.error_service_resolution import ServiceResolutionError
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.registration.model_registration_snapshot_producer import (
    ModelRegistrationSnapshotProducer,
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
        from aiokafka import AIOKafkaProducer

        from omnibase_infra.event_bus.kafka_auth import (
            build_aiokafka_auth_kwargs_from_env,
        )

        return ModelRegistrationSnapshotProducer(
            producer=AIOKafkaProducer(  # no-contract-check: effect handler owns the producer
                bootstrap_servers=request.kafka_bootstrap_servers,
                **build_aiokafka_auth_kwargs_from_env(),
            )
        )
