# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Service registration operations mixin.

This mixin provides service registration and deregistration operations
for HandlerConsul, extracted to reduce class complexity.

Operations:
    - consul.register: Register service with Consul agent
    - consul.deregister: Deregister service from Consul agent

Event Bus Integration:
    When the payload contains an 'event_bus_config' field, this mixin will:
    1. Store the event bus configuration in Consul KV
    2. Update the topic -> node_id reverse index for routing
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar
from uuid import UUID

T = TypeVar("T")

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.constants_topic_patterns import TOPIC_NAME_PATTERN
from omnibase_infra.enums import EnumInfraTransportType, EnumMessageCategory
from omnibase_infra.errors import (
    InfraConsulError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.handlers.models.consul import (
    ConsulPayload,
    ModelConsulDeregisterPayload,
    ModelConsulRegisterPayload,
)
from omnibase_infra.handlers.models.model_consul_handler_response import (
    ModelConsulHandlerResponse,
)
from omnibase_infra.models.registration import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)

if TYPE_CHECKING:
    import consul as consul_lib

logger = logging.getLogger(__name__)


class ProtocolConsulServiceDependencies(Protocol):
    """Protocol defining required dependencies for service operations.

    HandlerConsul must provide these attributes/methods for the mixin to work.
    """

    _client: consul_lib.Consul | None
    _config: object | None

    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        correlation_id: UUID,
    ) -> T:
        """Execute operation with retry logic."""
        ...

    def _build_response(
        self,
        typed_payload: ConsulPayload,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Build standardized response."""
        ...

    async def _store_node_event_bus(
        self,
        node_id: str,
        event_bus: ModelNodeEventBusConfig,
        correlation_id: UUID,
    ) -> None:
        """Store event_bus config in Consul KV - provided by MixinConsulTopicIndex."""
        ...

    async def _update_topic_index(
        self,
        node_id: str,
        event_bus: ModelNodeEventBusConfig,
        correlation_id: UUID,
    ) -> None:
        """Update topic index - provided by MixinConsulTopicIndex."""
        ...


class MixinConsulService:
    """Mixin providing Consul service registration operations.

    This mixin extracts service operations from HandlerConsul to reduce
    class complexity while maintaining full functionality.

    Required Dependencies (from host class):
        - _client: consul.Consul client instance
        - _config: Handler configuration
        - _execute_with_retry: Retry execution method
        - _build_response: Response builder method
    """

    # Instance attribute declarations for type checking
    _client: consul_lib.Consul | None
    _config: object | None

    # Methods from host class (abstract stubs for type checking)
    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        correlation_id: UUID,
    ) -> T:
        """Execute operation with retry logic - provided by host class."""
        raise NotImplementedError("Must be provided by implementing class")  # type: ignore[return-value]

    def _build_response(
        self,
        typed_payload: ConsulPayload,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Build standardized response - provided by host class."""
        raise NotImplementedError("Must be provided by implementing class")  # type: ignore[return-value]

    async def _store_node_event_bus(
        self,
        node_id: str,
        event_bus: ModelNodeEventBusConfig,
        correlation_id: UUID,
    ) -> None:
        """Store event_bus config - provided by MixinConsulTopicIndex."""
        raise NotImplementedError("Must be provided by implementing class")

    async def _update_topic_index(
        self,
        node_id: str,
        event_bus: ModelNodeEventBusConfig,
        correlation_id: UUID,
    ) -> None:
        """Update topic index - provided by MixinConsulTopicIndex."""
        raise NotImplementedError("Must be provided by implementing class")

    def _validate_topic_entry(
        self,
        entry: dict[str, object],
        location: str,
        correlation_id: UUID,
    ) -> tuple[str, str]:
        """Validate a single topic entry and return sanitized values.

        Validates:
        1. Topic is a non-empty string after stripping whitespace
        2. Topic format matches TOPIC_NAME_PATTERN (alphanumeric, dots, underscores, hyphens)
        3. message_category is a valid EnumMessageCategory value

        Args:
            entry: The full topic entry dict containing 'topic' and optional 'message_category'.
            location: Location string for error messages (e.g., "subscribe_topics[0]").
            correlation_id: Correlation ID for tracing.

        Returns:
            Tuple of (stripped_topic, validated_message_category).

        Raises:
            ProtocolConfigurationError: If validation fails.
        """
        raw_topic = entry.get("topic")

        # Validate topic is a non-empty string BEFORE any coercion
        if not isinstance(raw_topic, str) or not raw_topic.strip():
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="parse_event_bus_config",
            )
            raise ProtocolConfigurationError(
                f"Invalid or missing 'topic' in {location}: "
                f"expected non-empty string, got {type(raw_topic).__name__}",
                context=ctx,
                parameter=f"{location}.topic",
                value=str(raw_topic) if raw_topic is not None else "None",
            )

        stripped_topic = raw_topic.strip()

        # Validate topic format (fail fast before storage/indexing)
        if not TOPIC_NAME_PATTERN.match(stripped_topic):
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="parse_event_bus_config",
            )
            raise ProtocolConfigurationError(
                f"Topic '{stripped_topic}' in {location} "
                "contains invalid characters. Only alphanumeric characters, periods (.), "
                "underscores (_), and hyphens (-) are allowed.",
                context=ctx,
                parameter=f"{location}.topic",
                value=stripped_topic,
            )

        # Validate message_category if provided
        raw_category = entry.get("message_category", "EVENT")
        if isinstance(raw_category, str):
            category_upper = raw_category.upper()
        else:
            category_upper = str(raw_category).upper()

        # Valid categories: EVENT, COMMAND, INTENT (case-insensitive)
        valid_categories = {
            cat.value.upper(): cat.value.upper() for cat in EnumMessageCategory
        }
        if category_upper not in valid_categories:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="parse_event_bus_config",
            )
            raise ProtocolConfigurationError(
                f"Invalid 'message_category' in {location}: "
                f"'{raw_category}'. Valid values are: {', '.join(sorted(valid_categories.keys()))}",
                context=ctx,
                parameter=f"{location}.message_category",
                value=str(raw_category),
            )

        return stripped_topic, category_upper

    def _parse_event_bus_config(
        self,
        event_bus_data: dict[str, object],
        correlation_id: UUID,
    ) -> ModelNodeEventBusConfig:
        """Parse event_bus_config from payload dict to typed model.

        Args:
            event_bus_data: Raw event_bus_config dict from payload.
            correlation_id: Correlation ID for tracing.

        Returns:
            Parsed ModelNodeEventBusConfig instance.

        Raises:
            ProtocolConfigurationError: If any topic entry has an invalid topic,
                invalid format, or invalid message_category.
        """
        subscribe_topics: list[ModelEventBusTopicEntry] = []
        publish_topics: list[ModelEventBusTopicEntry] = []

        raw_subscribe = event_bus_data.get("subscribe_topics")
        if isinstance(raw_subscribe, list):
            for idx, entry in enumerate(raw_subscribe):
                if isinstance(entry, dict):
                    stripped_topic, message_category = self._validate_topic_entry(
                        entry=entry,
                        location=f"subscribe_topics[{idx}]",
                        correlation_id=correlation_id,
                    )
                    subscribe_topics.append(
                        ModelEventBusTopicEntry(
                            topic=stripped_topic,
                            event_type=entry.get("event_type")
                            if isinstance(entry.get("event_type"), str)
                            else None,
                            message_category=message_category,
                            description=entry.get("description")
                            if isinstance(entry.get("description"), str)
                            else None,
                        )
                    )

        raw_publish = event_bus_data.get("publish_topics")
        if isinstance(raw_publish, list):
            for idx, entry in enumerate(raw_publish):
                if isinstance(entry, dict):
                    stripped_topic, message_category = self._validate_topic_entry(
                        entry=entry,
                        location=f"publish_topics[{idx}]",
                        correlation_id=correlation_id,
                    )
                    publish_topics.append(
                        ModelEventBusTopicEntry(
                            topic=stripped_topic,
                            event_type=entry.get("event_type")
                            if isinstance(entry.get("event_type"), str)
                            else None,
                            message_category=message_category,
                            description=entry.get("description")
                            if isinstance(entry.get("description"), str)
                            else None,
                        )
                    )

        return ModelNodeEventBusConfig(
            subscribe_topics=subscribe_topics,
            publish_topics=publish_topics,
        )

    async def _register_service(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Register service with Consul agent.

        Args:
            payload: dict containing:
                - name: Service name (required)
                - service_id: Optional unique service ID (defaults to name)
                - address: Optional service address
                - port: Optional service port
                - tags: Optional list of tags
                - check: Optional health check configuration dict
                - node_id: Optional node ID for event bus registration
                - event_bus_config: Optional event bus configuration dict containing:
                    - subscribe_topics: List of topic entries to subscribe to
                    - publish_topics: List of topic entries to publish to
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the registration result with correlation tracking

        Event Bus Integration:
            When event_bus_config is provided along with node_id, this method will:
            1. Store the event bus configuration in Consul KV at onex/nodes/{node_id}/event_bus/
            2. Update the topic -> node_id reverse index at onex/topics/{topic}/subscribers
        """
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.register",
                target_name="consul_handler",
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'name' in payload: "
                f"expected non-empty string, got {type(name).__name__}",
                context=ctx,
                parameter="name",
                value=str(name) if name is not None else "None",
            )

        service_id = payload.get("service_id")
        service_id_str: str | None = service_id if isinstance(service_id, str) else None

        address = payload.get("address")
        address_str: str | None = address if isinstance(address, str) else None

        port = payload.get("port")
        port_int: int | None = port if isinstance(port, int) else None

        tags = payload.get("tags")
        tags_list: list[str] | None = None
        if isinstance(tags, list):
            tags_list = [str(t) for t in tags]

        check = payload.get("check")
        check_dict: dict[str, object] | None = (
            check if isinstance(check, dict) else None
        )

        if self._client is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.register",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise InfraConsulError(
                "Consul client not initialized",
                context=context,
                service_name=name,
            )

        def register_func() -> bool:
            if self._client is None:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.register",
                    target_name="consul_handler",
                    correlation_id=correlation_id,
                )
                raise InfraConsulError(
                    "Consul client not initialized",
                    context=ctx,
                    service_name=name,
                )
            self._client.agent.service.register(
                name=name,
                service_id=service_id_str,
                address=address_str,
                port=port_int,
                tags=tags_list,
                check=check_dict,
            )
            return True

        await self._execute_with_retry(
            "consul.register",
            register_func,
            correlation_id,
        )

        # Handle event bus configuration if provided
        event_bus_data = payload.get("event_bus_config")
        node_id = payload.get("node_id")

        # Fail fast: if event_bus_config is present, node_id is REQUIRED
        if isinstance(event_bus_data, dict):
            if not isinstance(node_id, str) or not node_id.strip():
                ctx = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.register",
                )
                raise ProtocolConfigurationError(
                    "event_bus_config requires a valid 'node_id': "
                    f"expected non-empty string, got {type(node_id).__name__}",
                    context=ctx,
                    parameter="node_id",
                    value=str(node_id) if node_id is not None else "None",
                )

            logger.info(
                "Processing event_bus_config for node %s",
                node_id,
                extra={"correlation_id": str(correlation_id), "node_id": node_id},
            )

            # Parse the event bus config
            event_bus = self._parse_event_bus_config(event_bus_data, correlation_id)

            # Update topic index FIRST (uses old topics from previous registration)
            # This computes delta and updates reverse index
            await self._update_topic_index(node_id, event_bus, correlation_id)

            # Store the new event bus config AFTER index update
            # Order matters: _update_topic_index reads old topics before we overwrite
            await self._store_node_event_bus(node_id, event_bus, correlation_id)

            logger.info(
                "Completed event_bus registration for node %s",
                node_id,
                extra={"correlation_id": str(correlation_id), "node_id": node_id},
            )

        typed_payload = ModelConsulRegisterPayload(
            registered=True,
            name=name,
            consul_service_id=service_id_str or name,
        )
        return self._build_response(typed_payload, correlation_id, input_envelope_id)

    async def _deregister_service(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Deregister service from Consul agent.

        Args:
            payload: dict containing:
                - service_id: Service ID to deregister (required)
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the deregistration result with correlation tracking
        """
        service_id = payload.get("service_id")
        if not isinstance(service_id, str) or not service_id.strip():
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.deregister",
                target_name="consul_handler",
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'service_id' in payload: "
                f"expected non-empty string, got {type(service_id).__name__}",
                context=ctx,
                parameter="service_id",
                value=str(service_id) if service_id is not None else "None",
            )

        if self._client is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.deregister",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise InfraConsulError(
                "Consul client not initialized",
                context=context,
            )

        def deregister_func() -> bool:
            if self._client is None:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.deregister",
                    target_name="consul_handler",
                    correlation_id=correlation_id,
                )
                raise InfraConsulError(
                    "Consul client not initialized",
                    context=ctx,
                )
            self._client.agent.service.deregister(service_id)
            return True

        await self._execute_with_retry(
            "consul.deregister",
            deregister_func,
            correlation_id,
        )

        typed_payload = ModelConsulDeregisterPayload(
            deregistered=True,
            consul_service_id=service_id,
        )
        return self._build_response(typed_payload, correlation_id, input_envelope_id)


__all__: list[str] = ["MixinConsulService"]
