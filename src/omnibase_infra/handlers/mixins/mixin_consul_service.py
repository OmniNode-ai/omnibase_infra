# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Service registration operations mixin.

This mixin provides service registration and deregistration operations
for ConsulHandler, extracted to reduce class complexity.

Operations:
    - consul.register: Register service with Consul agent
    - consul.deregister: Deregister service from Consul agent
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar
from uuid import UUID

T = TypeVar("T")

from omnibase_core.models.dispatch import ModelHandlerOutput

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
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

if TYPE_CHECKING:
    import consul as consul_lib
    from omnibase_core.types import JsonType


class ProtocolConsulServiceDependencies(Protocol):
    """Protocol defining required dependencies for service operations.

    ConsulHandler must provide these attributes/methods for the mixin to work.
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


class MixinConsulService:
    """Mixin providing Consul service registration operations.

    This mixin extracts service operations from ConsulHandler to reduce
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
        raise NotImplementedError("Must be provided by implementing class")

    def _build_response(
        self,
        typed_payload: ConsulPayload,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Build standardized response - provided by host class."""

    async def _register_service(
        self,
        payload: dict[str, JsonType],
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
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the registration result with correlation tracking
        """
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.register",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'name' in payload",
                context=ctx,
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
        check_dict: dict[str, JsonType] | None = (
            check if isinstance(check, dict) else None
        )

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def register_func() -> bool:
            if self._client is None:
                raise RuntimeError("Client not initialized")
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

        typed_payload = ModelConsulRegisterPayload(
            registered=True,
            name=name,
            consul_service_id=service_id_str or name,
        )
        return self._build_response(typed_payload, correlation_id, input_envelope_id)

    async def _deregister_service(
        self,
        payload: dict[str, JsonType],
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
        if not isinstance(service_id, str) or not service_id:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.deregister",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'service_id' in payload",
                context=ctx,
            )

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def deregister_func() -> bool:
            if self._client is None:
                raise RuntimeError("Client not initialized")
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
