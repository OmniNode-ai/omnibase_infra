# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Key-Value operations mixin.

This mixin provides key-value store operations for HandlerConsul,
extracted to reduce class complexity and improve maintainability.

Operations:
    - consul.kv_get: Retrieve value(s) from KV store
    - consul.kv_put: Store value in KV store
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar, cast
from uuid import UUID

T = TypeVar("T")

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConsulError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers.models.consul import (
    ConsulPayload,
    ModelConsulKVGetFoundPayload,
    ModelConsulKVGetNotFoundPayload,
    ModelConsulKVGetRecursePayload,
    ModelConsulKVItem,
    ModelConsulKVPutPayload,
)
from omnibase_infra.handlers.models.model_consul_handler_response import (
    ModelConsulHandlerResponse,
)

if TYPE_CHECKING:
    import consul as consul_lib


class ProtocolConsulKVDependencies(Protocol):
    """Protocol defining required dependencies for KV operations.

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


class MixinConsulKV:
    """Mixin providing Consul Key-Value store operations.

    This mixin extracts KV operations from HandlerConsul to reduce
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

    async def _kv_get(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Get value from Consul KV store.

        Args:
            payload: dict containing:
                - key: KV key path (required)
                - recurse: Optional bool to get all keys under prefix (default: False)
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the KV data with correlation tracking
        """
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_get",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'key' in payload",
                context=ctx,
            )

        recurse = payload.get("recurse", False)
        recurse_bool = recurse is True or recurse == "true"

        if self._client is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_get",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise InfraConsulError(
                "Consul client not initialized",
                context=context,
                consul_key=key,
            )

        def get_func() -> tuple[
            int, list[dict[str, object]] | dict[str, object] | None
        ]:
            if self._client is None:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_get",
                    target_name="consul_handler",
                    correlation_id=correlation_id,
                )
                raise InfraConsulError(
                    "Consul client not initialized",
                    context=ctx,
                    consul_key=key,
                )
            index, data = self._client.kv.get(key, recurse=recurse_bool)
            return index, data

        # Type alias for KV get result
        KVGetResult = tuple[int, list[dict[str, object]] | dict[str, object] | None]
        result = await self._execute_with_retry(
            "consul.kv_get",
            get_func,
            correlation_id,
        )
        index, data = cast("KVGetResult", result)

        # Handle response - data can be None if key doesn't exist
        if data is None:
            typed_payload = ModelConsulKVGetNotFoundPayload(
                key=key,
                index=index,
            )
            return self._build_response(
                typed_payload, correlation_id, input_envelope_id
            )

        # Handle single key or recurse results
        if isinstance(data, list):
            # Recurse mode - multiple keys
            items: list[ModelConsulKVItem] = []
            for item in data:
                value = item.get("Value")
                decoded_value = (
                    value.decode("utf-8") if isinstance(value, bytes) else value
                )
                item_key = item.get("Key")
                item_flags = item.get("Flags")
                item_modify_index = item.get("ModifyIndex")
                items.append(
                    ModelConsulKVItem(
                        key=item_key if isinstance(item_key, str) else "",
                        value=decoded_value if isinstance(decoded_value, str) else None,
                        flags=item_flags if isinstance(item_flags, int) else None,
                        modify_index=item_modify_index
                        if isinstance(item_modify_index, int)
                        else None,
                    )
                )
            typed_payload_recurse = ModelConsulKVGetRecursePayload(
                found=len(items) > 0,
                items=items,
                count=len(items),
                index=index,
            )
            return self._build_response(
                typed_payload_recurse, correlation_id, input_envelope_id
            )
        else:
            # Single key mode
            value = data.get("Value")
            decoded_value = value.decode("utf-8") if isinstance(value, bytes) else value
            data_key = data.get("Key")
            data_flags = data.get("Flags")
            data_modify_index = data.get("ModifyIndex")
            typed_payload_found = ModelConsulKVGetFoundPayload(
                key=data_key if isinstance(data_key, str) else key,
                value=decoded_value if isinstance(decoded_value, str) else None,
                flags=data_flags if isinstance(data_flags, int) else None,
                modify_index=data_modify_index
                if isinstance(data_modify_index, int)
                else None,
                index=index,
            )
            return self._build_response(
                typed_payload_found, correlation_id, input_envelope_id
            )

    async def _kv_put(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelConsulHandlerResponse]:
        """Put value to Consul KV store.

        Args:
            payload: dict containing:
                - key: KV key path (required)
                - value: Value to store (required, string)
                - flags: Optional integer flags
                - cas: Optional check-and-set index for optimistic locking
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the operation result with correlation tracking
        """
        key = payload.get("key")
        if not isinstance(key, str) or not key:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_put",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'key' in payload",
                context=ctx,
            )

        value = payload.get("value")
        if not isinstance(value, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_put",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'value' in payload - must be a string",
                context=ctx,
            )

        flags = payload.get("flags")
        flags_int: int | None = flags if isinstance(flags, int) else None

        cas = payload.get("cas")
        cas_int: int | None = cas if isinstance(cas, int) else None

        if self._client is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.CONSUL,
                operation="consul.kv_put",
                target_name="consul_handler",
                correlation_id=correlation_id,
            )
            raise InfraConsulError(
                "Consul client not initialized",
                context=context,
                consul_key=key,
            )

        def put_func() -> bool:
            if self._client is None:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_put",
                    target_name="consul_handler",
                    correlation_id=correlation_id,
                )
                raise InfraConsulError(
                    "Consul client not initialized",
                    context=ctx,
                    consul_key=key,
                )
            result: bool = self._client.kv.put(key, value, flags=flags_int, cas=cas_int)
            return result

        result = await self._execute_with_retry(
            "consul.kv_put",
            put_func,
            correlation_id,
        )
        success = cast("bool", result)

        typed_payload = ModelConsulKVPutPayload(
            success=success,
            key=key,
        )
        return self._build_response(typed_payload, correlation_id, input_envelope_id)


__all__: list[str] = ["MixinConsulKV"]
