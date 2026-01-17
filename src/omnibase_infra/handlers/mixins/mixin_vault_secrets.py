# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault secrets mixin for HandlerVault.

Provides CRUD operations for secrets in Vault KV v2 secrets engine.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar
from uuid import UUID

import hvac

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraVaultError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers.models.vault import ModelVaultHandlerConfig

T = TypeVar("T")

# Handler ID for ModelHandlerOutput
HANDLER_ID_VAULT: str = "vault-handler"

DEFAULT_MOUNT_POINT: str = "secret"


class MixinVaultSecrets:
    """Mixin providing secret operations for HandlerVault.

    Contains methods for:
    - Reading secrets from KV v2
    - Writing secrets to KV v2
    - Deleting secrets from KV v2
    - Listing secrets at a path
    """

    # Instance attributes (declared for type checking)
    _client: hvac.Client | None
    _config: ModelVaultHandlerConfig | None

    # Methods from other mixins that will be available at runtime
    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        correlation_id: UUID,
    ) -> T:
        """Execute with retry - provided by MixinVaultRetry."""
        raise NotImplementedError("Must be provided by implementing class")

    async def _read_secret(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Read secret from Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput with secret data
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.read_secret",
                target_name="vault_handler",
                namespace=self._config.namespace if self._config else None,
            )
            # Payload validation error (bad request) - not an infrastructure failure
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        mount_point = payload.get("mount_point", DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = DEFAULT_MOUNT_POINT

        if self._client is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.read_secret",
                target_name="vault_handler",
                namespace=self._config.namespace if self._config else None,
            )
            raise InfraVaultError(
                "Vault client not initialized for operation 'vault.read_secret'",
                context=ctx,
            )

        # Capture namespace for use in closure
        namespace = self._config.namespace if self._config else None

        def read_func() -> dict[str, object]:
            if self._client is None:
                ctx = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.VAULT,
                    operation="vault.read_secret",
                    target_name="vault_handler",
                    namespace=namespace,
                )
                raise InfraVaultError(
                    "Vault client not initialized for operation 'vault.read_secret'",
                    context=ctx,
                )
            result: dict[str, object] = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=mount_point,
            )
            return result

        result = await self._execute_with_retry(
            "vault.read_secret",
            read_func,
            correlation_id,
        )

        # Extract nested data with type checking
        data_obj = result.get("data", {})
        data_dict = data_obj if isinstance(data_obj, dict) else {}
        secret_data = data_dict.get("data", {})
        metadata = data_dict.get("metadata", {})

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": {
                    "data": secret_data if isinstance(secret_data, dict) else {},
                    "metadata": metadata if isinstance(metadata, dict) else {},
                },
                "correlation_id": str(correlation_id),
            },
        )

    async def _write_secret(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Write secret to Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - data: Secret data dict (required)
                - mount_point: KV mount point (default: "secret")
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput with write confirmation
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.write_secret",
                target_name="vault_handler",
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        data = payload.get("data")
        if not isinstance(data, dict):
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.write_secret",
                target_name="vault_handler",
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'data' in payload - must be a dict",
                context=ctx,
            )

        mount_point = payload.get("mount_point", DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = DEFAULT_MOUNT_POINT

        if self._client is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.write_secret",
                target_name="vault_handler",
                namespace=self._config.namespace if self._config else None,
            )
            raise InfraVaultError(
                "Vault client not initialized for operation 'vault.write_secret'",
                context=ctx,
            )

        # Capture namespace for use in closure
        namespace = self._config.namespace if self._config else None

        def write_func() -> dict[str, object]:
            if self._client is None:
                ctx = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.VAULT,
                    operation="vault.write_secret",
                    target_name="vault_handler",
                    namespace=namespace,
                )
                raise InfraVaultError(
                    "Vault client not initialized for operation 'vault.write_secret'",
                    context=ctx,
                )
            result: dict[str, object] = (
                self._client.secrets.kv.v2.create_or_update_secret(
                    path=path,
                    secret=data,
                    mount_point=mount_point,
                )
            )
            return result

        result = await self._execute_with_retry(
            "vault.write_secret",
            write_func,
            correlation_id,
        )

        # Extract nested data with type checking
        data_obj = result.get("data", {})
        data_dict = data_obj if isinstance(data_obj, dict) else {}

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": {
                    "version": data_dict.get("version"),
                    "created_time": data_dict.get("created_time"),
                },
                "correlation_id": str(correlation_id),
            },
        )

    async def _delete_secret(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Delete secret from Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput with deletion confirmation
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.delete_secret",
                target_name="vault_handler",
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        mount_point = payload.get("mount_point", DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = DEFAULT_MOUNT_POINT

        if self._client is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.delete_secret",
                target_name="vault_handler",
                namespace=self._config.namespace if self._config else None,
            )
            raise InfraVaultError(
                "Vault client not initialized for operation 'vault.delete_secret'",
                context=ctx,
            )

        # Capture namespace for use in closure
        namespace = self._config.namespace if self._config else None

        def delete_func() -> None:
            if self._client is None:
                ctx = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.VAULT,
                    operation="vault.delete_secret",
                    target_name="vault_handler",
                    namespace=namespace,
                )
                raise InfraVaultError(
                    "Vault client not initialized for operation 'vault.delete_secret'",
                    context=ctx,
                )
            # Delete latest version
            self._client.secrets.kv.v2.delete_latest_version_of_secret(
                path=path,
                mount_point=mount_point,
            )

        await self._execute_with_retry(
            "vault.delete_secret",
            delete_func,
            correlation_id,
        )

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": {"deleted": True},
                "correlation_id": str(correlation_id),
            },
        )

    async def _list_secrets(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """List secrets at path in Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput with list of secret keys
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.list_secrets",
                target_name="vault_handler",
                namespace=self._config.namespace if self._config else None,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        mount_point = payload.get("mount_point", DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = DEFAULT_MOUNT_POINT

        if self._client is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.list_secrets",
                target_name="vault_handler",
                namespace=self._config.namespace if self._config else None,
            )
            raise InfraVaultError(
                "Vault client not initialized for operation 'vault.list_secrets'",
                context=ctx,
            )

        # Capture namespace for use in closure
        namespace = self._config.namespace if self._config else None

        def list_func() -> dict[str, object]:
            if self._client is None:
                ctx = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.VAULT,
                    operation="vault.list_secrets",
                    target_name="vault_handler",
                    namespace=namespace,
                )
                raise InfraVaultError(
                    "Vault client not initialized for operation 'vault.list_secrets'",
                    context=ctx,
                )
            result: dict[str, object] = self._client.secrets.kv.v2.list_secrets(
                path=path,
                mount_point=mount_point,
            )
            return result

        result = await self._execute_with_retry(
            "vault.list_secrets",
            list_func,
            correlation_id,
        )

        # Extract nested data with type checking
        data_obj = result.get("data", {})
        data_dict = data_obj if isinstance(data_obj, dict) else {}
        keys = data_dict.get("keys", [])

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": {"keys": keys if isinstance(keys, list) else []},
                "correlation_id": str(correlation_id),
            },
        )


__all__: list[str] = ["MixinVaultSecrets"]
