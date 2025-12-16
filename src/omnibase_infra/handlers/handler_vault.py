# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HashiCorp Vault Handler - MVP implementation using hvac async client.

Supports secret management operations with configurable retry logic and
automatic token renewal management.

Security Features:
    - SecretStr protection for tokens (prevents accidental logging)
    - Sanitized error messages (never expose secrets in logs)
    - SSL verification enabled by default
    - Token auto-renewal management

All secret operations MUST use proper authentication and authorization.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import UUID, uuid4

import hvac
from omnibase_core.enums.enum_handler_type import EnumHandlerType
from pydantic import SecretStr

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    RuntimeHostError,
    SecretResolutionError,
)
from omnibase_infra.handlers.model_vault_handler_config import ModelVaultHandlerConfig

logger = logging.getLogger(__name__)

_DEFAULT_MOUNT_POINT: str = "secret"
_SUPPORTED_OPERATIONS: frozenset[str] = frozenset({
    "vault.read_secret",
    "vault.write_secret",
    "vault.delete_secret",
    "vault.list_secrets",
    "vault.renew_token",
    "vault.health_check",
})


class VaultHandler:
    """HashiCorp Vault handler using hvac client (MVP: KV v2 secrets engine).

    Security Policy - Token Handling:
        The Vault token contains sensitive credentials and is treated as a secret
        throughout this handler. The following security measures are enforced:

        1. Token is stored as SecretStr in config (never logged or exposed)
        2. All error messages use generic descriptions without exposing token
        3. Health check responses exclude token information
        4. Token renewal is automatic when TTL falls below threshold
        5. The describe() method returns capabilities without credentials

        See CLAUDE.md "Error Sanitization Guidelines" for the full security policy
        on what information is safe vs unsafe to include in errors and logs.

    Token Renewal Management:
        - Tokens are automatically renewed when TTL < token_renewal_threshold_seconds
        - Token renewal is checked before each operation
        - Failed renewal raises InfraAuthenticationError
        - Token expiration tracking uses self._token_expires_at

    Retry Logic:
        - All operations use exponential backoff retry logic
        - Retry configuration from ModelVaultRetryConfig
        - Backoff calculation: initial_backoff * (exponential_base ** attempt)
        - Max backoff capped at max_backoff_seconds
    """

    def __init__(self) -> None:
        """Initialize VaultHandler in uninitialized state."""
        self._client: hvac.Client | None = None
        self._config: ModelVaultHandlerConfig | None = None
        self._initialized: bool = False
        self._token_expires_at: float = 0.0

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return EnumHandlerType.VAULT."""
        return EnumHandlerType.VAULT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize Vault client with configuration.

        Args:
            config: Configuration dict containing:
                - url: Vault server URL (required)
                - token: Vault authentication token (required)
                - namespace: Optional Vault namespace for Enterprise
                - timeout_seconds: Optional timeout (default 30.0)
                - verify_ssl: Optional SSL verification (default True)
                - token_renewal_threshold_seconds: Optional renewal threshold (default 300.0)
                - retry: Optional retry configuration dict

        Raises:
            RuntimeHostError: If URL or token is missing, or client initialization fails.
            InfraAuthenticationError: If token authentication fails.
            InfraConnectionError: If connection to Vault server fails.

        Security:
            Token must be provided via environment variable, not hardcoded in config.
            Use SecretStr for token to prevent accidental logging.
        """
        init_correlation_id = uuid4()

        # Parse configuration using Pydantic model
        try:
            # Handle SecretStr token conversion
            token_raw = config.get("token")
            if isinstance(token_raw, str):
                config = dict(config)  # Make mutable copy
                config["token"] = SecretStr(token_raw)

            # Type ignore for dict unpacking - Pydantic handles validation
            self._config = ModelVaultHandlerConfig(**config)  # type: ignore[arg-type]
        except Exception as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
            )
            raise RuntimeHostError(
                f"Invalid Vault configuration: {type(e).__name__}",
                context=ctx,
            ) from e

        # Validate required fields
        if not self._config.url:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
            )
            raise RuntimeHostError(
                "Missing 'url' in config - Vault server URL required",
                context=ctx,
            )

        if self._config.token is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
            )
            raise RuntimeHostError(
                "Missing 'token' in config - Vault authentication token required",
                context=ctx,
            )

        try:
            # Initialize hvac client (synchronous)
            self._client = hvac.Client(
                url=self._config.url,
                token=self._config.token.get_secret_value(),
                namespace=self._config.namespace,
                verify=self._config.verify_ssl,
                timeout=self._config.timeout_seconds,
            )

            # Verify authentication
            if not self._client.is_authenticated():
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.VAULT,
                    operation="initialize",
                    target_name="vault_handler",
                    correlation_id=init_correlation_id,
                )
                raise InfraAuthenticationError(
                    "Vault authentication failed - check token validity",
                    context=ctx,
                )

            # Initialize token expiration tracking
            self._token_expires_at = time.time() + 3600.0  # Default 1 hour TTL

            self._initialized = True
            logger.info(
                "VaultHandler initialized",
                extra={
                    "url": self._config.url,
                    "namespace": self._config.namespace,
                    "timeout_seconds": self._config.timeout_seconds,
                    "verify_ssl": self._config.verify_ssl,
                },
            )

        except InfraAuthenticationError:
            # Re-raise our own authentication errors without wrapping
            raise
        except hvac.exceptions.InvalidRequest as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
            )
            raise InfraAuthenticationError(
                "Vault authentication failed - invalid token or permissions",
                context=ctx,
            ) from e
        except hvac.exceptions.VaultError as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
            )
            raise InfraConnectionError(
                f"Failed to connect to Vault: {type(e).__name__}",
                context=ctx,
            ) from e
        except Exception as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="initialize",
                target_name="vault_handler",
                correlation_id=init_correlation_id,
            )
            raise RuntimeHostError(
                f"Failed to initialize Vault client: {type(e).__name__}",
                context=ctx,
            ) from e

    async def shutdown(self) -> None:
        """Close Vault client and release resources."""
        if self._client is not None:
            # hvac.Client doesn't have async close, just clear reference
            self._client = None
        self._initialized = False
        self._config = None
        logger.info("VaultHandler shutdown complete")

    async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
        """Execute Vault operation from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: Vault operation (vault.read_secret, vault.write_secret, etc.)
                - payload: dict with operation-specific parameters
                - correlation_id: Optional correlation ID for tracing

        Returns:
            Response envelope with status, payload, and correlation_id

        Raises:
            RuntimeHostError: If handler not initialized or invalid input.
            InfraConnectionError: If Vault connection fails.
            InfraAuthenticationError: If authentication fails.
            SecretResolutionError: If secret resolution fails.
        """
        correlation_id = self._extract_correlation_id(envelope)

        if not self._initialized or self._client is None or self._config is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="execute",
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "VaultHandler not initialized. Call initialize() first.",
                context=ctx,
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="execute",
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'operation' in envelope",
                context=ctx,
            )

        if operation not in _SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation=operation,
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Operation '{operation}' not supported in MVP. "
                f"Available: {', '.join(sorted(_SUPPORTED_OPERATIONS))}",
                context=ctx,
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation=operation,
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        # Check token renewal before operation
        await self._check_token_renewal(correlation_id)

        # Route to appropriate handler
        if operation == "vault.read_secret":
            return await self._read_secret(payload, correlation_id)
        elif operation == "vault.write_secret":
            return await self._write_secret(payload, correlation_id)
        elif operation == "vault.delete_secret":
            return await self._delete_secret(payload, correlation_id)
        elif operation == "vault.list_secrets":
            return await self._list_secrets(payload, correlation_id)
        elif operation == "vault.renew_token":
            return await self._renew_token_operation(correlation_id)
        else:  # vault.health_check
            return await self._health_check_operation(correlation_id)

    def _extract_correlation_id(self, envelope: dict[str, object]) -> UUID:
        """Extract or generate correlation ID from envelope."""
        raw = envelope.get("correlation_id")
        if isinstance(raw, UUID):
            return raw
        if isinstance(raw, str):
            try:
                return UUID(raw)
            except ValueError:
                pass
        return uuid4()

    async def _check_token_renewal(self, correlation_id: UUID) -> None:
        """Check if token needs renewal and renew if necessary.

        Args:
            correlation_id: Correlation ID for tracing

        Raises:
            InfraAuthenticationError: If token renewal fails
        """
        if self._config is None or self._client is None:
            return

        current_time = time.time()
        time_until_expiry = self._token_expires_at - current_time

        if time_until_expiry < self._config.token_renewal_threshold_seconds:
            logger.info(
                "Token approaching expiration, renewing",
                extra={
                    "time_until_expiry_seconds": time_until_expiry,
                    "threshold_seconds": self._config.token_renewal_threshold_seconds,
                    "correlation_id": str(correlation_id),
                },
            )
            await self.renew_token()

    async def _execute_with_retry(
        self,
        operation: str,
        func: Any,
        correlation_id: UUID,
    ) -> Any:
        """Execute operation with exponential backoff retry logic.

        Args:
            operation: Operation name for logging
            func: Callable to execute (synchronous hvac method)
            correlation_id: Correlation ID for tracing

        Returns:
            Result from func()

        Raises:
            InfraTimeoutError: If all retries exhausted or operation times out
            InfraConnectionError: If connection fails
            InfraAuthenticationError: If authentication fails
        """
        if self._config is None:
            raise RuntimeError("Config not initialized")

        retry_config = self._config.retry
        last_exception: Exception | None = None

        for attempt in range(retry_config.max_attempts):
            try:
                # hvac is synchronous, wrap in thread executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func),
                    timeout=self._config.timeout_seconds,
                )
                return result

            except TimeoutError as e:
                last_exception = e
                if attempt == retry_config.max_attempts - 1:
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.VAULT,
                        operation=operation,
                        target_name="vault_handler",
                        correlation_id=correlation_id,
                    )
                    raise InfraTimeoutError(
                        f"Vault operation timed out after {self._config.timeout_seconds}s",
                        context=ctx,
                        timeout_seconds=self._config.timeout_seconds,
                    ) from e

            except hvac.exceptions.Forbidden as e:
                # Don't retry authentication failures
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.VAULT,
                    operation=operation,
                    target_name="vault_handler",
                    correlation_id=correlation_id,
                )
                raise InfraAuthenticationError(
                    "Vault operation forbidden - check token permissions",
                    context=ctx,
                ) from e

            except hvac.exceptions.InvalidPath as e:
                # Don't retry invalid path errors
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.VAULT,
                    operation=operation,
                    target_name="vault_handler",
                    correlation_id=correlation_id,
                )
                raise SecretResolutionError(
                    "Secret path not found or invalid",
                    context=ctx,
                ) from e

            except hvac.exceptions.VaultDown as e:
                last_exception = e
                if attempt == retry_config.max_attempts - 1:
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.VAULT,
                        operation=operation,
                        target_name="vault_handler",
                        correlation_id=correlation_id,
                    )
                    raise InfraUnavailableError(
                        "Vault server is unavailable",
                        context=ctx,
                    ) from e

            except Exception as e:
                last_exception = e
                if attempt == retry_config.max_attempts - 1:
                    ctx = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.VAULT,
                        operation=operation,
                        target_name="vault_handler",
                        correlation_id=correlation_id,
                    )
                    raise InfraConnectionError(
                        f"Vault operation failed: {type(e).__name__}",
                        context=ctx,
                    ) from e

            # Calculate exponential backoff
            backoff = min(
                retry_config.initial_backoff_seconds * (retry_config.exponential_base ** attempt),
                retry_config.max_backoff_seconds,
            )

            logger.debug(
                "Retrying Vault operation",
                extra={
                    "operation": operation,
                    "attempt": attempt + 1,
                    "max_attempts": retry_config.max_attempts,
                    "backoff_seconds": backoff,
                    "correlation_id": str(correlation_id),
                },
            )

            await asyncio.sleep(backoff)

        # Should never reach here, but satisfy type checker
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Retry loop completed without result")

    async def _read_secret(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Read secret from Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")

        Returns:
            Response envelope with secret data
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.read_secret",
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        mount_point = payload.get("mount_point", _DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = _DEFAULT_MOUNT_POINT

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def read_func() -> dict[str, Any]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, Any] = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=mount_point,
            )
            return result

        result = await self._execute_with_retry(
            "vault.read_secret",
            read_func,
            correlation_id,
        )

        return {
            "status": "success",
            "payload": {
                "data": result.get("data", {}).get("data", {}),
                "metadata": result.get("data", {}).get("metadata", {}),
            },
            "correlation_id": correlation_id,
        }

    async def _write_secret(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Write secret to Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - data: Secret data dict (required)
                - mount_point: KV mount point (default: "secret")

        Returns:
            Response envelope with write confirmation
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.write_secret",
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        data = payload.get("data")
        if not isinstance(data, dict):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.write_secret",
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'data' in payload - must be a dict",
                context=ctx,
            )

        mount_point = payload.get("mount_point", _DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = _DEFAULT_MOUNT_POINT

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def write_func() -> dict[str, Any]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, Any] = self._client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=data,
                mount_point=mount_point,
            )
            return result

        result = await self._execute_with_retry(
            "vault.write_secret",
            write_func,
            correlation_id,
        )

        return {
            "status": "success",
            "payload": {
                "version": result.get("data", {}).get("version"),
                "created_time": result.get("data", {}).get("created_time"),
            },
            "correlation_id": correlation_id,
        }

    async def _delete_secret(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Delete secret from Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")

        Returns:
            Response envelope with deletion confirmation
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.delete_secret",
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        mount_point = payload.get("mount_point", _DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = _DEFAULT_MOUNT_POINT

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def delete_func() -> None:
            if self._client is None:
                raise RuntimeError("Client not initialized")
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

        return {
            "status": "success",
            "payload": {"deleted": True},
            "correlation_id": correlation_id,
        }

    async def _list_secrets(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """List secrets at path in Vault KV v2 secrets engine.

        Args:
            payload: dict containing:
                - path: Secret path (required)
                - mount_point: KV mount point (default: "secret")

        Returns:
            Response envelope with list of secret keys
        """
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.list_secrets",
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'path' in payload",
                context=ctx,
            )

        mount_point = payload.get("mount_point", _DEFAULT_MOUNT_POINT)
        if not isinstance(mount_point, str):
            mount_point = _DEFAULT_MOUNT_POINT

        if self._client is None:
            raise RuntimeError("Client not initialized")

        def list_func() -> dict[str, Any]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, Any] = self._client.secrets.kv.v2.list_secrets(
                path=path,
                mount_point=mount_point,
            )
            return result

        result = await self._execute_with_retry(
            "vault.list_secrets",
            list_func,
            correlation_id,
        )

        keys = result.get("data", {}).get("keys", [])

        return {
            "status": "success",
            "payload": {"keys": keys},
            "correlation_id": correlation_id,
        }

    async def renew_token(self) -> dict[str, Any]:
        """Renew Vault authentication token.

        Returns:
            Token renewal information including new TTL

        Raises:
            InfraAuthenticationError: If token renewal fails
        """
        correlation_id = uuid4()

        if self._client is None or self._config is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.renew_token",
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "VaultHandler not initialized",
                context=ctx,
            )

        def renew_func() -> dict[str, Any]:
            if self._client is None:
                raise RuntimeError("Client not initialized")
            result: dict[str, Any] = self._client.auth.token.renew_self()
            return result

        try:
            result = await self._execute_with_retry(
                "vault.renew_token",
                renew_func,
                correlation_id,
            )

            # Update token expiration tracking
            auth_data = result.get("auth", {})
            lease_duration = auth_data.get("lease_duration", 3600)
            self._token_expires_at = time.time() + lease_duration

            logger.info(
                "Token renewed successfully",
                extra={
                    "new_ttl_seconds": lease_duration,
                    "correlation_id": str(correlation_id),
                },
            )

            # Explicit type annotation for mypy
            return_value: dict[str, Any] = result
            return return_value

        except Exception as e:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.renew_token",
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise InfraAuthenticationError(
                "Failed to renew Vault token",
                context=ctx,
            ) from e

    async def _renew_token_operation(
        self,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Execute token renewal operation from envelope.

        Args:
            correlation_id: Correlation ID for tracing

        Returns:
            Response envelope with renewal information
        """
        result = await self.renew_token()

        auth_data = result.get("auth", {})
        return {
            "status": "success",
            "payload": {
                "renewable": auth_data.get("renewable", False),
                "lease_duration": auth_data.get("lease_duration", 0),
            },
            "correlation_id": correlation_id,
        }

    async def health_check(self) -> dict[str, object]:
        """Return handler health status.

        Returns:
            Health status dict with handler state information
        """
        healthy = False
        if self._initialized and self._client is not None:
            try:
                # Synchronous hvac health check
                loop = asyncio.get_event_loop()
                health_result = await loop.run_in_executor(
                    None,
                    self._client.sys.read_health_status,
                )
                healthy = health_result.get("initialized", False)
            except Exception as e:
                logger.warning(
                    "Health check failed",
                    extra={
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                )
                healthy = False

        return {
            "healthy": healthy,
            "initialized": self._initialized,
            "handler_type": self.handler_type.value,
            "timeout_seconds": self._config.timeout_seconds if self._config else 30.0,
        }

    async def _health_check_operation(
        self,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Execute health check operation from envelope.

        Args:
            correlation_id: Correlation ID for tracing

        Returns:
            Response envelope with health check information
        """
        health_status = await self.health_check()

        return {
            "status": "success",
            "payload": health_status,
            "correlation_id": correlation_id,
        }

    def describe(self) -> dict[str, object]:
        """Return handler metadata and capabilities.

        Returns:
            Handler description with supported operations and configuration
        """
        return {
            "handler_type": self.handler_type.value,
            "supported_operations": sorted(_SUPPORTED_OPERATIONS),
            "timeout_seconds": self._config.timeout_seconds if self._config else 30.0,
            "initialized": self._initialized,
            "version": "0.1.0-mvp",
        }


__all__: list[str] = ["VaultHandler"]
