# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault token mixin for HandlerVault.

Provides token management operations including renewal, TTL tracking,
and automatic token refresh before expiration.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar
from uuid import UUID, uuid4

import hvac

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraVaultError,
    ModelInfraErrorContext,
)
from omnibase_infra.handlers.models.vault import ModelVaultHandlerConfig

T = TypeVar("T")

# Handler ID for ModelHandlerOutput
HANDLER_ID_VAULT: str = "vault-handler"

logger = logging.getLogger(__name__)


class MixinVaultToken:
    """Mixin providing token management for HandlerVault.

    Contains methods for:
    - Token renewal checking and auto-refresh
    - Token TTL extraction from Vault responses
    - Token renewal operation execution
    """

    # Instance attributes (declared for type checking)
    _client: hvac.Client | None
    _config: ModelVaultHandlerConfig | None
    _token_expires_at: float
    _executor: ThreadPoolExecutor | None

    # Methods from other mixins that will be available at runtime
    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        correlation_id: UUID,
    ) -> T:
        """Execute with retry - provided by MixinVaultRetry."""
        raise NotImplementedError("Must be provided by implementing class")

    async def _check_token_renewal(self, correlation_id: UUID) -> None:
        """Check if token needs renewal and renew if necessary.

        Args:
            correlation_id: Correlation ID for tracing

        Raises:
            InfraAuthenticationError: If token renewal fails
        """
        if self._config is None or self._client is None:
            logger.debug(
                "Token renewal check skipped - handler not initialized",
                extra={
                    "config_initialized": self._config is not None,
                    "client_initialized": self._client is not None,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        current_time = time.time()
        time_until_expiry = self._token_expires_at - current_time
        threshold = self._config.token_renewal_threshold_seconds
        needs_renewal = time_until_expiry < threshold

        # Log edge case when expiry time exactly equals threshold
        # This helps troubleshoot boundary condition behavior
        is_edge_case = abs(time_until_expiry - threshold) < 0.001  # Within 1ms

        logger.debug(
            "Token renewal check",
            extra={
                "current_time": current_time,
                "token_expires_at": self._token_expires_at,
                "time_until_expiry_seconds": time_until_expiry,
                "threshold_seconds": threshold,
                "needs_renewal": needs_renewal,
                "is_threshold_edge_case": is_edge_case,
                "correlation_id": str(correlation_id),
            },
        )

        if is_edge_case:
            logger.debug(
                "Token renewal edge case detected - expiry equals threshold",
                extra={
                    "time_until_expiry_seconds": time_until_expiry,
                    "threshold_seconds": threshold,
                    "difference_ms": abs(time_until_expiry - threshold) * 1000,
                    "will_renew": needs_renewal,
                    "correlation_id": str(correlation_id),
                },
            )

        if needs_renewal:
            logger.info(
                "Token approaching expiration, renewing",
                extra={
                    "time_until_expiry_seconds": time_until_expiry,
                    "threshold_seconds": threshold,
                    "correlation_id": str(correlation_id),
                },
            )
            await self.renew_token(correlation_id=correlation_id)
            logger.debug(
                "Token renewal completed successfully",
                extra={
                    "new_expires_at": self._token_expires_at,
                    "new_time_until_expiry_seconds": self._token_expires_at
                    - time.time(),
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            logger.debug(
                "Token renewal skipped - token still valid",
                extra={
                    "time_until_expiry_seconds": time_until_expiry,
                    "threshold_seconds": threshold,
                    "margin_seconds": time_until_expiry - threshold,
                    "correlation_id": str(correlation_id),
                },
            )

    def _validate_renewal_preconditions(self, correlation_id: UUID) -> None:
        """Validate that handler is initialized before renewal.

        Args:
            correlation_id: Correlation ID for tracing

        Raises:
            InfraVaultError: If handler not initialized
        """
        if self._client is None or self._config is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="vault.renew_token",
                target_name="vault_handler",
                correlation_id=correlation_id,
                namespace=self._config.namespace if self._config else None,
            )
            raise InfraVaultError(
                "HandlerVault not initialized",
                context=ctx,
            )

    def _extract_ttl_from_renewal_response(
        self,
        result: dict[str, object],
        default_ttl: int,
        correlation_id: UUID,
    ) -> int:
        """Extract TTL from token renewal response.

        Args:
            result: Renewal response from Vault
            default_ttl: Default TTL to use if extraction fails
            correlation_id: Correlation ID for tracing

        Returns:
            Token TTL in seconds
        """
        auth_data = result.get("auth", {})

        if isinstance(auth_data, dict):
            lease_duration = auth_data.get("lease_duration")
            if isinstance(lease_duration, int) and lease_duration > 0:
                return lease_duration

        # Fallback to config or safe default
        logger.warning(
            "Token TTL not in renewal response, using fallback",
            extra={
                "ttl": default_ttl,
                "correlation_id": str(correlation_id),
            },
        )
        return default_ttl

    async def _refresh_ttl_from_vault_lookup(
        self,
        current_ttl: int,
        correlation_id: UUID,
    ) -> int:
        """Refresh TTL by querying actual value from Vault.

        Args:
            current_ttl: Current TTL value (used as fallback)
            correlation_id: Correlation ID for tracing

        Returns:
            Refreshed TTL in seconds
        """
        if self._client is None:
            return current_ttl

        try:
            loop = asyncio.get_running_loop()
            token_info = await loop.run_in_executor(
                self._executor,
                self._client.auth.token.lookup_self,
            )
            token_data = token_info.get("data", {})
            if isinstance(token_data, dict):
                ttl_seconds = token_data.get("ttl")
                if isinstance(ttl_seconds, int) and ttl_seconds > 0:
                    logger.info(
                        "Token TTL refreshed from Vault lookup",
                        extra={
                            "ttl_seconds": ttl_seconds,
                            "correlation_id": str(correlation_id),
                        },
                    )
                    return ttl_seconds
        except Exception as e:
            logger.debug(
                "Token lookup after renewal failed, using lease_duration",
                extra={
                    "error_type": type(e).__name__,
                    "correlation_id": str(correlation_id),
                },
            )

        return current_ttl

    async def renew_token(
        self, correlation_id: UUID | None = None
    ) -> dict[str, object]:
        """Renew Vault authentication token.

        Token TTL Extraction Logic:
            1. Extract 'auth.lease_duration' from Vault renewal response
            2. If lease_duration is invalid or missing, use default_token_ttl
            3. Update _token_expires_at = current_time + extracted_ttl
            4. Log warning when falling back to default TTL

        Args:
            correlation_id: Optional correlation ID for tracing. When called via
                envelope dispatch, this preserves the request's correlation_id.
                When called directly (e.g., by monitoring), a new ID is generated.

        Returns:
            Token renewal information including new TTL

        Raises:
            InfraAuthenticationError: If token authentication/authorization fails
            InfraConnectionError: If connection to Vault fails
            InfraTimeoutError: If renewal operation times out
            InfraUnavailableError: If circuit breaker is open
        """
        if correlation_id is None:
            correlation_id = uuid4()

        self._validate_renewal_preconditions(correlation_id)

        # At this point, _client and _config are guaranteed non-None
        assert self._client is not None
        assert self._config is not None

        # Capture namespace for use in closure
        namespace = self._config.namespace

        def renew_func() -> dict[str, object]:
            if self._client is None:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.VAULT,
                    operation="vault.renew_token",
                    target_name="vault_handler",
                    correlation_id=correlation_id,
                    namespace=namespace,
                )
                raise InfraVaultError("Vault client not initialized", context=ctx)
            result: dict[str, object] = self._client.auth.token.renew_self()
            return result

        # _execute_with_retry already raises properly typed errors:
        # - InfraAuthenticationError for auth failures (Forbidden, InvalidRequest)
        # - InfraConnectionError for connection failures
        # - InfraTimeoutError for timeout errors
        # - InfraUnavailableError for circuit breaker open
        # Let these errors propagate naturally without masking them
        result = await self._execute_with_retry(
            "vault.renew_token",
            renew_func,
            correlation_id,
        )

        # Extract TTL from renewal response
        token_ttl = self._extract_ttl_from_renewal_response(
            result, self._config.default_token_ttl, correlation_id
        )

        # Refresh TTL from Vault lookup (may override renewal response)
        token_ttl = await self._refresh_ttl_from_vault_lookup(token_ttl, correlation_id)

        # Update token expiration tracking
        self._token_expires_at = time.time() + token_ttl

        logger.info(
            "Token renewed successfully",
            extra={
                "new_ttl_seconds": token_ttl,
                "correlation_id": str(correlation_id),
            },
        )

        return result

    async def _renew_token_operation(
        self,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute token renewal operation from envelope.

        Args:
            correlation_id: Correlation ID for tracing
            input_envelope_id: Input envelope ID for causality tracking

        Returns:
            ModelHandlerOutput with renewal information
        """
        result = await self.renew_token(correlation_id=correlation_id)

        # Extract nested auth data with type checking
        auth_obj = result.get("auth", {})
        auth_data = auth_obj if isinstance(auth_obj, dict) else {}

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_VAULT,
            result={
                "status": "success",
                "payload": {
                    "renewable": auth_data.get("renewable", False),
                    "lease_duration": auth_data.get("lease_duration", 0),
                },
                "correlation_id": str(correlation_id),
            },
        )


__all__: list[str] = ["MixinVaultToken"]
