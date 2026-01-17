# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault retry mixin for HandlerVault.

Provides retry logic and error handling methods for vault operations
with exponential backoff and circuit breaker integration.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar
from uuid import UUID

import hvac

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    InfraVaultError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    SecretResolutionError,
)
from omnibase_infra.handlers.models import ModelOperationContext, ModelRetryState
from omnibase_infra.handlers.models.vault import ModelVaultHandlerConfig

T = TypeVar("T")

logger = logging.getLogger(__name__)


class MixinVaultRetry:
    """Mixin providing retry and error handling for HandlerVault.

    Contains methods for:
    - Retry loop with exponential backoff
    - Error context creation
    - Circuit breaker failure tracking
    - Error type-specific handling (timeout, forbidden, invalid path, down)
    """

    # Instance attributes (declared for type checking)
    _config: ModelVaultHandlerConfig | None
    _executor: ThreadPoolExecutor | None
    _circuit_breaker_initialized: bool
    _circuit_breaker_lock: asyncio.Lock

    # Methods from MixinAsyncCircuitBreaker that will be available at runtime
    async def _check_circuit_breaker(
        self, operation: str, correlation_id: UUID | None = None
    ) -> None:
        """Check circuit breaker - provided by MixinAsyncCircuitBreaker."""

    async def _record_circuit_failure(
        self, operation: str, correlation_id: UUID | None = None
    ) -> None:
        """Record circuit failure - provided by MixinAsyncCircuitBreaker."""

    async def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker - provided by MixinAsyncCircuitBreaker."""

    def _create_vault_error_context(
        self, operation: str, correlation_id: UUID
    ) -> ModelInfraErrorContext:
        """Create standard error context for Vault operations.

        Args:
            operation: Operation name
            correlation_id: Correlation ID for tracing

        Returns:
            ModelInfraErrorContext configured for Vault transport
        """
        return ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation=operation,
            target_name="vault_handler",
            correlation_id=correlation_id,
            namespace=self._config.namespace if self._config else None,
        )

    async def _record_circuit_failure_if_final(
        self,
        retry_state: ModelRetryState,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Record circuit breaker failure only on final retry attempt.

        Args:
            retry_state: Current retry state
            operation: Operation name
            correlation_id: Correlation ID for tracing
        """
        if not retry_state.is_retriable() and self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)

    async def _handle_vault_timeout(
        self,
        error: TimeoutError,
        retry_state: ModelRetryState,
        operation: str,
        correlation_id: UUID,
        timeout_seconds: float,
    ) -> None:
        """Handle timeout error - raise InfraTimeoutError if retries exhausted.

        Args:
            error: The timeout error
            retry_state: Current retry state (after next_attempt called)
            operation: Operation name
            correlation_id: Correlation ID for tracing
            timeout_seconds: Timeout value for error message

        Raises:
            InfraTimeoutError: If retries exhausted
        """
        await self._record_circuit_failure_if_final(
            retry_state, operation, correlation_id
        )
        if not retry_state.is_retriable():
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation=operation,
                target_name="vault_handler",
                correlation_id=correlation_id,
                timeout_seconds=timeout_seconds,
            )
            raise InfraTimeoutError(
                f"Vault operation timed out after {timeout_seconds}s",
                context=timeout_ctx,
            ) from error

    async def _handle_vault_forbidden(
        self,
        error: hvac.exceptions.Forbidden,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Handle forbidden error - always raise, no retry.

        Args:
            error: The forbidden exception
            operation: Operation name
            correlation_id: Correlation ID for tracing

        Raises:
            InfraAuthenticationError: Always
        """
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation, correlation_id)
        ctx = self._create_vault_error_context(operation, correlation_id)
        raise InfraAuthenticationError(
            "Vault operation forbidden - check token permissions",
            context=ctx,
        ) from error

    def _handle_vault_invalid_path(
        self,
        error: hvac.exceptions.InvalidPath,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Handle invalid path error - always raise, no retry, no circuit breaker.

        Args:
            error: The invalid path exception
            operation: Operation name
            correlation_id: Correlation ID for tracing

        Raises:
            SecretResolutionError: Always
        """
        ctx = self._create_vault_error_context(operation, correlation_id)
        raise SecretResolutionError(
            "Secret path not found or invalid",
            context=ctx,
        ) from error

    async def _handle_vault_down(
        self,
        error: hvac.exceptions.VaultDown,
        retry_state: ModelRetryState,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Handle Vault down error - raise InfraUnavailableError if retries exhausted.

        Args:
            error: The VaultDown exception
            retry_state: Current retry state (after next_attempt called)
            operation: Operation name
            correlation_id: Correlation ID for tracing

        Raises:
            InfraUnavailableError: If retries exhausted
        """
        await self._record_circuit_failure_if_final(
            retry_state, operation, correlation_id
        )
        if not retry_state.is_retriable():
            ctx = self._create_vault_error_context(operation, correlation_id)
            raise InfraUnavailableError(
                "Vault server is unavailable",
                context=ctx,
            ) from error

    async def _handle_vault_general_error(
        self,
        error: Exception,
        retry_state: ModelRetryState,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Handle general error - raise InfraConnectionError if retries exhausted.

        Args:
            error: The exception
            retry_state: Current retry state (after next_attempt called)
            operation: Operation name
            correlation_id: Correlation ID for tracing

        Raises:
            InfraConnectionError: If retries exhausted
        """
        await self._record_circuit_failure_if_final(
            retry_state, operation, correlation_id
        )
        if not retry_state.is_retriable():
            ctx = self._create_vault_error_context(operation, correlation_id)
            raise InfraConnectionError(
                f"Vault operation failed: {type(error).__name__}",
                context=ctx,
            ) from error

    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        correlation_id: UUID,
    ) -> T:
        """Execute operation with exponential backoff retry logic and circuit breaker.

        Thread Pool Integration:
            All hvac operations (which are synchronous) are executed in a dedicated
            thread pool via loop.run_in_executor(). This prevents blocking the async
            event loop and allows concurrent Vault operations up to max_workers limit.

        Circuit breaker integration (via MixinAsyncCircuitBreaker):
            - Checks circuit state before execution (raises if OPEN)
            - Records success/failure for circuit state management
            - Allows test request in HALF_OPEN state

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
            InfraUnavailableError: If circuit breaker is OPEN
        """
        if self._config is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation=operation,
                target_name="vault_handler",
                correlation_id=correlation_id,
            )
            raise InfraVaultError("Vault config not initialized", context=ctx)

        # Check circuit breaker before execution (async mixin pattern)
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker(operation, correlation_id)

        # Initialize retry state with config values
        retry_config = self._config.retry
        retry_state = ModelRetryState(
            attempt=0,
            max_attempts=retry_config.max_attempts,
            delay_seconds=retry_config.initial_backoff_seconds,
            backoff_multiplier=retry_config.exponential_base,
        )

        # Create operation context for tracking
        op_context = ModelOperationContext.create(
            operation_name=operation,
            correlation_id=correlation_id,
            timeout_seconds=self._config.timeout_seconds,
            metadata={"namespace": self._config.namespace or "default"},
        )

        while retry_state.is_retriable():
            try:
                result = await self._execute_vault_operation(func, op_context)
                await self._record_circuit_success()
                return result
            except TimeoutError as e:
                retry_state = retry_state.next_attempt(
                    error_message=f"Timeout after {op_context.timeout_seconds}s",
                    max_delay_seconds=retry_config.max_backoff_seconds,
                )
                await self._handle_vault_timeout(
                    e,
                    retry_state,
                    operation,
                    correlation_id,
                    op_context.timeout_seconds,
                )
            except hvac.exceptions.Forbidden as e:
                await self._handle_vault_forbidden(e, operation, correlation_id)
            except hvac.exceptions.InvalidPath as e:
                self._handle_vault_invalid_path(e, operation, correlation_id)
            except hvac.exceptions.VaultDown as e:
                retry_state = retry_state.next_attempt(
                    error_message=f"Vault down: {type(e).__name__}",
                    max_delay_seconds=retry_config.max_backoff_seconds,
                )
                await self._handle_vault_down(e, retry_state, operation, correlation_id)
            except Exception as e:
                retry_state = retry_state.next_attempt(
                    error_message=f"Unexpected error: {type(e).__name__}",
                    max_delay_seconds=retry_config.max_backoff_seconds,
                )
                await self._handle_vault_general_error(
                    e, retry_state, operation, correlation_id
                )

            self._log_retry_attempt(retry_state, operation, correlation_id)
            await asyncio.sleep(retry_state.delay_seconds)

        # Should never reach here, but satisfy type checker
        ctx = self._create_vault_error_context(operation, correlation_id)
        if retry_state.last_error is not None:
            raise InfraVaultError(
                f"Vault operation retry exhausted: {retry_state.last_error}",
                context=ctx,
                retry_count=retry_state.attempt,
                last_error=retry_state.last_error,
            )
        raise InfraVaultError(
            "Vault retry loop completed without result",
            context=ctx,
            retry_count=retry_state.attempt,
        )

    async def _execute_vault_operation(
        self, func: Callable[[], T], op_context: ModelOperationContext
    ) -> T:
        """Execute a vault operation in thread pool with timeout.

        Args:
            func: Callable to execute (synchronous hvac method)
            op_context: Operation context with timeout

        Returns:
            Result from func()
        """
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(self._executor, func),
            timeout=op_context.timeout_seconds,
        )

    async def _record_circuit_success(self) -> None:
        """Record success for circuit breaker if initialized."""
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

    def _log_retry_attempt(
        self,
        retry_state: ModelRetryState,
        operation: str,
        correlation_id: UUID,
    ) -> None:
        """Log retry attempt details.

        Args:
            retry_state: Current retry state
            operation: Operation name
            correlation_id: Correlation ID for tracing
        """
        logger.debug(
            "Retrying Vault operation",
            extra={
                "operation": operation,
                "attempt": retry_state.attempt,
                "max_attempts": retry_state.max_attempts,
                "backoff_seconds": retry_state.delay_seconds,
                "last_error": retry_state.last_error,
                "correlation_id": str(correlation_id),
            },
        )


__all__: list[str] = ["MixinVaultRetry"]
