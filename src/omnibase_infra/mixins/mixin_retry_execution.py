# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Reusable retry execution mixin for infrastructure handlers.

This module provides a mixin class that encapsulates retry logic with
exponential backoff, circuit breaker integration, and standardized
error handling for infrastructure handlers like Consul and Vault.

Features:
    - Exponential backoff with configurable parameters
    - Circuit breaker integration via MixinAsyncCircuitBreaker
    - Error classification for determining retry eligibility
    - Standardized error context creation
    - Thread pool executor integration for sync operations

Usage:
    ```python
    from omnibase_infra.mixins import MixinRetryExecution, MixinAsyncCircuitBreaker

    class HandlerConsul(MixinAsyncCircuitBreaker, MixinRetryExecution):
        async def _my_operation(self, ...):
            result = await self._execute_with_retry(
                operation="consul.kv_get",
                func=lambda: self._client.kv.get(key),
                correlation_id=correlation_id,
                retry_config=self._config.retry,
                timeout_seconds=self._config.timeout_seconds,
            )
    ```

Design Rationale:
    This mixin extracts common retry patterns from HandlerConsul and HandlerVault
    to reduce code duplication and cyclomatic complexity. The error classification
    methods are designed to be overridden by subclasses for handler-specific
    exception types (e.g., consul.ACLPermissionDenied, hvac.exceptions.Forbidden).

See Also:
    - docs/patterns/error_recovery_patterns.md for retry pattern documentation
    - MixinAsyncCircuitBreaker for circuit breaker integration
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, cast
from uuid import UUID

from omnibase_infra.enums import EnumInfraTransportType, EnumRetryErrorCategory
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
)
from omnibase_infra.mixins.protocol_circuit_breaker_aware import (
    ProtocolCircuitBreakerAware,
)
from omnibase_infra.models.model_retry_error_classification import (
    ModelRetryErrorClassification,
)

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from omnibase_infra.handlers.models import ModelOperationContext, ModelRetryState
    from omnibase_infra.handlers.models.model_consul_retry_config import (
        ModelConsulRetryConfig,
    )
    from omnibase_infra.handlers.models.model_vault_retry_config import (
        ModelVaultRetryConfig,
    )

    RetryConfigType = ModelConsulRetryConfig | ModelVaultRetryConfig

T = TypeVar("T")

logger = logging.getLogger(__name__)


class MixinRetryExecution(ABC):
    """Mixin providing retry execution with exponential backoff and circuit breaker.

    This mixin provides the core retry logic used by infrastructure handlers.
    Subclasses must implement error classification for handler-specific exceptions.

    Required Mixin Dependencies:
        - MixinAsyncCircuitBreaker: For circuit breaker state management

    Required Instance Attributes:
        - _circuit_breaker_initialized: bool
        - _circuit_breaker_lock: asyncio.Lock (from MixinAsyncCircuitBreaker)
        - _executor: ThreadPoolExecutor | None

    Abstract Methods:
        Subclasses must implement:
        - _classify_error: Classify an exception for retry handling
        - _get_transport_type: Return the transport type for error context
        - _get_target_name: Return the target name for error context

    Example:
        ```python
        class HandlerConsul(MixinAsyncCircuitBreaker, MixinRetryExecution):
            def _classify_error(self, error: Exception, operation: str) -> ModelRetryErrorClassification:
                if isinstance(error, consul.ACLPermissionDenied):
                    return ModelRetryErrorClassification(
                        category=EnumRetryErrorCategory.AUTHENTICATION,
                        should_retry=False,
                        record_circuit_failure=True,
                        error_message="Consul ACL permission denied",
                    )
                # ... more error types
        ```
    """

    # Type hints for required attributes from other mixins/base class
    _circuit_breaker_initialized: bool
    _executor: ThreadPoolExecutor | None

    @abstractmethod
    def _classify_error(
        self, error: Exception, operation: str
    ) -> ModelRetryErrorClassification:
        """Classify an exception for retry handling.

        Subclasses must implement this to handle handler-specific exception types.

        Args:
            error: The exception to classify.
            operation: The operation name for context.

        Returns:
            ModelRetryErrorClassification with retry decision and error details.
        """
        ...

    @abstractmethod
    def _get_transport_type(self) -> EnumInfraTransportType:
        """Return the transport type for error context.

        Returns:
            The transport type enum value (e.g., CONSUL, VAULT).
        """
        ...

    @abstractmethod
    def _get_target_name(self) -> str:
        """Return the target name for error context.

        Returns:
            The target name string (e.g., "consul_handler", "vault_handler").
        """
        ...

    def _get_namespace(self) -> str | None:
        """Return the namespace for error context (optional).

        Override in subclasses that support namespaces.

        Returns:
            The namespace string or None if not applicable.
        """
        return None

    def _build_error_context(
        self, operation: str, correlation_id: UUID
    ) -> ModelInfraErrorContext:
        """Build standardized error context for infrastructure errors.

        Args:
            operation: Operation name for the error context.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            ModelInfraErrorContext with handler-specific configuration.
        """
        return ModelInfraErrorContext(
            transport_type=self._get_transport_type(),
            operation=operation,
            target_name=self._get_target_name(),
            correlation_id=correlation_id,
            namespace=self._get_namespace(),
        )

    def _as_circuit_breaker(self) -> ProtocolCircuitBreakerAware:
        """Cast self to ProtocolCircuitBreakerAware for type-safe circuit breaker access.

        This helper enables type-safe access to circuit breaker methods when
        the implementing class also inherits from MixinAsyncCircuitBreaker.

        Returns:
            Self cast as ProtocolCircuitBreakerAware for type checker satisfaction.

        Note:
            This should only be called when _circuit_breaker_initialized is True,
            which guarantees the circuit breaker methods are available.
        """
        return cast("ProtocolCircuitBreakerAware", self)

    async def _record_circuit_failure_if_enabled(
        self, operation: str, correlation_id: UUID
    ) -> None:
        """Record circuit breaker failure if circuit breaker is enabled.

        Args:
            operation: Operation name for logging.
            correlation_id: Correlation ID for tracing.
        """
        if self._circuit_breaker_initialized:
            cb = self._as_circuit_breaker()
            async with cb._circuit_breaker_lock:
                await cb._record_circuit_failure(operation, correlation_id)

    async def _reset_circuit_if_enabled(self) -> None:
        """Reset circuit breaker to closed state if enabled."""
        if self._circuit_breaker_initialized:
            cb = self._as_circuit_breaker()
            async with cb._circuit_breaker_lock:
                await cb._reset_circuit_breaker()

    async def _check_circuit_if_enabled(
        self, operation: str, correlation_id: UUID
    ) -> None:
        """Check circuit breaker state if enabled.

        Args:
            operation: Operation name for error context.
            correlation_id: Correlation ID for tracing.

        Raises:
            InfraUnavailableError: If circuit breaker is open.
        """
        if self._circuit_breaker_initialized:
            cb = self._as_circuit_breaker()
            async with cb._circuit_breaker_lock:
                await cb._check_circuit_breaker(operation, correlation_id)

    async def _handle_retriable_error(
        self,
        classification: ModelRetryErrorClassification,
        retry_state: ModelRetryState,
        max_delay_seconds: float,
        operation: str,
        correlation_id: UUID,
        op_context: ModelOperationContext,
        original_error: Exception,
    ) -> tuple[ModelRetryState, Exception | None]:
        """Handle an error that may be eligible for retry.

        Args:
            classification: The error classification result.
            retry_state: Current retry state.
            max_delay_seconds: Maximum delay cap for backoff.
            operation: Operation name for context.
            correlation_id: Correlation ID for tracing.
            op_context: Operation context with timeout info.
            original_error: The original exception.

        Returns:
            Tuple of (new_retry_state, error_to_raise_or_none).
            If error_to_raise_or_none is not None, caller should raise it.
        """
        new_state = retry_state.next_attempt(
            error_message=classification.error_message,
            max_delay_seconds=max_delay_seconds,
        )

        if not new_state.is_retriable():
            if classification.record_circuit_failure:
                await self._record_circuit_failure_if_enabled(operation, correlation_id)

            ctx = self._build_error_context(operation, correlation_id)
            error_to_raise = self._create_final_error(
                classification, ctx, op_context, original_error
            )
            return new_state, error_to_raise

        return new_state, None

    def _create_final_error(
        self,
        classification: ModelRetryErrorClassification,
        ctx: ModelInfraErrorContext,
        op_context: ModelOperationContext,
        original_error: Exception,
    ) -> Exception:
        """Create the appropriate error to raise after retries exhausted.

        Args:
            classification: Error classification result.
            ctx: Error context.
            op_context: Operation context.
            original_error: Original exception for chaining.

        Returns:
            The infrastructure error to raise.
        """
        if classification.category == EnumRetryErrorCategory.TIMEOUT:
            # Convert ModelInfraErrorContext to ModelTimeoutErrorContext for stricter typing
            # ModelTimeoutErrorContext requires transport_type and operation to be non-None
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=ctx.transport_type
                or self._get_transport_type(),  # Fallback to handler's transport type
                operation=ctx.operation or "unknown",
                target_name=ctx.target_name,
                correlation_id=ctx.correlation_id,
                timeout_seconds=op_context.timeout_seconds,
            )
            return InfraTimeoutError(
                f"Operation timed out after {op_context.timeout_seconds}s",
                context=timeout_ctx,
            )
        elif classification.category == EnumRetryErrorCategory.AUTHENTICATION:
            return InfraAuthenticationError(
                classification.error_message,
                context=ctx,
            )
        else:
            return InfraConnectionError(
                classification.error_message,
                context=ctx,
            )

    async def _handle_non_retriable_error(
        self,
        classification: ModelRetryErrorClassification,
        operation: str,
        correlation_id: UUID,
        original_error: Exception,
    ) -> Exception:
        """Handle an error that should not be retried.

        Args:
            classification: The error classification result.
            operation: Operation name for context.
            correlation_id: Correlation ID for tracing.
            original_error: The original exception.

        Returns:
            The infrastructure error to raise.
        """
        if classification.record_circuit_failure:
            await self._record_circuit_failure_if_enabled(operation, correlation_id)

        ctx = self._build_error_context(operation, correlation_id)

        if classification.category == EnumRetryErrorCategory.AUTHENTICATION:
            return InfraAuthenticationError(
                classification.error_message,
                context=ctx,
            )
        else:
            return InfraConnectionError(
                classification.error_message,
                context=ctx,
            )

    async def _log_retry_attempt(
        self, operation: str, retry_state: ModelRetryState, correlation_id: UUID
    ) -> None:
        """Log a retry attempt with standardized format.

        Args:
            operation: Operation name.
            retry_state: Current retry state after increment.
            correlation_id: Correlation ID for tracing.
        """
        logger.debug(
            "Retrying operation",
            extra={
                "operation": operation,
                "attempt": retry_state.attempt,
                "max_attempts": retry_state.max_attempts,
                "backoff_seconds": retry_state.delay_seconds,
                "last_error": retry_state.last_error,
                "correlation_id": str(correlation_id),
            },
        )


__all__: list[str] = [
    "EnumRetryErrorCategory",
    "MixinRetryExecution",
    "ModelRetryErrorClassification",
]
