# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol lifecycle executor for ONEX Infrastructure.

This module provides the ProtocolLifecycleExecutor class for managing protocol
handler lifecycle operations including shutdown, health checks, and priority
ordering.

The ProtocolLifecycleExecutor is a reusable utility extracted from
RuntimeHostProcess to enable lifecycle management in other contexts (e.g., test
fixtures, standalone management, future node implementations).

Example Usage:
    ```python
    from omnibase_infra.runtime.protocol_lifecycle_executor import (
        ProtocolLifecycleExecutor,
    )

    executor = ProtocolLifecycleExecutor(health_check_timeout_seconds=10.0)

    # Check handler health - returns ModelHealthCheckResult
    result = await executor.check_handler_health("http", handler)
    print(f"Handler: {result.handler_type}, Healthy: {result.healthy}")

    # Shutdown handler - returns ModelLifecycleResult
    result = await executor.shutdown_handler("http", handler)
    if result.success:
        print(f"Handler {result.handler_type} shutdown successfully")
    else:
        print(f"Handler failed: {result.error_message}")

    # Shutdown all handlers by priority - returns ModelBatchLifecycleResult
    batch_result = await executor.shutdown_handlers_by_priority(handlers)
    print(f"Succeeded: {batch_result.succeeded_handlers}")
    print(f"Failed: {batch_result.failure_count}")
    ```
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from omnibase_infra.runtime.models import (
    ModelBatchLifecycleResult,
    ModelHealthCheckResult,
    ModelLifecycleResult,
)

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolContainerAware

logger = logging.getLogger(__name__)

# Health check timeout bounds (per ModelLifecycleSubcontract)
MIN_HEALTH_CHECK_TIMEOUT = 1.0
MAX_HEALTH_CHECK_TIMEOUT = 60.0
DEFAULT_HEALTH_CHECK_TIMEOUT = 5.0


class ProtocolLifecycleExecutor:
    """Executes protocol lifecycle operations (shutdown, health check, priority).

    Reusable utility for managing protocol lifecycles. Used by
    RuntimeHostProcess and can be used by future node implementations.

    This class provides:
    - Priority-based shutdown ordering (higher priority handlers shutdown first)
    - Health check execution with configurable timeouts
    - Parallel shutdown within priority groups for performance

    Thread Safety:
        This class is thread-safe for concurrent method calls.

        **Instance State**:
        - `_health_check_timeout_seconds`: Immutable after initialization (read-only)

        **Method Safety**:
        - `get_shutdown_priority()`: Static method, no shared state
        - `shutdown_handler()`: Async, operates on provided handler only
        - `check_handler_health()`: Async, operates on provided handler only
        - `shutdown_handlers_by_priority()`: Async, manages parallel execution internally

        **Handler Safety Requirement**:
        While this executor is thread-safe, callers must ensure that the same
        handler instance is not passed to multiple concurrent operations (e.g.,
        do not call `shutdown_handler` on the same handler from two coroutines).
        Each handler should only be shut down once.

        **Safe Pattern**:
        ```python
        executor = ProtocolLifecycleExecutor()
        # Multiple executors can share handlers dict safely
        await executor.shutdown_handlers_by_priority(handlers)
        ```

        **Unsafe Pattern** (avoid):
        ```python
        # DO NOT shutdown same handler from multiple coroutines
        await asyncio.gather(
            executor.shutdown_handler("db", db_handler),  # First call
            executor.shutdown_handler("db", db_handler),  # Duplicate - unsafe
        )
        ```

    Attributes:
        health_check_timeout_seconds: Default timeout for health checks (1-60 seconds).

    Example:
        ```python
        executor = ProtocolLifecycleExecutor(health_check_timeout_seconds=10.0)

        # Get shutdown priority for ordering
        priority = ProtocolLifecycleExecutor.get_shutdown_priority(handler)

        # Check individual handler health - returns ModelHealthCheckResult
        result = await executor.check_handler_health("db", handler)
        if result.healthy:
            print(f"Handler {result.handler_type} is healthy")

        # Shutdown with error handling - returns ModelLifecycleResult
        result = await executor.shutdown_handler("db", handler)
        if result.success:
            print(f"Shutdown complete for {result.handler_type}")
        else:
            print(f"Shutdown failed: {result.error_message}")
        ```
    """

    def __init__(
        self, health_check_timeout_seconds: float = DEFAULT_HEALTH_CHECK_TIMEOUT
    ) -> None:
        """Initialize the protocol lifecycle executor.

        Args:
            health_check_timeout_seconds: Default timeout for health checks.
                Valid range: 1-60 seconds per ModelLifecycleSubcontract.
                Values outside this range are clamped with a warning.
        """
        # Validate and clamp timeout to valid bounds
        if (
            health_check_timeout_seconds < MIN_HEALTH_CHECK_TIMEOUT
            or health_check_timeout_seconds > MAX_HEALTH_CHECK_TIMEOUT
        ):
            logger.warning(
                "health_check_timeout_seconds out of valid range, clamping",
                extra={
                    "original_value": health_check_timeout_seconds,
                    "min_value": MIN_HEALTH_CHECK_TIMEOUT,
                    "max_value": MAX_HEALTH_CHECK_TIMEOUT,
                    "clamped_value": max(
                        MIN_HEALTH_CHECK_TIMEOUT,
                        min(health_check_timeout_seconds, MAX_HEALTH_CHECK_TIMEOUT),
                    ),
                },
            )
            health_check_timeout_seconds = max(
                MIN_HEALTH_CHECK_TIMEOUT,
                min(health_check_timeout_seconds, MAX_HEALTH_CHECK_TIMEOUT),
            )

        self._health_check_timeout_seconds: float = health_check_timeout_seconds

    @property
    def health_check_timeout_seconds(self) -> float:
        """Return the configured health check timeout.

        Returns:
            The timeout in seconds for health checks.
        """
        return self._health_check_timeout_seconds

    @staticmethod
    def get_shutdown_priority(handler: ProtocolContainerAware) -> int:
        """Get shutdown priority for a handler.

        Returns the shutdown priority for the given handler. Handlers with higher
        priority values are shutdown before handlers with lower priority values.

        This method uses duck typing to check if the handler implements
        shutdown_priority(). If not, returns default priority of 0.

        Shutdown Priority Guidelines:
            - Higher values = shutdown first
            - Consumers should have higher priority than producers (e.g., 100 vs 50)
            - Connections should have higher priority than connection pools (e.g., 80 vs 40)
            - Downstream resources should shutdown before upstream resources

        Example Priority Scheme:
            - 100: Consumers (Kafka consumers, event subscribers)
            - 80: Active connections (HTTP clients, DB connections)
            - 50: Producers (Kafka producers, event publishers)
            - 40: Connection pools (DB pools, HTTP connection pools)
            - 0: Default (handlers without shutdown_priority)

        Args:
            handler: The handler instance to get priority for.

        Returns:
            Shutdown priority. Higher values shutdown first. Default is 0.
        """
        if hasattr(handler, "shutdown_priority"):
            try:
                priority = handler.shutdown_priority()
                if isinstance(priority, int):
                    return priority
                logger.warning(
                    "Handler shutdown_priority() returned non-int, using default",
                    extra={
                        "handler_class": type(handler).__name__,
                        "returned_type": type(priority).__name__,
                    },
                )
            except Exception as e:
                logger.warning(
                    "Error calling handler shutdown_priority(), using default",
                    extra={
                        "handler_class": type(handler).__name__,
                        "error": str(e),
                    },
                )
        return 0

    async def shutdown_handler(
        self,
        handler_type: str,
        handler: ProtocolContainerAware,
    ) -> ModelLifecycleResult:
        """Shutdown a single handler with error handling.

        This method performs individual handler shutdown with comprehensive error
        handling to ensure all handlers get a chance to cleanup even if one fails.

        Args:
            handler_type: The handler type identifier.
            handler: The handler instance to shutdown.

        Returns:
            ModelLifecycleResult with handler_type, success status, and optional
            error_message if the shutdown failed.
        """
        try:
            if hasattr(handler, "shutdown"):
                await handler.shutdown()
                logger.debug(
                    "Handler shutdown completed",
                    extra={"handler_type": handler_type},
                )
                return ModelLifecycleResult.succeeded(handler_type)
            else:
                # Handler doesn't implement shutdown - considered successful
                logger.debug(
                    "Handler has no shutdown method, skipping",
                    extra={"handler_type": handler_type},
                )
                return ModelLifecycleResult.succeeded(handler_type)
        except Exception as e:
            # Log exception but return failure status instead of raising
            # This ensures all handlers get a chance to cleanup even if one fails
            logger.exception(
                "Error shutting down handler",
                extra={"handler_type": handler_type, "error": str(e)},
            )
            return ModelLifecycleResult.failed(handler_type, str(e))

    async def check_handler_health(
        self,
        handler_type: str,
        handler: ProtocolContainerAware,
        timeout_seconds: float = -1.0,
    ) -> ModelHealthCheckResult:
        """Check health of a single handler with timeout.

        This method performs an individual handler health check with a configurable
        timeout to prevent slow handlers from blocking the overall health check.

        Args:
            handler_type: The handler type identifier.
            handler: The handler instance to check.
            timeout_seconds: Override timeout for this specific check. If negative
                (default: -1.0), uses the configured health_check_timeout_seconds.

        Returns:
            ModelHealthCheckResult with handler_type, healthy status, and details.
        """
        # Use provided timeout or fall back to configured instance timeout
        # Negative value signals "use default from config"
        effective_timeout = (
            timeout_seconds
            if timeout_seconds > 0
            else self._health_check_timeout_seconds
        )

        try:
            if hasattr(handler, "health_check"):
                handler_health = await asyncio.wait_for(
                    handler.health_check(),
                    timeout=effective_timeout,
                )
                return ModelHealthCheckResult.from_handler_response(
                    handler_type=handler_type,
                    health_response=handler_health,
                )
            else:
                # Handler doesn't implement health_check - assume healthy
                return ModelHealthCheckResult.no_health_check_result(handler_type)
        except TimeoutError:
            logger.warning(
                "Handler health check timed out",
                extra={
                    "handler_type": handler_type,
                    "timeout_seconds": effective_timeout,
                },
            )
            return ModelHealthCheckResult.timeout_result(
                handler_type=handler_type,
                timeout_seconds=effective_timeout,
            )
        except Exception as e:
            logger.warning(
                "Handler health check failed",
                extra={"handler_type": handler_type, "error": str(e)},
            )
            return ModelHealthCheckResult.error_result(
                handler_type=handler_type,
                error=str(e),
            )

    async def shutdown_handlers_by_priority(
        self,
        handlers: dict[str, ProtocolContainerAware],
    ) -> ModelBatchLifecycleResult:
        """Shutdown all handlers grouped by priority (higher first, parallel within group).

        Handlers are shutdown in priority order, with higher priority handlers
        shutting down first. Within the same priority level, handlers are
        shutdown in parallel for performance.

        Priority is determined by the handler's shutdown_priority() method:
        - Higher values = shutdown first
        - Handlers without shutdown_priority() get default priority of 0

        Recommended Priority Scheme:
            - 100: Consumers (stop receiving before stopping producers)
            - 80: Active connections (close before closing pools)
            - 50: Producers (stop producing before closing pools)
            - 40: Connection pools (close last)
            - 0: Default for handlers without explicit priority

        Args:
            handlers: Dict mapping handler_type to handler instance.

        Returns:
            ModelBatchLifecycleResult with all results, succeeded_handlers,
            and failed_handlers categorized.
        """
        if not handlers:
            return ModelBatchLifecycleResult.empty()

        # Group handlers by priority
        priority_groups: dict[int, list[tuple[str, ProtocolContainerAware]]] = {}
        for handler_type, handler in handlers.items():
            priority = self.get_shutdown_priority(handler)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append((handler_type, handler))

        # Sort priorities in descending order (higher priority first)
        sorted_priorities = sorted(priority_groups.keys(), reverse=True)

        # Track all results
        all_results: list[ModelLifecycleResult] = []

        # Shutdown each priority group sequentially, handlers within group in parallel
        for priority in sorted_priorities:
            handlers_in_group = priority_groups[priority]

            logger.debug(
                "Shutting down handler priority group",
                extra={
                    "priority": priority,
                    "handlers": [h[0] for h in handlers_in_group],
                },
            )

            shutdown_tasks = [
                self.shutdown_handler(handler_type, handler)
                for handler_type, handler in handlers_in_group
            ]
            results = await asyncio.gather(*shutdown_tasks)

            # Collect results
            all_results.extend(results)

        # Build batch result
        batch_result = ModelBatchLifecycleResult.from_results(all_results)

        # Log summary
        if batch_result.has_failures:
            for failed in batch_result.failed_handlers:
                logger.error(
                    "Handler shutdown failed",
                    extra={
                        "handler_type": failed.handler_type,
                        "error": failed.error_message,
                    },
                )

        logger.info(
            "Priority-based handler shutdown completed",
            extra={
                "succeeded_handlers": batch_result.succeeded_handlers,
                "failed_handlers": [
                    f.handler_type for f in batch_result.failed_handlers
                ],
                "total_handlers": batch_result.total_count,
                "success_count": batch_result.success_count,
                "failure_count": batch_result.failure_count,
                "priority_groups": {
                    p: [h[0] for h in handlers_in_group]
                    for p, handlers_in_group in priority_groups.items()
                },
            },
        )

        return batch_result


__all__: list[str] = [
    "DEFAULT_HEALTH_CHECK_TIMEOUT",
    "MAX_HEALTH_CHECK_TIMEOUT",
    "MIN_HEALTH_CHECK_TIMEOUT",
    "ProtocolLifecycleExecutor",
]
