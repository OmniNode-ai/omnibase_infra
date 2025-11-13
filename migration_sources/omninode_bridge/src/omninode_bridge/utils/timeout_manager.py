"""Timeout management utilities for OmniNode Bridge workflows and services."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import uuid4

T = TypeVar("T")

from circuitbreaker import circuit

logger = logging.getLogger(__name__)


@dataclass
class TimeoutConfig:
    """Configuration for timeout management."""

    operation_timeout: int = 300  # Default 5 minutes for individual operations
    global_timeout: int = 3600  # Default 1 hour for entire workflow
    grace_period: int = 30  # Grace period before hard kill
    warning_threshold: float = 0.8  # Warn when 80% of timeout reached
    enable_progressive_timeout: bool = True  # Enable progressive timeout warnings


@dataclass
class TimeoutContext:
    """Context information for a timeout operation."""

    operation_id: str
    operation_name: str
    start_time: datetime
    timeout_seconds: int
    parent_timeout_id: str | None = None
    warning_sent: bool = False
    grace_period_started: bool = False


class TimeoutManager:
    """Comprehensive timeout management for workflows and services."""

    def __init__(self, config: TimeoutConfig | None = None):
        """Initialize timeout manager.

        Args:
            config: Timeout configuration (uses defaults if None)
        """
        self.config = config or TimeoutConfig()
        self.active_timeouts: dict[str, TimeoutContext] = {}
        self.timeout_tasks: dict[str, asyncio.Task] = {}
        self.metrics = {
            "total_operations": 0,
            "timeout_warnings": 0,
            "timeout_failures": 0,
            "grace_period_recoveries": 0,
        }

    @asynccontextmanager
    async def timeout_context(
        self,
        operation_name: str,
        timeout_seconds: int | None = None,
        parent_timeout_id: str | None = None,
        enable_warnings: bool = True,
    ):
        """Context manager for timeout-protected operations.

        Args:
            operation_name: Name of the operation for logging
            timeout_seconds: Custom timeout (uses config default if None)
            parent_timeout_id: ID of parent timeout context for nested operations
            enable_warnings: Whether to send warning notifications

        Yields:
            TimeoutContext: Context information for the operation

        Raises:
            asyncio.TimeoutError: If operation exceeds timeout
        """
        operation_id = str(uuid4())
        timeout_seconds = timeout_seconds or self.config.operation_timeout

        context = TimeoutContext(
            operation_id=operation_id,
            operation_name=operation_name,
            start_time=datetime.now(UTC),
            timeout_seconds=timeout_seconds,
            parent_timeout_id=parent_timeout_id,
        )

        self.active_timeouts[operation_id] = context
        self.metrics["total_operations"] += 1

        logger.info(
            f"Starting timeout-protected operation: {operation_name} "
            f"(timeout: {timeout_seconds}s, id: {operation_id})",
        )

        # Start timeout monitoring task
        if enable_warnings and self.config.enable_progressive_timeout:
            monitor_task = asyncio.create_task(self._monitor_timeout_progress(context))
            self.timeout_tasks[operation_id] = monitor_task

        try:
            async with asyncio.timeout(timeout_seconds):
                yield context

            logger.info(
                f"Operation completed successfully: {operation_name} "
                f"(duration: {self._get_elapsed_time(context):.2f}s)",
            )

        except TimeoutError:
            self.metrics["timeout_failures"] += 1
            elapsed = self._get_elapsed_time(context)

            logger.error(
                f"Operation timed out: {operation_name} "
                f"(timeout: {timeout_seconds}s, elapsed: {elapsed:.2f}s)",
            )

            # Try grace period recovery
            if not context.grace_period_started:
                grace_recovery = await self._attempt_grace_period_recovery(context)
                if grace_recovery:
                    logger.info(
                        f"Grace period recovery successful for {operation_name}",
                    )
                    self.metrics["grace_period_recoveries"] += 1
                    return

            raise

        finally:
            # Cleanup
            self.active_timeouts.pop(operation_id, None)
            if operation_id in self.timeout_tasks:
                self.timeout_tasks[operation_id].cancel()
                self.timeout_tasks.pop(operation_id, None)

    async def timeout_operation(
        self,
        operation: Callable[..., Awaitable[T]],
        operation_name: str,
        timeout_seconds: int | None = None,
        *args,
        **kwargs,
    ) -> T:
        """Execute an operation with timeout protection.

        Args:
            operation: Async function to execute
            operation_name: Name for logging and monitoring
            timeout_seconds: Custom timeout (uses config default if None)
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation

        Raises:
            asyncio.TimeoutError: If operation exceeds timeout
        """
        async with self.timeout_context(operation_name, timeout_seconds) as context:
            return await operation(*args, **kwargs)

    @circuit(failure_threshold=5, recovery_timeout=60, expected_exception=Exception)
    async def timeout_with_circuit_breaker(
        self,
        operation: Callable[..., Awaitable[T]],
        operation_name: str,
        timeout_seconds: int | None = None,
        *args,
        **kwargs,
    ) -> T:
        """Execute operation with both timeout and circuit breaker protection.

        Args:
            operation: Async function to execute
            operation_name: Name for logging and monitoring
            timeout_seconds: Custom timeout (uses config default if None)
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of the operation

        Raises:
            asyncio.TimeoutError: If operation exceeds timeout
            Exception: If circuit breaker is open
        """
        return await self.timeout_operation(
            operation,
            operation_name,
            timeout_seconds,
            *args,
            **kwargs,
        )

    async def _monitor_timeout_progress(self, context: TimeoutContext) -> None:
        """Monitor timeout progress and send warnings."""
        try:
            while context.operation_id in self.active_timeouts:
                elapsed = self._get_elapsed_time(context)
                progress = elapsed / context.timeout_seconds

                # Send warning at threshold
                if (
                    not context.warning_sent
                    and progress >= self.config.warning_threshold
                ):
                    context.warning_sent = True
                    self.metrics["timeout_warnings"] += 1

                    remaining = context.timeout_seconds - elapsed
                    logger.warning(
                        f"Timeout warning: {context.operation_name} "
                        f"({progress:.1%} complete, {remaining:.1f}s remaining)",
                    )

                # Check if we're approaching timeout
                if progress >= 0.95:  # 95% of timeout reached
                    logger.warning(
                        f"Critical timeout warning: {context.operation_name} "
                        f"(only {context.timeout_seconds - elapsed:.1f}s remaining)",
                    )

                await asyncio.sleep(5)  # Check every 5 seconds

        except asyncio.CancelledError:
            pass  # Normal cleanup

    async def _attempt_grace_period_recovery(self, context: TimeoutContext) -> bool:
        """Attempt to recover during grace period.

        Args:
            context: Timeout context

        Returns:
            True if recovery successful, False otherwise
        """
        context.grace_period_started = True

        logger.info(
            f"Starting grace period for {context.operation_name} "
            f"({self.config.grace_period}s grace period)",
        )

        try:
            # Wait for grace period
            await asyncio.sleep(self.config.grace_period)

            # Check if operation completed during grace period
            if context.operation_id not in self.active_timeouts:
                return True  # Operation completed

        except Exception as e:
            logger.error(f"Grace period recovery failed: {e}")

        return False

    def _get_elapsed_time(self, context: TimeoutContext) -> float:
        """Get elapsed time for a timeout context."""
        return (datetime.now(UTC) - context.start_time).total_seconds()

    async def get_active_operations(self) -> dict[str, dict[str, Any]]:
        """Get information about currently active timeout-protected operations.

        Returns:
            Dictionary mapping operation IDs to operation info
        """
        operations = {}

        for op_id, context in self.active_timeouts.items():
            elapsed = self._get_elapsed_time(context)
            progress = elapsed / context.timeout_seconds
            remaining = context.timeout_seconds - elapsed

            operations[op_id] = {
                "operation_name": context.operation_name,
                "start_time": context.start_time.isoformat(),
                "timeout_seconds": context.timeout_seconds,
                "elapsed_seconds": elapsed,
                "progress_percent": progress * 100,
                "remaining_seconds": remaining,
                "warning_sent": context.warning_sent,
                "grace_period_started": context.grace_period_started,
                "parent_timeout_id": context.parent_timeout_id,
            }

        return operations

    async def get_timeout_metrics(self) -> dict[str, Any]:
        """Get timeout management metrics.

        Returns:
            Dictionary with timeout metrics
        """
        active_operations = len(self.active_timeouts)

        return {
            "config": {
                "operation_timeout": self.config.operation_timeout,
                "global_timeout": self.config.global_timeout,
                "grace_period": self.config.grace_period,
                "warning_threshold": self.config.warning_threshold,
                "progressive_timeout_enabled": self.config.enable_progressive_timeout,
            },
            "active_operations": active_operations,
            "metrics": self.metrics.copy(),
            "success_rate": (
                (self.metrics["total_operations"] - self.metrics["timeout_failures"])
                / max(self.metrics["total_operations"], 1)
            ),
        }

    async def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a specific timeout-protected operation.

        Args:
            operation_id: ID of operation to cancel

        Returns:
            True if operation was cancelled, False if not found
        """
        if operation_id in self.timeout_tasks:
            self.timeout_tasks[operation_id].cancel()
            self.timeout_tasks.pop(operation_id, None)

        if operation_id in self.active_timeouts:
            context = self.active_timeouts.pop(operation_id)
            logger.info(f"Cancelled operation: {context.operation_name}")
            return True

        return False

    async def cancel_all_operations(self) -> int:
        """Cancel all active timeout-protected operations.

        Returns:
            Number of operations cancelled
        """
        cancelled_count = 0

        # Cancel all monitoring tasks
        for task in self.timeout_tasks.values():
            task.cancel()

        # Clear all active operations
        for context in self.active_timeouts.values():
            logger.info(f"Cancelled operation: {context.operation_name}")
            cancelled_count += 1

        self.timeout_tasks.clear()
        self.active_timeouts.clear()

        logger.info(f"Cancelled {cancelled_count} active operations")
        return cancelled_count
