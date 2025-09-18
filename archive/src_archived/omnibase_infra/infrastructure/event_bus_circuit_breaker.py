"""Event Bus Circuit Breaker for RedPanda Reliability.

Implements circuit breaker pattern for RedPanda event publishing to handle:
- RedPanda service unavailability
- Network failures and timeouts
- Graceful degradation with event queuing
- Dead letter queue for failed events

Following ONEX infrastructure reliability patterns with fail-fast behavior
when circuit is open but graceful degradation for non-critical operations.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.model.core.model_onex_event import ModelOnexEvent

# Import new environment configuration model
from ..models.infrastructure.model_circuit_breaker_environment_config import (
    ModelCircuitBreakerConfig,
    ModelCircuitBreakerEnvironmentConfig,
)


class CircuitBreakerState(Enum):
    """Circuit breaker states for event publishing reliability."""

    CLOSED = "closed"  # Normal operation - events published directly
    OPEN = "open"  # Failure state - events queued or dropped based on policy
    HALF_OPEN = "half_open"  # Testing state - limited event publishing to test recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for event bus circuit breaker with memory management.

    Note: This dataclass is maintained for backward compatibility.
    For environment-specific configuration, use ModelCircuitBreakerEnvironmentConfig.
    """

    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout: int = 60  # Seconds before transitioning to half-open
    success_threshold: int = 3  # Successes needed in half-open to close
    timeout_seconds: int = 30  # Event publishing timeout
    max_queue_size: int = 1000  # Max queued events when circuit is open
    dead_letter_enabled: bool = True  # Enable dead letter queue for failed events
    graceful_degradation: bool = True  # Allow operations to continue without events

    # Memory management configuration
    max_dead_letter_size: int = 500  # Max dead letter queue entries
    dead_letter_ttl_hours: int = 24  # Dead letter entry expiration (hours)
    cleanup_interval_seconds: int = 300  # Memory cleanup interval (5 minutes)
    memory_monitor_enabled: bool = True  # Enable memory usage monitoring

    @classmethod
    def from_environment_config(
        cls, env_config: ModelCircuitBreakerConfig,
    ) -> "CircuitBreakerConfig":
        """Create CircuitBreakerConfig from environment-specific configuration."""
        return cls(
            failure_threshold=env_config.failure_threshold,
            recovery_timeout=env_config.recovery_timeout,
            success_threshold=env_config.success_threshold,
            timeout_seconds=env_config.timeout_seconds,
            max_queue_size=env_config.max_queue_size,
            dead_letter_enabled=env_config.dead_letter_enabled,
            graceful_degradation=env_config.graceful_degradation,
        )


@dataclass
class EventBusMetrics:
    """Metrics tracking for event bus circuit breaker."""

    total_events: int = 0
    successful_events: int = 0
    failed_events: int = 0
    queued_events: int = 0
    dropped_events: int = 0
    dead_letter_events: int = 0
    circuit_opens: int = 0
    circuit_closes: int = 0
    last_failure: datetime | None = None
    last_success: datetime | None = None


class EventBusCircuitBreaker:
    """
    Circuit breaker for RedPanda event bus reliability.

    Provides resilient event publishing with:
    - Automatic failure detection and circuit opening
    - Graceful degradation with event queuing
    - Dead letter queue for permanently failed events
    - Recovery testing and automatic circuit closing
    - Comprehensive metrics and observability
    - Environment-specific configuration support
    """

    def __init__(self, config: CircuitBreakerConfig | ModelCircuitBreakerConfig):
        """Initialize circuit breaker with configuration and memory management.

        Args:
            config: Circuit breaker configuration (legacy dataclass or new Pydantic model)
        """
        # Convert Pydantic model to dataclass for internal use (backward compatibility)
        if isinstance(config, ModelCircuitBreakerConfig):
            self.config = CircuitBreakerConfig.from_environment_config(config)
        else:
            self.config = config

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.event_queue: list[ModelOnexEvent] = []
        self.dead_letter_queue: list[dict[str, Any]] = []
        self.metrics = EventBusMetrics()
        self.logger = logging.getLogger(f"{__name__}.EventBusCircuitBreaker")
        self._lock = asyncio.Lock()

        # Memory management components
        self._memory_cleanup_task: asyncio.Task | None = None
        self._start_memory_cleanup()
        self._last_memory_cleanup = time.time()

    def _start_memory_cleanup(self):
        """Start background memory cleanup task."""
        if self.config.memory_monitor_enabled:
            self._memory_cleanup_task = asyncio.create_task(self._memory_cleanup_loop())

    async def _memory_cleanup_loop(self):
        """Background loop for memory management and cleanup."""
        try:
            while True:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                await self._perform_memory_cleanup()
        except asyncio.CancelledError:
            self.logger.info("Memory cleanup task cancelled")
        except Exception as e:
            self.logger.error(f"Memory cleanup error: {e}")

    async def _perform_memory_cleanup(self):
        """Perform memory cleanup operations."""
        async with self._lock:
            initial_dead_letter_size = len(self.dead_letter_queue)
            initial_queue_size = len(self.event_queue)

            # Clean up expired dead letter entries
            await self._cleanup_expired_dead_letters()

            # Enforce dead letter queue size limit
            await self._enforce_dead_letter_size_limit()

            # Cleanup statistics
            dead_letter_cleaned = initial_dead_letter_size - len(self.dead_letter_queue)

            if dead_letter_cleaned > 0:
                self.logger.info(
                    f"Memory cleanup completed: {dead_letter_cleaned} dead letter entries removed",
                )

            self._last_memory_cleanup = time.time()

    async def _cleanup_expired_dead_letters(self):
        """Remove expired entries from dead letter queue."""
        current_time = datetime.now()
        ttl_delta = timedelta(hours=self.config.dead_letter_ttl_hours)

        # Filter out expired entries
        self.dead_letter_queue = [
            entry
            for entry in self.dead_letter_queue
            if self._is_dead_letter_entry_valid(entry, current_time, ttl_delta)
        ]

    def _is_dead_letter_entry_valid(
        self, entry: dict[str, Any], current_time: datetime, ttl_delta: timedelta,
    ) -> bool:
        """Check if a dead letter entry is still valid (not expired)."""
        try:
            entry_time = datetime.fromisoformat(entry.get("timestamp", ""))
            return (current_time - entry_time) < ttl_delta
        except (ValueError, TypeError):
            # Invalid timestamp - remove entry
            return False

    async def _enforce_dead_letter_size_limit(self):
        """Enforce maximum dead letter queue size."""
        if len(self.dead_letter_queue) > self.config.max_dead_letter_size:
            # Remove oldest entries (FIFO)
            excess_count = (
                len(self.dead_letter_queue) - self.config.max_dead_letter_size
            )
            self.dead_letter_queue = self.dead_letter_queue[excess_count:]
            self.logger.warning(
                f"Dead letter queue size limit exceeded, removed {excess_count} oldest entries",
            )

    async def close(self):
        """Close circuit breaker and cleanup resources."""
        # Cancel memory cleanup task
        if self._memory_cleanup_task and not self._memory_cleanup_task.done():
            self._memory_cleanup_task.cancel()
            try:
                await self._memory_cleanup_task
            except asyncio.CancelledError:
                pass

        # Clear all queues to free memory
        self.event_queue.clear()
        self.dead_letter_queue.clear()

        self.logger.info("Circuit breaker closed and resources cleaned up")

    @classmethod
    def from_environment(
        cls,
        environment_config: ModelCircuitBreakerEnvironmentConfig | None = None,
        environment: str | None = None,
    ) -> "EventBusCircuitBreaker":
        """Create circuit breaker with environment-specific configuration.

        Args:
            environment_config: Environment configuration model (optional)
            environment: Target environment name (optional, detected from ENV if not provided)

        Returns:
            EventBusCircuitBreaker configured for the environment

        Raises:
            OnexError: If environment detection or configuration fails
        """
        # Use provided environment config or create default
        if environment_config is None:
            environment_config = (
                ModelCircuitBreakerEnvironmentConfig.create_default_config()
            )

        # Detect environment from ENV variable if not provided
        if environment is None:
            environment = cls._detect_environment()

        try:
            env_config = environment_config.get_config_for_environment(
                environment,
                default_environment="development",
            )

            # Create circuit breaker with environment-specific config
            instance = cls(env_config)
            instance.logger.info(
                f"Circuit breaker initialized for environment: {environment}",
            )
            return instance

        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.CONFIGURATION_ERROR,
                message=f"Failed to create environment-specific circuit breaker: {e!s}",
            ) from e

    @staticmethod
    def _detect_environment() -> str:
        """Detect current deployment environment from environment variables.

        Returns:
            Environment name (production, staging, or development)
        """
        # Check multiple common environment variable patterns
        env_vars_to_check = [
            "ENVIRONMENT",
            "ENV",
            "DEPLOYMENT_ENV",
            "NODE_ENV",
            "OMNIBASE_ENV",
            "ONEX_ENV",
        ]

        for env_var in env_vars_to_check:
            env_value = os.getenv(env_var)
            if env_value:
                env_value = env_value.lower()
                # Map common variations to standard environment names
                if env_value in ["prod", "production", "live"]:
                    return "production"
                if env_value in ["stage", "staging", "stg"]:
                    return "staging"
                if env_value in ["dev", "development", "local"]:
                    return "development"
                if env_value in ["test", "testing"]:
                    return "development"  # Map test to development config

        # Default to development if no environment detected
        return "development"

    async def publish_event(
        self, event: ModelOnexEvent, publisher_func: Callable,
    ) -> bool:
        """
        Publish event through circuit breaker protection.

        Args:
            event: Event to publish
            publisher_func: Async function to publish event

        Returns:
            bool: True if published successfully, False if queued or dropped

        Raises:
            OnexError: For critical failures that should fail-fast
        """
        async with self._lock:
            self.metrics.total_events += 1

            # Check circuit state and handle accordingly
            if self.state == CircuitBreakerState.OPEN:
                return await self._handle_open_circuit(event)
            if self.state == CircuitBreakerState.HALF_OPEN:
                return await self._handle_half_open_circuit(event, publisher_func)
            # CLOSED
            return await self._handle_closed_circuit(event, publisher_func)

    async def _handle_closed_circuit(
        self, event: ModelOnexEvent, publisher_func: Callable,
    ) -> bool:
        """Handle event publishing when circuit is closed (normal operation)."""
        try:
            # Attempt to publish event with timeout
            await asyncio.wait_for(
                publisher_func(event), timeout=self.config.timeout_seconds,
            )

            # Success - reset failure count and update metrics
            self.failure_count = 0
            self.metrics.successful_events += 1
            self.metrics.last_success = datetime.now()

            self.logger.debug(f"Event published successfully: {event.correlation_id}")
            return True

        except TimeoutError:
            await self._handle_failure(
                f"Event publishing timeout after {self.config.timeout_seconds}s",
            )
            return await self._queue_or_drop_event(event)

        except Exception as e:
            await self._handle_failure(f"Event publishing failed: {e!s}")
            return await self._queue_or_drop_event(event)

    async def _handle_half_open_circuit(
        self, event: ModelOnexEvent, publisher_func: Callable,
    ) -> bool:
        """Handle event publishing when circuit is half-open (testing recovery)."""
        try:
            # Attempt limited publishing to test recovery
            await asyncio.wait_for(
                publisher_func(event), timeout=self.config.timeout_seconds,
            )

            # Success in half-open state
            self.success_count += 1
            self.metrics.successful_events += 1
            self.metrics.last_success = datetime.now()

            self.logger.info(
                f"Half-open success {self.success_count}/{self.config.success_threshold}",
            )

            # Check if we can close the circuit
            if self.success_count >= self.config.success_threshold:
                await self._close_circuit()

            return True

        except Exception as e:
            # Failure in half-open - immediately open circuit again
            await self._open_circuit(f"Half-open test failed: {e!s}")
            return await self._queue_or_drop_event(event)

    async def _handle_open_circuit(self, event: ModelOnexEvent) -> bool:
        """Handle event when circuit is open (failure state)."""
        # Check if we should transition to half-open for recovery testing
        if self._should_attempt_reset():
            await self._transition_to_half_open()
            # Don't publish this event yet - queue it for safety
            return await self._queue_or_drop_event(event)

        # Circuit remains open - queue or drop event
        return await self._queue_or_drop_event(event)

    async def _handle_failure(self, error_message: str):
        """Handle event publishing failure."""
        self.failure_count += 1
        self.metrics.failed_events += 1
        self.metrics.last_failure = datetime.now()
        self.last_failure_time = time.time()

        self.logger.warning(
            f"Event publishing failure {self.failure_count}/{self.config.failure_threshold}: {error_message}",
        )

        # Open circuit if failure threshold reached
        if self.failure_count >= self.config.failure_threshold:
            await self._open_circuit(f"Failure threshold reached: {error_message}")

    async def _open_circuit(self, reason: str):
        """Open the circuit breaker."""
        if self.state != CircuitBreakerState.OPEN:
            self.state = CircuitBreakerState.OPEN
            self.metrics.circuit_opens += 1
            self.logger.error(f"Circuit breaker OPENED: {reason}")

    async def _close_circuit(self):
        """Close the circuit breaker (recovery complete)."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.metrics.circuit_closes += 1
        self.logger.info("Circuit breaker CLOSED - recovery complete")

        # Process any queued events
        await self._process_queued_events()

    async def _transition_to_half_open(self):
        """Transition circuit to half-open state for recovery testing."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.logger.info("Circuit breaker transitioned to HALF-OPEN - testing recovery")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset to half-open."""
        if self.last_failure_time is None:
            return False

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout

    async def _queue_or_drop_event(self, event: ModelOnexEvent) -> bool:
        """Queue event or drop it based on queue capacity and configuration."""
        if not self.config.graceful_degradation:
            # Fail-fast mode - raise error for critical operations
            raise OnexError(
                code=CoreErrorCode.INTEGRATION_SERVICE_UNAVAILABLE,
                message="Event bus circuit breaker open - event publishing failed",
                details={
                    "circuit_state": self.state.value,
                    "queued_events": len(self.event_queue),
                },
            )

        # Graceful degradation mode - queue if possible
        if len(self.event_queue) < self.config.max_queue_size:
            self.event_queue.append(event)
            self.metrics.queued_events += 1
            self.logger.info(
                f"Event queued (circuit {self.state.value}): {event.correlation_id}",
            )
            return False  # Not published, but queued
        # Queue full - move to dead letter queue if enabled
        if self.config.dead_letter_enabled:
            await self._add_to_dead_letter_queue(event, "Queue capacity exceeded")

        self.metrics.dropped_events += 1
        self.logger.warning(f"Event dropped - queue full: {event.correlation_id}")
        return False

    async def _add_to_dead_letter_queue(self, event: ModelOnexEvent, reason: str):
        """Add failed event to dead letter queue for later processing."""
        dead_letter_entry = {
            "event": event.model_dump(),
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "circuit_state": self.state.value,
            "retry_count": 0,
        }

        self.dead_letter_queue.append(dead_letter_entry)
        self.metrics.dead_letter_events += 1
        self.logger.info(
            f"Event added to dead letter queue: {event.correlation_id} - {reason}",
        )

    async def _process_queued_events(self):
        """Process queued events when circuit closes."""
        if not self.event_queue:
            return

        queued_count = len(self.event_queue)
        self.logger.info(
            f"Processing {queued_count} queued events after circuit recovery",
        )

        # Process events in background to avoid blocking
        asyncio.create_task(self._process_queue_background())

    async def _process_queue_background(self):
        """Background task to process queued events."""
        processed = 0
        failed = 0

        while self.event_queue and self.state == CircuitBreakerState.CLOSED:
            try:
                event = self.event_queue.pop(0)
                # TODO: Re-publish event through normal publisher
                # This would require passing the publisher function
                processed += 1

            except Exception as e:
                failed += 1
                self.logger.error(f"Failed to process queued event: {e}")

                if failed >= 3:  # Prevent infinite retry loops
                    break

        self.logger.info(
            f"Queued event processing complete: {processed} processed, {failed} failed",
        )

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state

    def get_metrics(self) -> EventBusMetrics:
        """Get current circuit breaker metrics."""
        return self.metrics

    def is_healthy(self) -> bool:
        """Check if circuit breaker is healthy for event publishing."""
        return (
            self.state == CircuitBreakerState.CLOSED
            or self.state == CircuitBreakerState.HALF_OPEN
        )

    async def reset_circuit(self):
        """Manually reset circuit breaker (for administrative purposes)."""
        async with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.logger.info("Circuit breaker manually reset to CLOSED state")

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status for monitoring."""
        return {
            "circuit_state": self.state.value,
            "is_healthy": self.is_healthy(),
            "failure_count": self.failure_count,
            "queued_events": len(self.event_queue),
            "dead_letter_events": len(self.dead_letter_queue),
            "configuration": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "max_queue_size": self.config.max_queue_size,
                "dead_letter_enabled": self.config.dead_letter_enabled,
                "graceful_degradation": self.config.graceful_degradation,
                "environment": self._detect_environment(),
                # Memory management configuration
                "max_dead_letter_size": self.config.max_dead_letter_size,
                "dead_letter_ttl_hours": self.config.dead_letter_ttl_hours,
                "cleanup_interval_seconds": self.config.cleanup_interval_seconds,
                "memory_monitor_enabled": self.config.memory_monitor_enabled,
            },
            "metrics": {
                "total_events": self.metrics.total_events,
                "successful_events": self.metrics.successful_events,
                "failed_events": self.metrics.failed_events,
                "success_rate": (
                    self.metrics.successful_events / max(self.metrics.total_events, 1)
                )
                * 100,
                "circuit_opens": self.metrics.circuit_opens,
                "circuit_closes": self.metrics.circuit_closes,
                "last_failure": (
                    self.metrics.last_failure.isoformat()
                    if self.metrics.last_failure
                    else None
                ),
                "last_success": (
                    self.metrics.last_success.isoformat()
                    if self.metrics.last_success
                    else None
                ),
            },
            "memory_management": {
                "event_queue_size": len(self.event_queue),
                "dead_letter_queue_size": len(self.dead_letter_queue),
                "dead_letter_utilization": (
                    len(self.dead_letter_queue) / self.config.max_dead_letter_size
                )
                * 100,
                "queue_utilization": (
                    len(self.event_queue) / self.config.max_queue_size
                )
                * 100,
                "memory_cleanup_enabled": self.config.memory_monitor_enabled,
                "last_cleanup": time.time() - self._last_memory_cleanup,
                "cleanup_task_running": (
                    self._memory_cleanup_task is not None
                    and not self._memory_cleanup_task.done()
                ),
            },
        }
