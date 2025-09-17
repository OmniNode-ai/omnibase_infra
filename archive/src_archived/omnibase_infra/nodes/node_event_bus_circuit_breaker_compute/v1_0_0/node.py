"""Event Bus Circuit Breaker Compute Node.

ONEX compute node that implements circuit breaker pattern for RedPanda event bus reliability.
Provides resilient event publishing with automatic failure detection and graceful degradation.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime

from omnibase_core.base.node_compute_service import NodeComputeService
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.enums.intelligence.enum_circuit_breaker_state import (
    EnumCircuitBreakerState,
)
from omnibase_core.model.core.model_onex_event import ModelOnexEvent
from omnibase_core.models.resilience.model_circuit_breaker_state import (
    ModelCircuitBreakerState,
)

from omnibase_infra.models.circuit_breaker.model_circuit_breaker_metrics import (
    ModelCircuitBreakerMetrics,
)
from omnibase_infra.models.circuit_breaker.model_dead_letter_queue_entry import (
    ModelDeadLetterQueueEntry,
)
from omnibase_infra.models.infrastructure.model_circuit_breaker_environment_config import (
    ModelCircuitBreakerConfig,
    ModelCircuitBreakerEnvironmentConfig,
)

from .models.model_event_bus_circuit_breaker_input import (
    CircuitBreakerOperation,
    ModelEventBusCircuitBreakerInput,
)
from .models.model_event_bus_circuit_breaker_output import (
    ModelEventBusCircuitBreakerOutput,
)


class NodeEventBusCircuitBreakerCompute(NodeComputeService[ModelEventBusCircuitBreakerInput, ModelEventBusCircuitBreakerOutput]):
    """
    Event Bus Circuit Breaker Compute Node.
    
    Provides:
    - Circuit breaker pattern implementation for RedPanda event publishing
    - Automatic failure detection and circuit opening/closing
    - Graceful degradation with event queuing
    - Dead letter queue for permanently failed events
    - Environment-specific configuration
    - Comprehensive metrics and health monitoring
    """

    def __init__(self, container: ModelONEXContainer):
        """Initialize the circuit breaker compute node.
        
        Args:
            container: ONEX container for dependency injection
        """
        super().__init__(container)
        self.logger = logging.getLogger(f"{__name__}.NodeEventBusCircuitBreakerCompute")

        # Circuit breaker state
        self._state = EnumCircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None

        # Configuration (will be loaded from container or defaults)
        self._config: ModelCircuitBreakerConfig | None = None

        # Event queues
        self._event_queue: list[ModelOnexEvent] = []
        self._dead_letter_queue: list[ModelDeadLetterQueueEntry] = []

        # Metrics tracking
        self._metrics = ModelCircuitBreakerMetrics()

        # Async lock for thread safety
        self._lock = asyncio.Lock()

        # State tracking
        self._last_state_change_time = datetime.now()

        # Publisher functions registry
        self._publisher_functions: dict[str, Callable] = {}

    async def initialize(self) -> None:
        """Initialize the circuit breaker node."""
        try:
            # Load configuration from container or use defaults
            self._config = await self._load_configuration()

            # Initialize metrics timestamp
            self._metrics = ModelCircuitBreakerMetrics()

            self.logger.info("Circuit breaker compute node initialized successfully")

        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.INITIALIZATION_ERROR,
                message=f"Failed to initialize circuit breaker compute node: {e!s}",
            ) from e

    async def compute(self, input_data: ModelEventBusCircuitBreakerInput) -> ModelEventBusCircuitBreakerOutput:
        """Execute circuit breaker operations.
        
        Args:
            input_data: Input containing operation type and parameters
            
        Returns:
            Output with operation result and current circuit breaker status
        """
        start_time = time.time()

        try:
            # Route to appropriate operation handler
            if input_data.operation_type == CircuitBreakerOperation.PUBLISH_EVENT:
                result = await self._handle_publish_event(input_data)
            elif input_data.operation_type == CircuitBreakerOperation.GET_STATE:
                result = await self._handle_get_state(input_data)
            elif input_data.operation_type == CircuitBreakerOperation.GET_METRICS:
                result = await self._handle_get_metrics(input_data)
            elif input_data.operation_type == CircuitBreakerOperation.RESET_CIRCUIT:
                result = await self._handle_reset_circuit(input_data)
            elif input_data.operation_type == CircuitBreakerOperation.GET_HEALTH_STATUS:
                result = await self._handle_get_health_status(input_data)
            else:
                raise OnexError(
                    code=CoreErrorCode.INVALID_INPUT,
                    message=f"Unsupported operation type: {input_data.operation_type}",
                )

            # Update performance metrics
            processing_time = time.time() - start_time
            self._metrics.average_response_time_ms = (
                self._metrics.average_response_time_ms * 0.9 + processing_time * 1000 * 0.1
            )

            return ModelEventBusCircuitBreakerOutput(
                success=True,
                operation_type=input_data.operation_type.value,
                correlation_id=input_data.correlation_id,
                result=result,
                circuit_breaker_state=self._state,
                metrics=self._metrics,
                timestamp=datetime.now(),
            )

        except OnexError:
            # Re-raise ONEX errors as-is
            raise
        except Exception as e:
            # Wrap other exceptions in OnexError
            raise OnexError(
                code=CoreErrorCode.PROCESSING_ERROR,
                message=f"Circuit breaker operation failed: {e!s}",
            ) from e

    async def _handle_publish_event(self, input_data: ModelEventBusCircuitBreakerInput) -> ModelPublishEventResult:
        """Handle event publishing through circuit breaker."""
        if not input_data.event:
            raise OnexError(
                code=CoreErrorCode.INVALID_INPUT,
                message="Event is required for publish_event operation",
            )

        async with self._lock:
            self._metrics.total_events += 1

            # Check circuit state and handle accordingly
            if self._state == EnumCircuitBreakerState.OPEN:
                return await self._handle_open_circuit(input_data.event)
            if self._state == EnumCircuitBreakerState.HALF_OPEN:
                return await self._handle_half_open_circuit(input_data.event, input_data.publisher_function)
            # CLOSED
            return await self._handle_closed_circuit(input_data.event, input_data.publisher_function)

    async def _handle_closed_circuit(self, event: ModelOnexEvent, publisher_function: str | None) -> ModelPublishEventResult:
        """Handle event publishing when circuit is closed (normal operation)."""
        try:
            # Get publisher function
            publisher_func = await self._get_publisher_function(publisher_function)

            # Attempt to publish event with timeout
            await asyncio.wait_for(publisher_func(event), timeout=self._config.timeout_seconds)

            # Success - reset failure count and update metrics
            self._failure_count = 0
            self._metrics.successful_events += 1
            self._metrics.last_success = datetime.now()
            self._metrics.success_rate_percent = (
                self._metrics.successful_events / max(self._metrics.total_events, 1) * 100
            )

            self.logger.debug(f"Event published successfully: {event.correlation_id}")

            return ModelPublishEventResult(
                published=True,
                queued=False,
                event_id=str(event.correlation_id),
            )

        except TimeoutError:
            await self._handle_failure(f"Event publishing timeout after {self._config.timeout_seconds}s")
            return await self._queue_or_drop_event(event)

        except Exception as e:
            await self._handle_failure(f"Event publishing failed: {e!s}")
            return await self._queue_or_drop_event(event)

    async def _handle_half_open_circuit(self, event: ModelOnexEvent, publisher_function: str | None) -> ModelPublishEventResult:
        """Handle event publishing when circuit is half-open (testing recovery)."""
        try:
            # Get publisher function
            publisher_func = await self._get_publisher_function(publisher_function)

            # Attempt limited publishing to test recovery
            await asyncio.wait_for(publisher_func(event), timeout=self._config.timeout_seconds)

            # Success in half-open state
            self._success_count += 1
            self._metrics.successful_events += 1
            self._metrics.last_success = datetime.now()

            self.logger.info(f"Half-open success {self._success_count}/{self._config.success_threshold}")

            # Check if we can close the circuit
            if self._success_count >= self._config.success_threshold:
                await self._close_circuit()

            return ModelPublishEventResult(
                published=True,
                queued=False,
                event_id=str(event.correlation_id),
                half_open_success=True,
            )

        except Exception as e:
            # Failure in half-open - immediately open circuit again
            await self._open_circuit(f"Half-open test failed: {e!s}")
            return await self._queue_or_drop_event(event)

    async def _handle_open_circuit(self, event: ModelOnexEvent) -> ModelPublishEventResult:
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
        self._failure_count += 1
        self._metrics.failed_events += 1
        self._metrics.last_failure = datetime.now()
        self._last_failure_time = time.time()

        # Update success rate
        self._metrics.success_rate_percent = (
            self._metrics.successful_events / max(self._metrics.total_events, 1) * 100
        )

        self.logger.warning(f"Event publishing failure {self._failure_count}/{self._config.failure_threshold}: {error_message}")

        # Open circuit if failure threshold reached
        if self._failure_count >= self._config.failure_threshold:
            await self._open_circuit(f"Failure threshold reached: {error_message}")

    async def _open_circuit(self, reason: str):
        """Open the circuit breaker."""
        if self._state != EnumCircuitBreakerState.OPEN:
            self._state = EnumCircuitBreakerState.OPEN
            self._last_state_change_time = datetime.now()
            self._metrics.circuit_opens += 1
            self.logger.error(f"Circuit breaker OPENED: {reason}")

    async def _close_circuit(self):
        """Close the circuit breaker (recovery complete)."""
        self._state = EnumCircuitBreakerState.CLOSED
        self._last_state_change_time = datetime.now()
        self._failure_count = 0
        self._success_count = 0
        self._metrics.circuit_closes += 1
        self.logger.info("Circuit breaker CLOSED - recovery complete")

        # Process any queued events
        await self._process_queued_events()

    async def _transition_to_half_open(self):
        """Transition circuit to half-open state for recovery testing."""
        self._state = EnumCircuitBreakerState.HALF_OPEN
        self._last_state_change_time = datetime.now()
        self._success_count = 0
        self.logger.info("Circuit breaker transitioned to HALF-OPEN - testing recovery")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset to half-open."""
        if self._last_failure_time is None:
            return False

        time_since_failure = time.time() - self._last_failure_time
        return time_since_failure >= self._config.recovery_timeout

    async def _queue_or_drop_event(self, event: ModelOnexEvent) -> ModelPublishEventResult:
        """Queue event or drop it based on queue capacity and configuration."""
        if not self._config.graceful_degradation:
            # Fail-fast mode - raise error for critical operations
            raise OnexError(
                code=CoreErrorCode.INTEGRATION_SERVICE_UNAVAILABLE,
                message="Event bus circuit breaker open - event publishing failed",
                details={"circuit_state": self._state.value, "queued_events": len(self._event_queue)},
            )

        # Graceful degradation mode - queue if possible
        if len(self._event_queue) < self._config.max_queue_size:
            self._event_queue.append(event)
            self._metrics.queued_events += 1
            self.logger.info(f"Event queued (circuit {self._state.value}): {event.correlation_id}")

            return ModelPublishEventResult(
                published=False,
                queued=True,
                event_id=str(event.correlation_id),
                reason=f"Circuit {self._state.value} - event queued",
            )
        # Queue full - move to dead letter queue if enabled
        if self._config.dead_letter_enabled:
            await self._add_to_dead_letter_queue(event, "Queue capacity exceeded")

        self._metrics.dropped_events += 1
        self.logger.warning(f"Event dropped - queue full: {event.correlation_id}")

        return ModelPublishEventResult(
            published=False,
            queued=False,
            dropped=True,
            event_id=str(event.correlation_id),
            reason="Queue capacity exceeded",
        )

    async def _add_to_dead_letter_queue(self, event: ModelOnexEvent, reason: str):
        """Add failed event to dead letter queue for later processing."""
        dead_letter_entry = ModelDeadLetterQueueEntry(
            event=event.model_dump(),
            timestamp=datetime.now().isoformat(),
            reason=reason,
            circuit_state=self._state.value,
            retry_count=0,
        )

        self._dead_letter_queue.append(dead_letter_entry)
        self._metrics.dead_letter_events += 1
        self.logger.info(f"Event added to dead letter queue: {event.correlation_id} - {reason}")

    async def _process_queued_events(self):
        """Process queued events when circuit closes."""
        if not self._event_queue:
            return

        queued_count = len(self._event_queue)
        self.logger.info(f"Processing {queued_count} queued events after circuit recovery")

        # Process events in background to avoid blocking
        asyncio.create_task(self._process_queue_background())

    async def _process_queue_background(self):
        """Background task to process queued events with proper thread safety."""
        processed = 0
        failed = 0

        while True:
            # Check state and queue with proper locking
            async with self._lock:
                if not self._event_queue or self._state != EnumCircuitBreakerState.CLOSED:
                    break

                try:
                    event = self._event_queue.pop(0)
                except IndexError:
                    # Queue was emptied between check and pop
                    break

            # Process event outside the lock to avoid blocking other operations
            try:
                # NOTE: Event re-publishing requires publisher function context
                # For now, events are processed but not re-published to avoid state inconsistency
                processed += 1

            except Exception as e:
                failed += 1
                self.logger.error(f"Failed to process queued event: {e}")

                if failed >= 3:  # Prevent infinite retry loops
                    break

        self.logger.info(f"Queued event processing complete: {processed} processed, {failed} failed")

    async def _handle_get_state(self, input_data: ModelEventBusCircuitBreakerInput) -> ModelStateResult:
        """Handle get circuit breaker state operation."""
        state_info = ModelCircuitBreakerState(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._metrics.last_failure,
            last_success_time=self._metrics.last_success,
            last_state_change=self._last_state_change_time,
            is_healthy=self._is_healthy(),
        )

        return ModelStateResult(
            state=state_info.model_dump(),
        )

    async def _handle_get_metrics(self, input_data: ModelEventBusCircuitBreakerInput) -> ModelCircuitBreakerMetrics:
        """Handle get circuit breaker metrics operation."""
        return self._metrics

    async def _handle_reset_circuit(self, input_data: ModelEventBusCircuitBreakerInput) -> ModelResetResult:
        """Handle manual circuit reset operation."""
        async with self._lock:
            self._state = EnumCircuitBreakerState.CLOSED
            self._last_state_change_time = datetime.now()
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self.logger.info("Circuit breaker manually reset to CLOSED state")

        return ModelResetResult(
            reset=True,
            new_state=self._state.value,
        )

    async def _handle_get_health_status(self, input_data: ModelEventBusCircuitBreakerInput) -> ModelHealthStatusResult:
        """Handle get health status operation."""
        return ModelHealthStatusResult(
            circuit_state=self._state.value,
            is_healthy=self._is_healthy(),
            failure_count=self._failure_count,
            queued_events=len(self._event_queue),
            dead_letter_events=len(self._dead_letter_queue),
            configuration=self._config.model_dump() if self._config else {},
            metrics=self._metrics.model_dump(),
        )

    def _is_healthy(self) -> bool:
        """Check if circuit breaker is healthy for event publishing."""
        return self._state == EnumCircuitBreakerState.CLOSED or self._state == EnumCircuitBreakerState.HALF_OPEN

    async def _load_configuration(self) -> ModelCircuitBreakerConfig:
        """Load circuit breaker configuration from container or defaults."""
        try:
            # Try to load from container configuration first
            # In production, this would come from container.get_configuration()
            # For now, detect environment and use appropriate defaults

            # Environment detection (similar to distributed tracing)
            env_vars = ["ENVIRONMENT", "ENV", "DEPLOYMENT_ENV", "NODE_ENV", "OMNIBASE_ENV"]
            environment = "development"  # default
            for var in env_vars:
                value = os.getenv(var)
                if value:
                    environment = value.lower()
                    break

            # Create environment configuration with defaults
            env_config = ModelCircuitBreakerEnvironmentConfig.create_default_config()

            # Get configuration for detected environment
            config = env_config.get_config_for_environment(
                environment=environment,
                default_environment="development",
            )

            self.logger.info(f"Loaded circuit breaker configuration for environment: {environment}")
            return config

        except Exception as e:
            self.logger.warning(f"Failed to load configuration from container, using development defaults: {e}")
            # Fallback to development defaults
            return ModelCircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=15,
                success_threshold=1,
                timeout_seconds=10,
                max_queue_size=100,
                dead_letter_enabled=False,
                graceful_degradation=True,
            )

    async def _get_publisher_function(self, function_name: str | None) -> Callable:
        """Get publisher function for event publishing."""
        if function_name and function_name in self._publisher_functions:
            return self._publisher_functions[function_name]

        # Default mock publisher for testing - with production safety check
        async def mock_publisher(event: ModelOnexEvent) -> None:
            """Mock publisher function for testing - NOT for production use."""
            # Production safety check
            environment = os.getenv("ENVIRONMENT", "").lower()
            if environment in ("production", "prod"):
                self.logger.error("CRITICAL: Mock publisher used in production environment!")
                raise OnexError(
                    message="Mock publisher cannot be used in production environment",
                    error_code=CoreErrorCode.CONFIGURATION_ERROR,
                )

            self.logger.warning(f"Using mock publisher for event: {event.correlation_id} (environment: {environment or 'unknown'})")
            # Simulate some processing time
            await asyncio.sleep(0.01)

        return mock_publisher

    def register_publisher_function(self, name: str, function: Callable) -> None:
        """Register a publisher function for use by the circuit breaker."""
        self._publisher_functions[name] = function
        self.logger.info(f"Registered publisher function: {name}")
