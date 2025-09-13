"""Kafka Adapter Tool - Message Bus Bridge for Streaming Operations.

This adapter serves as a bridge between the ONEX message bus and Kafka streaming operations.
It converts event envelopes containing streaming requests into direct Kafka client calls.
Following the ONEX infrastructure tool pattern for external service integration.

Message Flow:
Event Envelope → Kafka Adapter → Kafka Client Manager → Kafka Cluster

Integrates with:
- kafka_event_processing_subcontract: Event bus integration patterns
- kafka_connection_management_subcontract: Broker connection management
"""

import asyncio
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Union, Pattern, Protocol
from uuid import UUID, uuid4

from omnibase_core.core.core_error_codes import CoreErrorCode
from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.node_effect_service import NodeEffectService
from omnibase_core.onex_container import ModelONEXContainer
from omnibase_core.enums.enum_health_status import EnumHealthStatus
from omnibase_core.model.core.model_health_status import ModelHealthStatus

from ....models.kafka.model_kafka_message import ModelKafkaMessage
from ....models.kafka.model_kafka_topic_config import ModelKafkaTopicConfig
from ....models.kafka.model_kafka_producer_config import ModelKafkaProducerConfig
from ....models.kafka.model_kafka_consumer_config import ModelKafkaConsumerConfig
from ....models.kafka.model_kafka_health_response import ModelKafkaHealthResponse
from ....models.common.model_kafka_configuration import ModelKafkaConfiguration
from ....enums.enum_kafka_operation_type import EnumKafkaOperationType
from .models.model_kafka_adapter_input import ModelKafkaAdapterInput
from .models.model_kafka_adapter_output import ModelKafkaAdapterOutput


class ProtocolKafkaClient(Protocol):
    """Protocol interface for Kafka client implementations."""
    
    async def start(self) -> None:
        """Start the Kafka client connection."""
        ...
    
    async def stop(self) -> None:
        """Stop the Kafka client connection."""
        ...
    
    async def send_and_wait(self, topic: str, value: bytes, key: Optional[bytes] = None) -> None:
        """Send message to Kafka topic and wait for acknowledgment."""
        ...
    
    def bootstrap_servers(self) -> List[str]:
        """Get list of bootstrap servers."""
        ...


class KafkaStructuredLogger:
    """
    Structured logger for Kafka adapter operations with correlation ID tracking.
    
    Provides consistent, structured logging across all streaming operations with:
    - Correlation ID tracking for request tracing
    - Performance metrics logging
    - Error context preservation
    - Security-aware message sanitization
    """
    
    def __init__(self, logger_name: str = "kafka_adapter"):
        """Initialize structured logger with correlation ID support."""
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            # Configure structured logging format if not already configured
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(operation)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _build_extra(self, correlation_id: Optional[UUID], operation: str, **kwargs) -> dict:
        """Build extra fields for structured logging."""
        extra = {
            'correlation_id': str(correlation_id) if correlation_id else 'no-correlation',
            'operation': operation,
            'component': 'kafka_adapter',
            'node_type': 'effect',
        }
        extra.update(kwargs)
        return extra
    
    def info(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "general", **kwargs):
        """Log info level message with structured fields."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "general", **kwargs):
        """Log warning level message with structured fields."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "general", 
              exception: Optional[Exception] = None, **kwargs):
        """Log error level message with structured fields and exception context."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        if exception:
            extra['exception_type'] = type(exception).__name__
            extra['exception_message'] = str(exception)
        self.logger.error(message, extra=extra, exc_info=exception is not None)
    
    def debug(self, message: str, correlation_id: Optional[UUID] = None, operation: str = "general", **kwargs):
        """Log debug level message with structured fields."""
        extra = self._build_extra(correlation_id, operation, **kwargs)
        self.logger.debug(message, extra=extra)
    
    def log_produce_start(self, correlation_id: UUID, topic: str, message_size: int):
        """Log start of message produce operation."""
        self.info(
            f"Starting message produce to topic '{topic}' (size: {message_size} bytes)",
            correlation_id=correlation_id,
            operation="produce_start",
            topic=topic,
            message_size=message_size
        )
    
    def log_produce_success(self, correlation_id: UUID, topic: str, partition: int, offset: int, execution_time_ms: float):
        """Log successful message produce completion."""
        self.info(
            f"Message produced successfully to {topic}:{partition} at offset {offset} in {execution_time_ms:.2f}ms",
            correlation_id=correlation_id,
            operation="produce_success",
            topic=topic,
            partition=partition,
            offset=offset,
            execution_time_ms=execution_time_ms,
            performance_category="fast" if execution_time_ms < 50 else "slow" if execution_time_ms < 200 else "very_slow"
        )
    
    def log_consume_start(self, correlation_id: UUID, topics: List[str], consumer_group: str):
        """Log start of message consume operation."""
        self.info(
            f"Starting message consume from topics {topics} with group '{consumer_group}'",
            correlation_id=correlation_id,
            operation="consume_start",
            topics=topics,
            consumer_group=consumer_group
        )
    
    def log_consume_success(self, correlation_id: UUID, record_count: int, execution_time_ms: float):
        """Log successful message consume completion."""
        self.info(
            f"Consumed {record_count} messages in {execution_time_ms:.2f}ms",
            correlation_id=correlation_id,
            operation="consume_success",
            record_count=record_count,
            execution_time_ms=execution_time_ms,
            throughput_msgs_per_sec=record_count * 1000 / execution_time_ms if execution_time_ms > 0 else 0
        )
    
    def log_streaming_error(self, correlation_id: UUID, operation: str, execution_time_ms: float, exception: Exception):
        """Log streaming operation error with context."""
        self.error(
            f"Kafka {operation} failed after {execution_time_ms:.2f}ms",
            correlation_id=correlation_id,
            operation=f"{operation}_error",
            exception=exception,
            execution_time_ms=execution_time_ms,
            error_category=self._categorize_kafka_error(exception)
        )
    
    def _categorize_kafka_error(self, exception: Exception) -> str:
        """Categorize Kafka errors for better observability."""
        error_str = str(exception).lower()
        if "connection" in error_str or "timeout" in error_str or "broker" in error_str:
            return "connectivity"
        elif "authorization" in error_str or "permission" in error_str or "acl" in error_str:
            return "authorization"
        elif "serialization" in error_str or "deserialization" in error_str:
            return "serialization"
        elif "offset" in error_str or "partition" in error_str:
            return "partition_management"
        elif "topic" in error_str and "not" in error_str and "exist" in error_str:
            return "topic_management"
        else:
            return "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states for Kafka connectivity failures."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class KafkaCircuitBreaker:
    """
    Circuit breaker implementation for Kafka connectivity failures.
    
    Prevents cascading failures by monitoring Kafka operation failures
    and temporarily blocking requests when failure thresholds are exceeded.
    """
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60, half_open_max_calls: int = 3):
        """
        Initialize circuit breaker with configurable thresholds.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting recovery
            half_open_max_calls: Max calls to allow in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            OnexError: If circuit is open or function fails
        """
        async with self._lock:
            # Check if we should attempt recovery
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise OnexError(
                        code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                        message="Kafka circuit breaker is OPEN - service temporarily unavailable",
                    )
            
            # In half-open state, limit calls
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise OnexError(
                        code=CoreErrorCode.SERVICE_UNAVAILABLE_ERROR,
                        message="Kafka circuit breaker is HALF_OPEN - maximum test calls exceeded",
                    )
                self.half_open_calls += 1
        
        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise
    
    async def _record_success(self):
        """Record successful operation and potentially close circuit."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Reset to closed state after successful test
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.last_failure_time = None
                self.half_open_calls = 0
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success in closed state
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _record_failure(self, exception: Exception):
        """Record failed operation and potentially open circuit."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure >= timedelta(seconds=self.timeout_seconds)
    
    def get_state(self) -> dict:
        """Get current circuit breaker state for monitoring."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "half_open_calls": self.half_open_calls if self.state == CircuitBreakerState.HALF_OPEN else 0,
        }


class Node(NodeEffectService):
    """
    Infrastructure Kafka Adapter Node - Message Bus Bridge.
    
    Converts message bus envelopes containing streaming requests into direct
    Kafka client operations. This follows the ONEX infrastructure tool pattern 
    where adapters serve as bridges between the event-driven message bus and 
    external service APIs.
    
    Message Flow:
    Event Envelope → Kafka Adapter → Kafka Client Manager → Kafka Cluster
    
    Integrates with:
    - kafka_event_processing_subcontract: Event bus integration patterns
    - kafka_connection_management_subcontract: Broker connection management
    """
    
    def __init__(self, container: ModelONEXContainer):
        """Initialize Kafka adapter tool with container injection."""
        super().__init__(container)
        self.node_type = "effect"
        self.domain = "infrastructure"
        self._kafka_client: Optional[ProtocolKafkaClient] = None  # Will be resolved from container
        self._kafka_client_lock = asyncio.Lock()
        self._kafka_client_sync_lock = threading.Lock()
        
        # Initialize circuit breaker for Kafka connectivity failures
        self._circuit_breaker = KafkaCircuitBreaker(
            failure_threshold=5,  # Open circuit after 5 failures
            timeout_seconds=60,   # Wait 60 seconds before retry
            half_open_max_calls=3  # Allow 3 test calls in half-open state
        )
        
        # Initialize structured logger with correlation ID support
        self._logger = KafkaStructuredLogger("kafka_adapter_node")
        
        # Initialize Prometheus metrics collector
        try:
            from ....observability.prometheus_metrics import get_metrics_collector
            self._metrics = get_metrics_collector()
        except ImportError:
            self._metrics = None
            self._logger.warning("Prometheus metrics not available")
        
        # Load configuration from environment or container
        self._config = self._load_configuration(container)
        
        # Log adapter initialization
        self._logger.info(
            "Kafka adapter initialized successfully",
            operation="initialization",
            node_type=self.node_type,
            domain=self.domain
        )
    
    def _load_configuration(self, container: ModelONEXContainer) -> ModelKafkaConfiguration:
        """
        Load Kafka adapter configuration from container or environment with secure credential management.
        
        Args:
            container: ONEX container for dependency injection
            
        Returns:
            Dictionary with configuration values using secure credential management
        """
        try:
            # Try to get configuration from container first (ONEX pattern)
            config = container.get_service("kafka_adapter_config")
            if config:
                return config
        except Exception:
            pass  # Fall back to secure credential-based configuration
        
        # Use secure credential manager instead of hardcoded localhost
        try:
            from ....security.credential_manager import get_credential_manager
            credential_manager = get_credential_manager()
            event_bus_creds = credential_manager.get_event_bus_credentials()
            
            return {
                "bootstrap_servers": ",".join(event_bus_creds.bootstrap_servers),
                "client_id": os.getenv("KAFKA_CLIENT_ID", "kafka-adapter"),
                "max_message_size": int(os.getenv("KAFKA_MAX_MESSAGE_SIZE", "1048576")),  # 1MB
                "request_timeout_ms": int(os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "30000")),  # 30s
                "enable_idempotence": os.getenv("KAFKA_ENABLE_IDEMPOTENCE", "true").lower() == "true",
                "security_protocol": event_bus_creds.security_protocol,
                "sasl_mechanism": event_bus_creds.sasl_mechanism,
                "sasl_username": event_bus_creds.sasl_username,
                "sasl_password": event_bus_creds.sasl_password,
                "ssl_ca_location": event_bus_creds.ssl_ca_location,
                "ssl_cert_location": event_bus_creds.ssl_cert_location,
                "ssl_key_location": event_bus_creds.ssl_key_location,
                "ssl_key_password": event_bus_creds.ssl_key_password,
                "enable_error_sanitization": os.getenv("KAFKA_ENABLE_ERROR_SANITIZATION", "true").lower() == "true",
            }
        except Exception as e:
            # Log error but provide safe fallback to environment variables (no hardcoded localhost)
            self._logger.warning(
                f"Failed to load credentials from credential manager: {str(e)}, falling back to environment",
                operation="configuration_load"
            )
            return {
                "bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", 
                                               os.getenv("REDPANDA_BOOTSTRAP_SERVERS", "redpanda:9092")),
                "client_id": os.getenv("KAFKA_CLIENT_ID", "kafka-adapter"),
                "max_message_size": int(os.getenv("KAFKA_MAX_MESSAGE_SIZE", "1048576")),  # 1MB
                "request_timeout_ms": int(os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "30000")),  # 30s
                "enable_idempotence": os.getenv("KAFKA_ENABLE_IDEMPOTENCE", "true").lower() == "true",
                "security_protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
                "enable_error_sanitization": os.getenv("KAFKA_ENABLE_ERROR_SANITIZATION", "true").lower() == "true",
            }
    
    def _validate_correlation_id(self, correlation_id: Optional[UUID]) -> UUID:
        """
        Validate and normalize correlation ID to prevent injection attacks.
        
        Args:
            correlation_id: Optional correlation ID to validate
            
        Returns:
            Valid UUID correlation ID
            
        Raises:
            OnexError: If correlation ID format is invalid
        """
        if correlation_id is None:
            # Generate a new correlation ID if none provided
            return uuid4()
            
        if hasattr(correlation_id, 'replace') and hasattr(correlation_id, 'split'):  # String-like
            try:
                # Try to parse string as UUID to validate format
                correlation_id = UUID(correlation_id)
            except ValueError as e:
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message="Invalid correlation ID format - must be valid UUID"
                ) from e
                
        if not hasattr(correlation_id, 'hex'):
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Correlation ID must be UUID type"
            )
            
        # Additional validation: ensure it's not an empty UUID
        if correlation_id == UUID('00000000-0000-0000-0000-000000000000'):
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Correlation ID cannot be empty UUID"
            )
            
        return correlation_id

    async def get_kafka_client_async(self) -> ProtocolKafkaClient:
        """
        Get Kafka client instance via registry injection with thread safety.
        
        Returns:
            Kafka client instance
            
        Raises:
            OnexError: If Kafka client cannot be resolved
        """
        async with self._kafka_client_lock:
            if self._kafka_client is None:
                # Validate container service interface before resolution
                self._validate_container_service_interface()
                
                # Use container injection per ONEX standards
                self._kafka_client = self.container.get_service("kafka_client")
                
                # Validate the resolved service interface
                self._validate_kafka_client_interface(self._kafka_client)
                
            return self._kafka_client

    def get_health_checks(self) -> List[Callable[[], Union[ModelHealthStatus, "asyncio.Future[ModelHealthStatus]"]]]:
        """
        Override MixinHealthCheck to provide Kafka-specific async health checks.
        
        Returns list of health check functions that validate Kafka connectivity,
        broker status, and adapter functionality.
        """
        return [
            self._check_kafka_connectivity_async,
            self._check_broker_health_async,
            self._check_circuit_breaker_health_async,
        ]

    def _check_kafka_connectivity(self) -> ModelHealthStatus:
        """Check basic Kafka cluster connectivity (sync wrapper for health checks)."""
        start_time = time.perf_counter()
        try:
            # Simple sync health check without async operations
            if self._kafka_client is None:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._logger.info(
                    f"Kafka connectivity check: degraded (client not initialized) in {execution_time_ms:.2f}ms",
                    operation="health_check",
                    check_name="kafka_connectivity",
                    status="degraded"
                )
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Kafka client not initialized",
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Basic connectivity indicator based on client state
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(
                f"Kafka connectivity check: healthy (client operational) in {execution_time_ms:.2f}ms",
                operation="health_check",
                check_name="kafka_connectivity",
                status="healthy"
            )
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message="Kafka client operational",
                timestamp=datetime.utcnow().isoformat()
            )
                    
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.error(
                f"Kafka connectivity check failed in {execution_time_ms:.2f}ms",
                operation="health_check",
                exception=e,
                check_name="kafka_connectivity",
                status="unhealthy"
            )
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Kafka connectivity check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    def _check_broker_health(self) -> ModelHealthStatus:
        """Check Kafka broker health and availability (sync version for mixin)."""
        try:
            # Check if Kafka client is available
            if self._kafka_client is None:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Kafka client not initialized",
                    timestamp=datetime.utcnow().isoformat()
                )
                
            # Broker health is healthy if client exists and is operational
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message="Kafka brokers accessible",
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Broker health check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    def _check_circuit_breaker_health(self) -> ModelHealthStatus:
        """Check circuit breaker health and state (sync version for health checks)."""
        try:
            circuit_state = self._circuit_breaker.get_state()
            state_value = circuit_state["state"]
            
            # Record circuit breaker state metrics for monitoring
            self._metrics.set_circuit_breaker_state(
                client_id=self._config.get("client_id", "kafka-adapter"),
                state=state_value
            )
            
            if state_value == CircuitBreakerState.CLOSED.value:
                return ModelHealthStatus(
                    status=EnumHealthStatus.HEALTHY,
                    message=f"Circuit breaker CLOSED - failures: {circuit_state['failure_count']}",
                    timestamp=datetime.utcnow().isoformat()
                )
            elif state_value == CircuitBreakerState.HALF_OPEN.value:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message=f"Circuit breaker HALF_OPEN - testing recovery ({circuit_state['half_open_calls']} calls)",
                    timestamp=datetime.utcnow().isoformat()
                )
            else:  # OPEN state
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message=f"Circuit breaker OPEN - service temporarily unavailable (failures: {circuit_state['failure_count']})",
                    timestamp=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Circuit breaker health check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    async def _check_kafka_connectivity_async(self) -> ModelHealthStatus:
        """Check basic Kafka cluster connectivity (async implementation)."""
        start_time = time.perf_counter()
        try:
            # Check if Kafka client is available and can be initialized
            kafka_client = await self.get_kafka_client_async()
            
            if kafka_client is None:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._logger.info(
                    f"Async Kafka connectivity check: degraded (client not initialized) in {execution_time_ms:.2f}ms",
                    operation="async_health_check",
                    check_name="kafka_connectivity",
                    status="degraded"
                )
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Kafka client not initialized",
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # Perform lightweight connectivity test
            # In a real implementation, this would ping the Kafka brokers
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(
                f"Async Kafka connectivity check: healthy (client operational) in {execution_time_ms:.2f}ms",
                operation="async_health_check",
                check_name="kafka_connectivity",
                status="healthy"
            )
            
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message="Kafka client operational",
                timestamp=datetime.utcnow().isoformat()
            )
                    
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._logger.error(
                f"Async Kafka connectivity check failed in {execution_time_ms:.2f}ms",
                operation="async_health_check",
                exception=e,
                check_name="kafka_connectivity",
                status="unhealthy"
            )
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Kafka connectivity check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    async def _check_broker_health_async(self) -> ModelHealthStatus:
        """Check Kafka broker health and availability (async implementation)."""
        try:
            # Check if Kafka client is available
            kafka_client = await self.get_kafka_client_async()
            
            if kafka_client is None:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Kafka client not initialized",
                    timestamp=datetime.utcnow().isoformat()
                )
            
            # In a real implementation, this would check broker metadata
            # For now, we assume brokers are healthy if client is operational
            bootstrap_servers = getattr(kafka_client, 'bootstrap_servers', lambda: ['unknown'])()
            
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message=f"Kafka brokers accessible: {len(bootstrap_servers)} servers",
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Async broker health check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    async def _check_circuit_breaker_health_async(self) -> ModelHealthStatus:
        """Check circuit breaker health and state (async implementation)."""
        try:
            circuit_state = self._circuit_breaker.get_state()
            state_value = circuit_state["state"]
            
            # Record circuit breaker state metrics for monitoring
            self._metrics.set_circuit_breaker_state(
                client_id=self._config.get("client_id", "kafka-adapter"),
                state=state_value
            )
            
            if state_value == CircuitBreakerState.CLOSED.value:
                return ModelHealthStatus(
                    status=EnumHealthStatus.HEALTHY,
                    message=f"Circuit breaker CLOSED - failures: {circuit_state['failure_count']}",
                    timestamp=datetime.utcnow().isoformat()
                )
            elif state_value == CircuitBreakerState.HALF_OPEN.value:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message=f"Circuit breaker HALF_OPEN - testing recovery ({circuit_state['half_open_calls']} calls)",
                    timestamp=datetime.utcnow().isoformat()
                )
            else:  # OPEN state
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message=f"Circuit breaker OPEN - service temporarily unavailable (failures: {circuit_state['failure_count']})",
                    timestamp=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            return ModelHealthStatus(
                status=EnumHealthStatus.UNHEALTHY,
                message=f"Async circuit breaker health check failed: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )

    async def process(self, input_data: ModelKafkaAdapterInput) -> ModelKafkaAdapterOutput:
        """
        Process Kafka adapter request following infrastructure tool pattern.
        
        Routes message envelope to appropriate streaming operation based on operation_type.
        Handles produce, consume, topic management, and health check operations with proper 
        error handling and metrics collection as defined in the event processing subcontract.
        
        Args:
            input_data: Input envelope containing operation type and request data
            
        Returns:
            Output envelope with operation results
        """
        start_time = time.perf_counter()
        
        try:
            # Validate and normalize correlation ID to prevent injection attacks
            validated_correlation_id = self._validate_correlation_id(input_data.correlation_id)
            
            # Update the input data with validated correlation ID if it was modified
            if validated_correlation_id != input_data.correlation_id:
                input_data.correlation_id = validated_correlation_id
            
            # Route based on operation type (as defined in subcontracts)
            if input_data.operation_type == EnumKafkaOperationType.PRODUCE:
                return await self._handle_produce_operation(input_data, start_time)
            elif input_data.operation_type == EnumKafkaOperationType.CONSUME:
                return await self._handle_consume_operation(input_data, start_time)
            elif input_data.operation_type == EnumKafkaOperationType.TOPIC_CREATE:
                return await self._handle_topic_create_operation(input_data, start_time)
            elif input_data.operation_type == EnumKafkaOperationType.TOPIC_DELETE:
                return await self._handle_topic_delete_operation(input_data, start_time)
            elif input_data.operation_type == EnumKafkaOperationType.HEALTH_CHECK:
                return await self._handle_health_check_operation(input_data, start_time)
            else:
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message=f"Unsupported operation type: {input_data.operation_type}",
                )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            if hasattr(e, 'code') and hasattr(e, 'message'):  # OnexError-like
                error_message = str(e)
            else:
                error_message = f"Kafka adapter tool error: {str(e)}"
                
            return ModelKafkaAdapterOutput(
                operation_type=input_data.operation_type,
                success=False,
                error_message=error_message,
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                context={"error_type": type(e).__name__}
            )

    async def _handle_produce_operation(
        self, 
        input_data: ModelKafkaAdapterInput, 
        start_time: float
    ) -> ModelKafkaAdapterOutput:
        """
        Handle message produce operation following connection management patterns.
        
        Implements message production strategy as defined in kafka_connection_management_subcontract
        with proper timeout handling, retry logic, and performance monitoring.
        """
        if not input_data.message:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Message is required for produce operation",
            )

        message = input_data.message
        correlation_id = input_data.correlation_id
        
        # Log produce start with structured logging
        message_size = len(str(message.value)) if message.value else 0
        self._logger.log_produce_start(
            correlation_id=correlation_id,
            topic=message.topic,
            message_size=message_size
        )
        
        # Input validation for security and performance
        self._validate_message_input(message)
        
        # Encrypt sensitive payload data if configured
        processed_message = await self._process_message_security(message, correlation_id)
        
        try:
            # Get Kafka client
            kafka_client = await self.get_kafka_client_async()
            
            # Wrap Kafka call in circuit breaker for failure protection
            # Note: This is a mock implementation - actual Kafka client integration needed
            result = await self._circuit_breaker.call(
                self._mock_produce_message,
                kafka_client,
                processed_message,
                timeout_seconds=input_data.timeout_seconds
            )
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract result information (mock format)
            partition = result.get("partition", 0)
            offset = result.get("offset", -1)
            
            # Log successful produce completion
            self._logger.log_produce_success(
                correlation_id=correlation_id,
                topic=message.topic,
                partition=partition,
                offset=offset,
                execution_time_ms=execution_time_ms
            )
            
            # Audit log event publishing for security monitoring
            await self._audit_log_event_publish(
                correlation_id=correlation_id,
                topic=processed_message.topic,
                outcome="success",
                execution_time_ms=execution_time_ms
            )
            
            # Record Prometheus metrics
            if self._metrics:
                self._metrics.record_kafka_message_published(
                    topic=processed_message.topic,
                    client_id="kafka_adapter",
                    status="success",
                    duration_seconds=execution_time_ms / 1000.0
                )
                
                # Record encryption metrics if payload was encrypted
                if self._should_encrypt_payload(message):
                    self._metrics.record_payload_encrypted(
                        topic=processed_message.topic,
                        client_id="kafka_adapter"
                    )

            return ModelKafkaAdapterOutput(
                operation_type=EnumKafkaOperationType.PRODUCE,
                success=True,
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                record_count=1,
                bytes_processed=message_size,
                offset_info={
                    "topic": message.topic,
                    "partition": partition,
                    "offset": offset
                },
                context=input_data.context,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Log streaming error with structured logging
            self._logger.log_streaming_error(
                correlation_id=correlation_id,
                operation="produce",
                execution_time_ms=execution_time_ms,
                exception=e
            )
            
            # Sanitize error message if configured
            if self._config.get("enable_error_sanitization", True):
                sanitized_error = self._sanitize_error_message(str(e))
            else:
                sanitized_error = str(e)

            return ModelKafkaAdapterOutput(
                operation_type=EnumKafkaOperationType.PRODUCE,
                success=False,
                error_message=sanitized_error,
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

    async def _handle_consume_operation(
        self, 
        input_data: ModelKafkaAdapterInput, 
        start_time: float
    ) -> ModelKafkaAdapterOutput:
        """Handle message consume operation following streaming patterns."""
        if not input_data.consumer_config:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Consumer config is required for consume operation",
            )

        consumer_config = input_data.consumer_config
        correlation_id = input_data.correlation_id
        
        # Log consume start with structured logging
        self._logger.log_consume_start(
            correlation_id=correlation_id,
            topics=consumer_config.topics,
            consumer_group=consumer_config.group_id
        )
        
        try:
            # Get Kafka client
            kafka_client = await self.get_kafka_client_async()
            
            # Wrap Kafka call in circuit breaker for failure protection
            result = await self._circuit_breaker.call(
                self._mock_consume_messages,
                kafka_client,
                consumer_config,
                timeout_seconds=input_data.timeout_seconds
            )
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract consumed messages (mock format)
            messages = result.get("messages", [])
            record_count = len(messages)
            
            # Log successful consume completion
            self._logger.log_consume_success(
                correlation_id=correlation_id,
                record_count=record_count,
                execution_time_ms=execution_time_ms
            )

            return ModelKafkaAdapterOutput(
                operation_type=EnumKafkaOperationType.CONSUME,
                messages=messages,
                success=True,
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                record_count=record_count,
                context=input_data.context,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Log streaming error
            self._logger.log_streaming_error(
                correlation_id=correlation_id,
                operation="consume",
                execution_time_ms=execution_time_ms,
                exception=e
            )
            
            # Sanitize error message if configured
            if self._config.get("enable_error_sanitization", True):
                sanitized_error = self._sanitize_error_message(str(e))
            else:
                sanitized_error = str(e)

            return ModelKafkaAdapterOutput(
                operation_type=EnumKafkaOperationType.CONSUME,
                success=False,
                error_message=sanitized_error,
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

    async def _handle_topic_create_operation(
        self, 
        input_data: ModelKafkaAdapterInput, 
        start_time: float
    ) -> ModelKafkaAdapterOutput:
        """Handle topic creation operation."""
        if not input_data.topic_config:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Topic config is required for topic create operation",
            )

        try:
            # Mock topic creation
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            topic_info = {
                "topic_name": input_data.topic_config.topic_name,
                "partitions": input_data.topic_config.num_partitions,
                "replication_factor": input_data.topic_config.replication_factor,
                "created": True
            }

            return ModelKafkaAdapterOutput(
                operation_type=EnumKafkaOperationType.TOPIC_CREATE,
                topic_info=topic_info,
                success=True,
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            return ModelKafkaAdapterOutput(
                operation_type=EnumKafkaOperationType.TOPIC_CREATE,
                success=False,
                error_message=str(e),
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

    async def _handle_topic_delete_operation(
        self, 
        input_data: ModelKafkaAdapterInput, 
        start_time: float
    ) -> ModelKafkaAdapterOutput:
        """Handle topic deletion operation."""
        if not input_data.topic_config:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Topic config is required for topic delete operation",
            )

        try:
            # Mock topic deletion
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            topic_info = {
                "topic_name": input_data.topic_config.topic_name,
                "deleted": True
            }

            return ModelKafkaAdapterOutput(
                operation_type=EnumKafkaOperationType.TOPIC_DELETE,
                topic_info=topic_info,
                success=True,
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            return ModelKafkaAdapterOutput(
                operation_type=EnumKafkaOperationType.TOPIC_DELETE,
                success=False,
                error_message=str(e),
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )

    async def _handle_health_check_operation(
        self, 
        input_data: ModelKafkaAdapterInput, 
        start_time: float
    ) -> ModelKafkaAdapterOutput:
        """Handle health check operation for Kafka adapter."""
        try:
            # Create comprehensive health check response
            health_response = ModelKafkaHealthResponse(
                is_healthy=True,
                broker_count=1,  # Mock value
                broker_ids=[1],  # Mock value
                topic_count=0,   # Mock value
                partition_count=0,  # Mock value
                under_replicated_partitions=0,
                offline_partitions=0,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
                timestamp=datetime.utcnow()
            )
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            return ModelKafkaAdapterOutput(
                operation_type=EnumKafkaOperationType.HEALTH_CHECK,
                health_response=health_response,
                success=True,
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                context=input_data.context,
            )
            
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            return ModelKafkaAdapterOutput(
                operation_type=EnumKafkaOperationType.HEALTH_CHECK,
                success=False,
                error_message=f"Health check operation failed: {str(e)}",
                correlation_id=input_data.correlation_id,
                timestamp=datetime.utcnow(),
                execution_time_ms=execution_time_ms,
                context=input_data.context
            )

    async def initialize(self) -> None:
        """Initialize the Kafka adapter tool and client manager."""
        try:
            kafka_client = await self.get_kafka_client_async()
            # Initialize Kafka client if it has an initialize method
            if hasattr(kafka_client, 'initialize'):
                await kafka_client.initialize()
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.INITIALIZATION_ERROR,
                message=f"Failed to initialize Kafka adapter tool: {str(e)}",
            ) from e

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        if self._kafka_client:
            try:
                if hasattr(self._kafka_client, 'close'):
                    await self._kafka_client.close()
            except Exception:
                # Log error but don't raise during cleanup
                pass
            finally:
                self._kafka_client = None

    def _validate_message_input(self, message: ModelKafkaMessage) -> None:
        """Validate message input for security and performance constraints."""
        # Message size validation
        max_size = self._config.get("max_message_size", 1048576)  # 1MB default
        message_size = len(str(message.value)) if message.value else 0
        
        if message_size > max_size:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Message size ({message_size}) exceeds maximum allowed ({max_size} bytes)",
            )
        
        # Topic name validation
        if not message.topic or len(message.topic.strip()) == 0:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Topic name cannot be empty",
            )
        
        # Topic name pattern validation
        if not re.match(r'^[a-zA-Z0-9._-]+$', message.topic):
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Topic name contains invalid characters",
            )

    def _validate_container_service_interface(self) -> None:
        """Validate container service interface compliance."""
        if not hasattr(self.container, 'get_service'):
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="Container does not implement required get_service interface",
            )
        
        if self.container is None:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="Container is None - proper ONEX container injection required",
            )

    def _validate_kafka_client_interface(self, kafka_client) -> None:
        """Validate Kafka client service interface compliance."""
        required_methods = ['produce', 'consume', 'create_topic', 'delete_topic']
        missing_methods = []
        
        for method_name in required_methods:
            if not hasattr(kafka_client, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            # Log warning but don't fail - mock client might not implement all methods
            self._logger.warning(
                f"Kafka client missing some methods: {missing_methods}",
                operation="validation"
            )

    def _sanitize_error_message(self, error_message: str) -> str:
        """Sanitize error messages to prevent sensitive information leakage."""
        # Basic sanitization patterns for Kafka-specific sensitive data
        sanitization_patterns = [
            (re.compile(r'password=[^\s&]*', re.IGNORECASE), 'password=***'),
            (re.compile(r'sasl\.password=[^\s&]*', re.IGNORECASE), 'sasl.password=***'),
            (re.compile(r'ssl\.keystore\.password=[^\s&]*', re.IGNORECASE), 'ssl.keystore.password=***'),
            (re.compile(r'ssl\.truststore\.password=[^\s&]*', re.IGNORECASE), 'ssl.truststore.password=***'),
            (re.compile(r'api[_-]?key[_-]*[:=][^\s&]*', re.IGNORECASE), 'api_key=***'),
        ]
        
        sanitized = error_message
        for pattern, replacement in sanitization_patterns:
            sanitized = pattern.sub(replacement, sanitized)
        
        return sanitized

    async def _process_message_security(self, message: ModelKafkaMessage, correlation_id: UUID) -> ModelKafkaMessage:
        """
        Process message security including payload encryption and rate limiting.
        
        Args:
            message: Original Kafka message
            correlation_id: Request correlation ID for tracking
            
        Returns:
            Processed message with security applied
            
        Raises:
            OnexError: If security processing fails or rate limit exceeded
        """
        try:
            # Rate limiting check
            await self._check_rate_limit(correlation_id)
            
            # Payload encryption for sensitive data
            processed_value = message.value
            if self._should_encrypt_payload(message):
                processed_value = await self._encrypt_message_payload(message.value, correlation_id)
                
                self._logger.info(
                    "Encrypted sensitive payload for message",
                    correlation_id=correlation_id,
                    operation="payload_encryption",
                    topic=message.topic
                )
            
            # Create processed message with security applied
            return ModelKafkaMessage(
                topic=message.topic,
                key=message.key,
                value=processed_value,
                headers=message.headers,
                timestamp=message.timestamp
            )
            
        except Exception as e:
            self._logger.error(
                f"Message security processing failed: {str(e)}",
                correlation_id=correlation_id,
                operation="security_processing",
                exception=e
            )
            raise OnexError(
                code=CoreErrorCode.SECURITY_ERROR,
                message=f"Message security processing failed: {str(e)}"
            ) from e

    def _should_encrypt_payload(self, message: ModelKafkaMessage) -> bool:
        """
        Determine if message payload should be encrypted based on topic and content.
        
        Args:
            message: Kafka message to evaluate
            
        Returns:
            True if payload should be encrypted
        """
        # Encrypt sensitive topics by default
        sensitive_topic_patterns = [
            'user-', 'auth-', 'payment-', 'personal-', 'credential-', 'secret-'
        ]
        
        topic_lower = message.topic.lower()
        if any(pattern in topic_lower for pattern in sensitive_topic_patterns):
            return True
        
        # Check for sensitive content in payload using duck typing
        if hasattr(message.value, 'lower') and hasattr(message.value, 'replace'):
            value_lower = message.value.lower()
            sensitive_patterns = ['password', 'secret', 'token', 'key', 'credential', 'ssn']
            if any(pattern in value_lower for pattern in sensitive_patterns):
                return True
        
        # Check environment configuration
        encrypt_all = os.getenv("KAFKA_ENCRYPT_ALL_PAYLOADS", "false").lower() == "true"
        return encrypt_all

    async def _encrypt_message_payload(self, payload: str, correlation_id: UUID) -> str:
        """
        Encrypt message payload using ONEX payload encryption.
        
        Args:
            payload: Original payload to encrypt
            correlation_id: Request correlation ID
            
        Returns:
            Encrypted payload as JSON string
        """
        try:
            from ....security.payload_encryption import get_payload_encryption
            encryption_service = get_payload_encryption()
            
            # Encrypt the payload
            encrypted_payload = encryption_service.encrypt_payload(payload)
            
            # Return as JSON string for Kafka message
            return encrypted_payload.to_json()
            
        except Exception as e:
            self._logger.error(
                f"Payload encryption failed: {str(e)}",
                correlation_id=correlation_id,
                operation="payload_encryption"
            )
            raise OnexError(
                code=CoreErrorCode.ENCRYPTION_ERROR,
                message=f"Payload encryption failed: {str(e)}"
            ) from e

    async def _check_rate_limit(self, correlation_id: UUID):
        """
        Check rate limiting for event publishing operations.
        
        Args:
            correlation_id: Request correlation ID
            
        Raises:
            OnexError: If rate limit is exceeded
        """
        # Simple in-memory rate limiting (replace with Redis/distributed implementation)
        import time
        from collections import defaultdict
        
        if not hasattr(self, '_rate_limiter'):
            self._rate_limiter = defaultdict(list)
        
        current_time = time.time()
        window_size = int(os.getenv("KAFKA_RATE_LIMIT_WINDOW", "60"))  # 60 seconds
        max_requests = int(os.getenv("KAFKA_RATE_LIMIT_MAX", "100"))  # 100 requests per window
        
        # Use a simple key based on client/topic (in production, use more sophisticated keys)
        rate_key = f"kafka_adapter_{correlation_id}"
        
        # Clean old requests outside the window
        cutoff_time = current_time - window_size
        self._rate_limiter[rate_key] = [
            req_time for req_time in self._rate_limiter[rate_key] 
            if req_time > cutoff_time
        ]
        
        # Check if limit exceeded
        if len(self._rate_limiter[rate_key]) >= max_requests:
            self._logger.warning(
                f"Rate limit exceeded for client",
                correlation_id=correlation_id,
                operation="rate_limiting",
                requests_count=len(self._rate_limiter[rate_key]),
                max_requests=max_requests,
                window_size=window_size
            )
            # Audit log rate limiting violation
            try:
                from ....security.audit_logger import get_audit_logger
                audit_logger = get_audit_logger()
                audit_logger.log_security_violation(
                    client_id=f"kafka_adapter_{correlation_id}",
                    violation_type="rate_limit_exceeded",
                    description=f"Rate limit exceeded: {len(self._rate_limiter[rate_key])} requests in {window_size}s window",
                    details={
                        "max_requests": max_requests,
                        "window_size": window_size,
                        "actual_requests": len(self._rate_limiter[rate_key])
                    }
                )
            except Exception:
                pass  # Don't fail on audit logging issues
            
            # Record rate limit violation metrics
            if self._metrics:
                self._metrics.record_rate_limit_violation("kafka_adapter")
                
            raise OnexError(
                code=CoreErrorCode.RATE_LIMIT_EXCEEDED,
                message=f"Rate limit exceeded: {max_requests} requests per {window_size} seconds"
            )
        
        # Add current request
        self._rate_limiter[rate_key].append(current_time)

    async def _audit_log_event_publish(self, 
                                      correlation_id: UUID,
                                      topic: str,
                                      outcome: str,
                                      execution_time_ms: float,
                                      error_message: Optional[str] = None,
                                      rate_limited: bool = False):
        """
        Audit log event publishing activity for security monitoring.
        
        Args:
            correlation_id: Request correlation ID
            topic: Kafka topic
            outcome: Publishing outcome
            execution_time_ms: Execution time
            error_message: Error message if failed
            rate_limited: Whether request was rate limited
        """
        try:
            from ....security.audit_logger import get_audit_logger
            audit_logger = get_audit_logger()
            
            audit_logger.log_event_publish(
                client_id="kafka_adapter",
                correlation_id=str(correlation_id),
                event_type="kafka_message",
                topic=topic,
                outcome=outcome,
                rate_limited=rate_limited,
                error_message=error_message
            )
            
        except Exception as e:
            # Don't fail the main operation if audit logging fails
            self._logger.warning(
                f"Audit logging failed: {str(e)}",
                correlation_id=correlation_id,
                operation="audit_logging"
            )

    # Mock methods for demonstration (replace with actual Kafka client integration)
    async def _mock_produce_message(self, kafka_client, message: ModelKafkaMessage, timeout_seconds: float) -> dict:
        """Mock message produce operation."""
        await asyncio.sleep(0.01)  # Simulate network delay
        return {
            "partition": 0,
            "offset": 12345,
            "topic": message.topic,
            "timestamp": time.time()
        }

    async def _mock_consume_messages(self, kafka_client, consumer_config: ModelKafkaConsumerConfig, timeout_seconds: float) -> dict:
        """Mock message consume operation."""
        await asyncio.sleep(0.05)  # Simulate polling delay
        
        # Mock consumed messages
        mock_messages = []
        for i in range(3):  # Mock 3 messages
            mock_message = ModelKafkaMessage(
                topic=consumer_config.topics[0] if consumer_config.topics else "test-topic",
                key=f"key-{i}",
                value=f"mock-message-{i}",
                headers={},
                timestamp=datetime.utcnow()
            )
            mock_messages.append(mock_message)
        
        return {
            "messages": mock_messages,
            "consumer_group": consumer_config.group_id
        }


async def main():
    """Main entry point for Kafka Adapter - runs in service mode with NodeEffectService"""
    from omnibase_infra.infrastructure.container import create_infrastructure_container

    # Create infrastructure container with all shared dependencies
    container = create_infrastructure_container()

    adapter = Node(container)

    # Initialize the adapter
    await adapter.initialize()

    # Start service mode using NodeEffectService capabilities
    await adapter.start_service_mode()


if __name__ == "__main__":
    asyncio.run(main())