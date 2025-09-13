"""
Infrastructure Service Group Container

Provides proper dependency injection for all infrastructure services.
Implements duck typing for protocol resolution per ONEX standards.

Per user requirements:
- "it should be get_service("ProtocolEventBus") and we should have a onexcontainer
  with all dependencies at the root of the service group"
- "Everything needs to be resolved by duck typing"
- "the work for getting The instance of the event bus should be in there not in
  each base class. that's dumb"
"""

import asyncio
import json
import logging
import os
import time
from typing import Callable, Optional, Type, TypeVar, Union, Dict, Any, List

from omnibase_core.core.onex_container import ModelONEXContainer as ONEXContainer
from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus
from omnibase_core.model.core.model_onex_event import ModelOnexEvent
from omnibase_core.utils.generation.utility_schema_loader import UtilitySchemaLoader

# ONEX Security modules
from omnibase_infra.security.credential_manager import get_credential_manager
from omnibase_infra.security.tls_config import get_tls_manager
from omnibase_infra.security.rate_limiter import get_rate_limiter
from omnibase_infra.infrastructure.event_bus_circuit_breaker import (
    EventBusCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState
)
from omnibase_infra.infrastructure.infrastructure_observability import (
    InfrastructureObservability,
    MetricType
)

T = TypeVar("T")


class KafkaProducerPool:
    """
    Connection pool for Kafka producers with proper lifecycle management.
    
    Replaces singleton pattern with dependency injection to prevent memory leaks
    and improve testability. Implements proper cleanup and resource management.
    """
    
    def __init__(self, max_producers: int = 10, cleanup_interval: int = 300):
        self._max_producers = max_producers
        self._cleanup_interval = cleanup_interval
        self._producers: Dict[str, Any] = {}
        self._failed_producers: Dict[str, float] = {}
        self._producer_usage: Dict[str, int] = {}  # Track usage count
        self._last_cleanup = time.time()
        self._logger = logging.getLogger(__name__)
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._background_cleanup_loop())
    
    async def _background_cleanup_loop(self):
        """Background loop for periodic cleanup."""
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_idle_producers()
        except asyncio.CancelledError:
            self._logger.info("Background cleanup task cancelled")
        except Exception as e:
            self._logger.error(f"Background cleanup error: {e}")
    
    async def _cleanup_idle_producers(self):
        """Clean up idle and unhealthy producers."""
        current_time = time.time()
        producers_to_remove = []
        
        for servers_key, producer in self._producers.items():
            # Check if producer is still healthy
            if not await self._is_producer_healthy(producer):
                producers_to_remove.append(servers_key)
                continue
            
            # Remove low-usage producers if pool is at capacity
            if (len(self._producers) > self._max_producers // 2 and
                self._producer_usage.get(servers_key, 0) < 10):  # Low usage threshold
                producers_to_remove.append(servers_key)
        
        # Clean up selected producers
        for servers_key in producers_to_remove:
            await self._remove_producer(servers_key)
        
        # Clean up old failure records
        failed_to_remove = []
        for servers_key, fail_time in self._failed_producers.items():
            if current_time - fail_time > 3600:  # 1 hour cleanup
                failed_to_remove.append(servers_key)
        
        for servers_key in failed_to_remove:
            del self._failed_producers[servers_key]
        
        if producers_to_remove or failed_to_remove:
            self._logger.debug(f"Cleaned up {len(producers_to_remove)} producers, "
                              f"{len(failed_to_remove)} failure records")
    
    async def get_producer(self, bootstrap_servers: list, security_config=None, **config):
        """
        Get or create a producer for the given server configuration.
        
        Args:
            bootstrap_servers: List of Kafka bootstrap servers
            security_config: Security configuration for TLS/SASL
            **config: Additional producer configuration
            
        Returns:
            AIOKafkaProducer instance or None if unavailable
        """
        # Create a key from the server configuration
        servers_key = ','.join(sorted(bootstrap_servers))
        
        # Check if producer exists and is healthy
        if servers_key in self._producers:
            producer = self._producers[servers_key]
            # Validate producer is still connected
            if await self._is_producer_healthy(producer):
                # Track usage
                self._producer_usage[servers_key] = self._producer_usage.get(servers_key, 0) + 1
                return producer
            else:
                # Remove unhealthy producer
                self._logger.info(f"Removing unhealthy Kafka producer for servers: {bootstrap_servers}")
                await self._remove_producer(servers_key)
        
        # Skip creation if this producer has failed recently
        if servers_key in self._failed_producers:
            fail_time = self._failed_producers[servers_key]
            if (time.time() - fail_time) < 60:  # 60 second backoff
                self._logger.debug(f"Skipping producer creation for {servers_key} - recent failure")
                return None
        
        # Check pool capacity
        if len(self._producers) >= self._max_producers:
            # Remove least used producer to make room
            least_used_key = min(self._producer_usage.items(), key=lambda x: x[1])[0]
            await self._remove_producer(least_used_key)
            self._logger.info(f"Removed least used producer {least_used_key} to make room")
        
        # Create new producer
        try:
            from aiokafka import AIOKafkaProducer
            
            # Default configuration optimized for event publishing
            producer_config = {
                'bootstrap_servers': bootstrap_servers,
                'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
                'acks': 1,  # Wait for leader acknowledgment
                'retries': 3,  # Retry failed sends
                'max_in_flight_requests_per_connection': 5,
                'batch_size': 16384,  # 16KB batches
                'linger_ms': 5,  # Wait 5ms for batching
                'connections_max_idle_ms': 300000,  # 5 minutes idle timeout
                **config
            }
            
            # Add security configuration if provided
            if security_config:
                if security_config.security_protocol != "PLAINTEXT":
                    producer_config['security_protocol'] = security_config.security_protocol
                    
                    # SASL configuration
                    if security_config.sasl_mechanism:
                        producer_config['sasl_mechanism'] = security_config.sasl_mechanism
                        producer_config['sasl_plain_username'] = security_config.sasl_username
                        producer_config['sasl_plain_password'] = security_config.sasl_password
                    
                    # SSL configuration  
                    if security_config.security_protocol in ['SSL', 'SASL_SSL']:
                        if security_config.ssl_ca_location:
                            producer_config['ssl_cafile'] = security_config.ssl_ca_location
                        if security_config.ssl_cert_location:
                            producer_config['ssl_certfile'] = security_config.ssl_cert_location
                        if security_config.ssl_key_location:
                            producer_config['ssl_keyfile'] = security_config.ssl_key_location
                        if security_config.ssl_key_password:
                            producer_config['ssl_password'] = security_config.ssl_key_password
            
            producer = AIOKafkaProducer(**producer_config)
            await producer.start()
            self._producers[servers_key] = producer
            
            # Initialize usage tracking
            self._producer_usage[servers_key] = 1
            
            # Clear failure tracking on success
            if servers_key in self._failed_producers:
                del self._failed_producers[servers_key]
            
            self._logger.info(f"Created new Kafka producer in pool for servers: {bootstrap_servers}")
            return producer
            
        except ImportError:
            self._logger.warning("aiokafka not available, producer pool disabled")
            return None
        except Exception as e:
            self._logger.error(f"Failed to create Kafka producer: {e}")
            # Track failure for backoff
            self._failed_producers[servers_key] = time.time()
            return None
    
    async def _is_producer_healthy(self, producer) -> bool:
        """Check if a producer is still healthy and connected."""
        try:
            # Check if producer client is available and started
            if producer and hasattr(producer, '_sender') and producer._sender:
                return True
            return False
        except Exception:
            return False
    
    async def _remove_producer(self, servers_key: str):
        """Safely remove a producer from the pool."""
        if servers_key in self._producers:
            producer = self._producers[servers_key]
            try:
                await producer.stop()
                self._logger.debug(f"Stopped producer for servers: {servers_key}")
            except Exception as e:
                self._logger.error(f"Error stopping producer: {e}")
            finally:
                del self._producers[servers_key]
                # Clean up usage tracking
                if servers_key in self._producer_usage:
                    del self._producer_usage[servers_key]
    
    async def close_all(self):
        """Close all producers in the pool and cleanup resources."""
        # Cancel background cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all producers
        for servers_key, producer in list(self._producers.items()):
            try:
                await producer.stop()
                self._logger.info(f"Closed Kafka producer for servers: {servers_key}")
            except Exception as e:
                self._logger.error(f"Error closing Kafka producer: {e}")
        
        # Clear all tracking data
        self._producers.clear()
        self._producer_usage.clear()
        self._failed_producers.clear()
        
        self._logger.info("Kafka producer pool closed")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get producer pool statistics."""
        return {
            "active_producers": len(self._producers),
            "max_producers": self._max_producers,
            "failed_producers": len(self._failed_producers),
            "total_usage": sum(self._producer_usage.values()),
            "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done()
        }


class RedPandaEventBus(ProtocolEventBus):
    """
    Proper ProtocolEventBus implementation for RedPanda/Kafka integration.
    
    Conforms to the standard ProtocolEventBus interface using OnexEvent objects
    and publishes them to appropriate RedPanda topics using OmniNode topic routing.
    """
    
    def __init__(self, credentials=None, **kwargs):
        """Initialize RedPanda event bus with proper protocol compliance and security."""
        # Get secure credentials from credential manager
        credential_manager = get_credential_manager()
        event_bus_credentials = credential_manager.get_event_bus_credentials()
        
        self._bootstrap_servers = event_bus_credentials.bootstrap_servers
        self._security_config = event_bus_credentials
        
        # Get TLS configuration
        tls_manager = get_tls_manager()
        self._tls_config = tls_manager.get_kafka_tls_config()
        
        # Get rate limiter for event publishing
        self._rate_limiter = get_rate_limiter()
        
        # Use producer pool for efficient connection management
        self._producer_pool = KafkaProducerPool(max_producers=5, cleanup_interval=300)
        self._producer = None
        self._logger = logging.getLogger(__name__)
        
        # Initialize circuit breaker for reliability
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=int(os.getenv('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5')),
            recovery_timeout=int(os.getenv('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', '60')),
            success_threshold=int(os.getenv('CIRCUIT_BREAKER_SUCCESS_THRESHOLD', '3')),
            timeout_seconds=int(os.getenv('CIRCUIT_BREAKER_TIMEOUT', '30')),
            max_queue_size=int(os.getenv('CIRCUIT_BREAKER_MAX_QUEUE', '1000')),
            dead_letter_enabled=os.getenv('CIRCUIT_BREAKER_DEAD_LETTER', 'true').lower() == 'true',
            graceful_degradation=os.getenv('CIRCUIT_BREAKER_GRACEFUL_DEGRADATION', 'true').lower() == 'true'
        )
        self._circuit_breaker = EventBusCircuitBreaker(circuit_breaker_config)
        
        # Initialize observability system
        self._observability = InfrastructureObservability(retention_hours=24)
        self._observability.register_circuit_breaker("redpanda_event_bus", self._circuit_breaker)
        
        self._logger.info(f"RedPanda event bus initialized with servers: {self._bootstrap_servers}")
        self._logger.info(f"Security protocol: {self._tls_config.security_protocol}")
        self._logger.info(f"Circuit breaker enabled with failure threshold: {circuit_breaker_config.failure_threshold}")
        self._logger.info(f"Observability system initialized with 24h retention")
        
        # Protocol-compliant subscriber management
        self._subscribers = []
    
    def publish(self, event: ModelOnexEvent) -> None:
        """
        Publish an event to the bus (synchronous) through circuit breaker protection.
        
        Args:
            event: OnexEvent to emit
        """
        # Run async publish in sync context through circuit breaker
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule as task if loop is running
                loop.create_task(self._circuit_breaker_publish(event))
            else:
                # Run directly if no loop is running
                loop.run_until_complete(self._circuit_breaker_publish(event))
        except Exception as e:
            self._logger.error(f"RedPanda sync publish failed: {str(e)}")
    
    async def _circuit_breaker_publish(self, event: ModelOnexEvent) -> bool:
        """
        Publish event through circuit breaker protection with observability.
        
        Args:
            event: OnexEvent to emit
            
        Returns:
            bool: True if published successfully, False if queued or dropped
        """
        start_time = time.time()
        
        try:
            result = await self._circuit_breaker.publish_event(event, self._raw_publish_async)
            
            # Record successful latency
            latency = time.time() - start_time
            self._observability.record_event_latency(latency)
            
            # Record success metric
            self._observability.record_metric(
                "event_publishing_success_total",
                1,
                labels={"event_type": str(event.payload.event_type)},
                metric_type=MetricType.COUNTER
            )
            
            return result
            
        except Exception as e:
            # Record error latency
            latency = time.time() - start_time
            self._observability.record_event_latency(latency)
            
            # Record error metric
            self._observability.record_metric(
                "event_publishing_error_total",
                1,
                labels={"event_type": str(event.payload.event_type), "error": type(e).__name__},
                metric_type=MetricType.COUNTER
            )
            
            # Calculate and record error rate
            error_rate = await self._calculate_recent_error_rate()
            self._observability.record_error_rate("redpanda_event_bus", error_rate)
            
            raise
    
    async def publish_async(self, event: ModelOnexEvent) -> None:
        """
        Publish an event to the bus (asynchronous) through circuit breaker protection.
        
        Args:
            event: OnexEvent to emit
        """
        await self._circuit_breaker_publish(event)
    
    async def _raw_publish_async(self, event: ModelOnexEvent) -> None:
        """
        Raw event publishing without circuit breaker protection (used by circuit breaker).
        
        Args:
            event: OnexEvent to emit
        """
        # Extract client ID for rate limiting (use correlation ID or default)
        client_id = str(event.correlation_id) if event.correlation_id else "default_client"
        
        # Apply rate limiting
        rate_limit_allowed = await self._rate_limiter.check_rate_limit(
            client_id=client_id,
            operation_type="event_publish"
        )
        
        if not rate_limit_allowed:
            self._logger.warning(f"Rate limit exceeded for client {client_id}, event publish denied")
            return
        
        max_retries = int(os.getenv('REDPANDA_MAX_RETRIES', '3'))
        base_delay = float(os.getenv('REDPANDA_BASE_DELAY_SECONDS', '0.1'))
        
        for attempt in range(max_retries + 1):
            try:
                # Get producer from pool (creates if needed) with security config
                producer = await self._producer_pool.get_producer(
                    self._bootstrap_servers,
                    security_config=self._security_config
                )
                
                if not producer:
                    # Mock publishing for testing without aiokafka
                    self._logger.info(f"MOCK: Publishing OnexEvent {event.event_type} with correlation_id={event.correlation_id}")
                    return
                
                # Convert OnexEvent to RedPanda topic and message
                topic = self._event_to_topic(event)
                message_data = event.model_dump()
                partition_key = str(event.correlation_id) if event.correlation_id else None
                
                # Publish to RedPanda topic using pooled producer
                await producer.send_and_wait(
                    topic=topic,
                    value=message_data,
                    key=partition_key.encode('utf-8') if partition_key else None
                )
                
                self._logger.info(f"Published OnexEvent to RedPanda topic: {topic} (correlation_id={event.correlation_id})")
                return  # Success - exit retry loop
                
            except Exception as e:
                if attempt == max_retries:
                    # Final attempt failed - log error but don't raise
                    self._logger.error(f"RedPanda async publish failed after {max_retries + 1} attempts: {str(e)}")
                    return
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                self._logger.warning(f"RedPanda publish attempt {attempt + 1} failed, retrying in {delay:.2f}s: {str(e)}")
                await asyncio.sleep(delay)
    
    def subscribe(self, callback: Callable[[ModelOnexEvent], None], event_type=None) -> None:
        """
        Subscribe a callback to receive events (synchronous).
        
        Args:
            callback: Callable invoked with each OnexEvent
            event_type: Optional event type filter (for compatibility with omnibase_core mixin)
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)
            self._logger.debug(f"Subscribed callback to RedPanda event bus (event_type: {event_type})")
    
    async def subscribe_async(self, callback: Callable[[ModelOnexEvent], None], event_type=None) -> None:
        """
        Subscribe a callback to receive events (asynchronous).
        
        Args:
            callback: Callable invoked with each OnexEvent
            event_type: Optional event type filter (for compatibility with omnibase_core mixin)
        """
        self.subscribe(callback, event_type)
    
    def unsubscribe(self, callback: Callable[[ModelOnexEvent], None]) -> None:
        """
        Unsubscribe a previously registered callback (synchronous).
        
        Args:
            callback: Callable to remove
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            self._logger.debug(f"Unsubscribed callback from RedPanda event bus")
    
    async def unsubscribe_async(self, callback: Callable[[ModelOnexEvent], None]) -> None:
        """
        Unsubscribe a previously registered callback (asynchronous).
        
        Args:
            callback: Callable to remove
        """
        self.unsubscribe(callback)
    
    def clear(self) -> None:
        """Remove all subscribers from the event bus."""
        self._subscribers.clear()
        self._logger.info("Cleared all subscribers from RedPanda event bus")
    
    async def close(self):
        """Close RedPanda producer connection, circuit breaker, and observability cleanup."""
        if self._producer:
            await self._producer.stop()
            self._producer = None
        
        # Clean up producer pool
        await self._producer_pool.close_all()
        
        # Clean up observability system
        await self._observability.close()
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring and health checks."""
        return self._circuit_breaker.get_health_status()
    
    def is_healthy(self) -> bool:
        """Check if the event bus and circuit breaker are healthy."""
        return self._circuit_breaker.is_healthy()
    
    def get_observability_metrics(self) -> Dict[str, Any]:
        """Get comprehensive observability metrics."""
        return self._observability.get_current_metrics()
    
    def get_infrastructure_health_summary(self) -> Dict[str, Any]:
        """Get infrastructure health summary."""
        return self._observability.get_health_summary()
    
    def get_performance_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance trends over specified hours."""
        return self._observability.get_performance_trends(hours)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active infrastructure alerts."""
        return self._observability.get_alerts(active_only=True)
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return self._observability.export_prometheus_metrics()
    
    async def _calculate_recent_error_rate(self) -> float:
        """Calculate recent error rate for observability."""
        # Get recent metrics from circuit breaker
        metrics = self._circuit_breaker.get_metrics()
        
        if metrics.total_events == 0:
            return 0.0
        
        return metrics.failed_events / metrics.total_events
    
    def _event_to_topic(self, event: ModelOnexEvent) -> str:
        """
        Convert OnexEvent to appropriate RedPanda topic using OmniNode namespace.
        
        Args:
            event: OnexEvent to route
            
        Returns:
            RedPanda topic name following OmniNode format
        """
        # Extract environment configuration
        env = os.getenv('OMNINODE_ENV', 'dev')
        tenant = os.getenv('OMNINODE_TENANT', 'omnibase')
        context = os.getenv('OMNINODE_CONTEXT', 'onex')
        
        # Map event types to topic classes and names
        if event.event_type.startswith('core.database.'):
            topic_class = 'evt'
            if 'query_completed' in event.event_type:
                topic_name = 'postgres-query-completed'
            elif 'query_failed' in event.event_type:
                topic_name = 'postgres-query-failed'
            elif 'health_check' in event.event_type:
                topic_name = 'postgres-health-response'
                topic_class = 'qrs'
            else:
                topic_name = 'postgres-operation'
        else:
            # Default topic routing
            topic_class = 'evt'
            topic_name = event.event_type.replace('.', '-').lower()
        
        # Build OmniNode topic: <env>.<tenant>.<context>.<class>.<topic>.<version>
        topic = f"{env}.{tenant}.{context}.{topic_class}.{topic_name}.v1"
        
        return topic


def create_infrastructure_container() -> ONEXContainer:
    """
    Create infrastructure container with all shared dependencies.

    Per user requirements:
    - "it should be get_service("ProtocolEventBus") and we should have a onexcontainer
      with all dependencies at the root of the service group"
    - "Everything needs to be resolved by duck typing"

    Returns:
        Configured ONEXContainer with infrastructure dependencies
    """
    # Create base ONEX container
    container = ONEXContainer()

    # Set up all shared dependencies for infrastructure services
    _setup_infrastructure_dependencies(container)

    # Bind custom get_service method that handles our infrastructure services
    _bind_infrastructure_get_service_method(container)

    return container


def _setup_infrastructure_dependencies(container: ONEXContainer):
    """Set up all dependencies needed by infrastructure services."""

    # Get logger for container setup
    logger = logging.getLogger(__name__)
    
    # Event Bus - Proper ProtocolEventBus implementation for RedPanda
    event_bus = RedPandaEventBus()
    logger.info(f"Created RedPanda event bus: {type(event_bus).__name__}")

    # Schema Loader - required by MixinEventDrivenNode
    schema_loader = UtilitySchemaLoader()
    logger.info(f"Created schema loader: {type(schema_loader).__name__}")

    # PostgreSQL Connection Manager - required by some infrastructure services
    try:
        from omnibase_infra.infrastructure.postgres_connection_manager import PostgresConnectionManager
        connection_manager = PostgresConnectionManager()
        logger.info(f"Created connection manager: {type(connection_manager).__name__}")
    except Exception as e:
        logger.warning(f"PostgreSQL connection manager unavailable: {e}")
        connection_manager = None
    
    # Register services in the container's service registry
    _register_service(container, "event_bus", event_bus)
    _register_service(container, "ProtocolEventBus", event_bus)
    _register_service(container, "schema_loader", schema_loader)
    _register_service(container, "ProtocolSchemaLoader", schema_loader)
    if connection_manager:
        _register_service(container, "postgres_connection_manager", connection_manager)
        _register_service(container, "PostgresConnectionManager", connection_manager)
    
    # Verify registration
    logger.info("Registered services verification:")
    logger.info(f"  ProtocolEventBus: {type(container.get_service('ProtocolEventBus')).__name__ if container.get_service('ProtocolEventBus') else 'None'}")
    logger.info(f"  event_bus: {type(container.get_service('event_bus')).__name__ if container.get_service('event_bus') else 'None'}")
    postgres_manager = None
    try:
        postgres_manager = container.get_service('postgres_connection_manager')
    except Exception:
        pass
    logger.info(f"  postgres_connection_manager: {type(postgres_manager).__name__ if postgres_manager else 'None'}")
    if connection_manager:
        logger.info("  PostgreSQL connection manager successfully initialized")
    else:
        logger.info("  PostgreSQL connection manager skipped (environment not configured)")


def _register_service(container: ONEXContainer, service_name: str, service_instance):
    """Register a service in the container for later retrieval."""
    # Use the ONEX container's native service registration
    container.register_service(service_name, service_instance)


def _bind_infrastructure_get_service_method(container: ONEXContainer):
    """Configure infrastructure container with proper dependency injection."""
    # The ModelONEXContainer should handle get_service natively
    # We just need to register our services properly in the container
    pass