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
import os
from typing import Callable, Optional, Type, TypeVar, Union

from omnibase_core.core.onex_container import ModelONEXContainer as ONEXContainer
from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus
from omnibase_core.model.core.model_onex_event import ModelOnexEvent
from omnibase_core.utils.generation.utility_schema_loader import UtilitySchemaLoader

T = TypeVar("T")


class KafkaProducerPool:
    """Singleton connection pool for Kafka producers to avoid connection overhead."""
    
    _instance = None
    _producers = {}
    _failed_producers = {}  # Track failed producers for cleanup
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._producers = {}
            self._failed_producers = {}
            self._initialized = True
    
    async def get_producer(self, bootstrap_servers: list, **config):
        """
        Get or create a producer for the given server configuration.
        
        Args:
            bootstrap_servers: List of Kafka bootstrap servers
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
                return producer
            else:
                # Remove unhealthy producer
                print(f"Removing unhealthy Kafka producer for servers: {bootstrap_servers}")
                await self._remove_producer(servers_key)
        
        # Skip creation if this producer has failed recently
        if servers_key in self._failed_producers:
            fail_time = self._failed_producers[servers_key]
            if (asyncio.get_event_loop().time() - fail_time) < 60:  # 60 second backoff
                print(f"Skipping producer creation for {servers_key} - recent failure")
                return None
        
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
            
            producer = AIOKafkaProducer(**producer_config)
            await producer.start()
            self._producers[servers_key] = producer
            
            # Clear failure tracking on success
            if servers_key in self._failed_producers:
                del self._failed_producers[servers_key]
            
            print(f"Created new Kafka producer in pool for servers: {bootstrap_servers}")
            return producer
            
        except ImportError:
            print("WARNING: aiokafka not available, producer pool disabled")
            return None
        except Exception as e:
            print(f"Failed to create Kafka producer: {e}")
            # Track failure for backoff
            self._failed_producers[servers_key] = asyncio.get_event_loop().time()
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
                print(f"Stopped producer for servers: {servers_key}")
            except Exception as e:
                print(f"Error stopping producer: {e}")
            finally:
                del self._producers[servers_key]
    
    async def close_all(self):
        """Close all producers in the pool."""
        for servers_key, producer in self._producers.items():
            try:
                await producer.stop()
                print(f"Closed Kafka producer for servers: {servers_key}")
            except Exception as e:
                print(f"Error closing Kafka producer: {e}")
        self._producers.clear()


class RedPandaEventBus(ProtocolEventBus):
    """
    Proper ProtocolEventBus implementation for RedPanda/Kafka integration.
    
    Conforms to the standard ProtocolEventBus interface using OnexEvent objects
    and publishes them to appropriate RedPanda topics using OmniNode topic routing.
    """
    
    def __init__(self, credentials=None, **kwargs):
        """Initialize RedPanda event bus with proper protocol compliance."""
        # RedPanda connection configuration - use ONEX environment patterns
        redpanda_host = os.getenv('REDPANDA_HOST', 'localhost')
        redpanda_port = os.getenv('REDPANDA_EXTERNAL_PORT', '29102')
        self._bootstrap_servers = [f"{redpanda_host}:{redpanda_port}"]
        
        # Use producer pool for efficient connection management
        self._producer_pool = KafkaProducerPool()
        self._producer = None
        
        print(f"RedPanda event bus initialized with servers: {self._bootstrap_servers}")
        
        # Protocol-compliant subscriber management
        self._subscribers = []
    
    def publish(self, event: ModelOnexEvent) -> None:
        """
        Publish an event to the bus (synchronous).
        
        Args:
            event: OnexEvent to emit
        """
        # Run async publish in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule as task if loop is running
                loop.create_task(self.publish_async(event))
            else:
                # Run directly if no loop is running
                loop.run_until_complete(self.publish_async(event))
        except Exception as e:
            print(f"RedPanda sync publish failed: {str(e)}")
    
    async def publish_async(self, event: ModelOnexEvent) -> None:
        """
        Publish an event to the bus (asynchronous) with retry and exponential backoff.
        
        Args:
            event: OnexEvent to emit
        """
        max_retries = int(os.getenv('REDPANDA_MAX_RETRIES', '3'))
        base_delay = float(os.getenv('REDPANDA_BASE_DELAY_SECONDS', '0.1'))
        
        for attempt in range(max_retries + 1):
            try:
                # Get producer from pool (creates if needed)
                producer = await self._producer_pool.get_producer(self._bootstrap_servers)
                
                if not producer:
                    # Mock publishing for testing without aiokafka
                    print(f"MOCK: Publishing OnexEvent {event.event_type} with correlation_id={event.correlation_id}")
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
                
                print(f"Published OnexEvent to RedPanda topic: {topic} (correlation_id={event.correlation_id})")
                return  # Success - exit retry loop
                
            except Exception as e:
                if attempt == max_retries:
                    # Final attempt failed - log error but don't raise
                    print(f"RedPanda async publish failed after {max_retries + 1} attempts: {str(e)}")
                    return
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                print(f"RedPanda publish attempt {attempt + 1} failed, retrying in {delay:.2f}s: {str(e)}")
                await asyncio.sleep(delay)
    
    def subscribe(self, callback: Callable[[ModelOnexEvent], None]) -> None:
        """
        Subscribe a callback to receive events (synchronous).
        
        Args:
            callback: Callable invoked with each OnexEvent
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)
            print(f"Subscribed callback to RedPanda event bus")
    
    async def subscribe_async(self, callback: Callable[[ModelOnexEvent], None]) -> None:
        """
        Subscribe a callback to receive events (asynchronous).
        
        Args:
            callback: Callable invoked with each OnexEvent
        """
        self.subscribe(callback)
    
    def unsubscribe(self, callback: Callable[[ModelOnexEvent], None]) -> None:
        """
        Unsubscribe a previously registered callback (synchronous).
        
        Args:
            callback: Callable to remove
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            print(f"Unsubscribed callback from RedPanda event bus")
    
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
        print("Cleared all subscribers from RedPanda event bus")
    
    async def close(self):
        """Close RedPanda producer connection."""
        if self._producer:
            await self._producer.stop()
            self._producer = None
    
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

    # Event Bus - Proper ProtocolEventBus implementation for RedPanda
    event_bus = RedPandaEventBus()
    print(f"Created RedPanda event bus: {type(event_bus).__name__}")

    # Schema Loader - required by MixinEventDrivenNode
    schema_loader = UtilitySchemaLoader()
    print(f"Created schema loader: {type(schema_loader).__name__}")

    # PostgreSQL Connection Manager - required by infrastructure services
    from omnibase_infra.infrastructure.postgres_connection_manager import PostgresConnectionManager
    connection_manager = PostgresConnectionManager()
    print(f"Created connection manager: {type(connection_manager).__name__}")
    
    # Register services in the container's service registry
    _register_service(container, "event_bus", event_bus)
    _register_service(container, "ProtocolEventBus", event_bus)
    _register_service(container, "schema_loader", schema_loader)
    _register_service(container, "ProtocolSchemaLoader", schema_loader)
    _register_service(container, "postgres_connection_manager", connection_manager)
    _register_service(container, "PostgresConnectionManager", connection_manager)
    
    # Verify registration
    print(f"Registered services verification:")
    print(f"  ProtocolEventBus: {type(container.get_service('ProtocolEventBus')).__name__ if container.get_service('ProtocolEventBus') else 'None'}")
    print(f"  event_bus: {type(container.get_service('event_bus')).__name__ if container.get_service('event_bus') else 'None'}")
    print(f"  postgres_connection_manager: {type(container.get_service('postgres_connection_manager')).__name__ if container.get_service('postgres_connection_manager') else 'None'}")


def _register_service(container: ONEXContainer, service_name: str, service_instance):
    """Register a service in the container for later retrieval."""
    # Use the ONEX container's native service registration
    container.register_service(service_name, service_instance)


def _bind_infrastructure_get_service_method(container: ONEXContainer):
    """Configure infrastructure container with proper dependency injection."""
    # The ModelONEXContainer should handle get_service natively
    # We just need to register our services properly in the container
    pass