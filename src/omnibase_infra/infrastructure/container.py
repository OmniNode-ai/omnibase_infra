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

import types
from typing import Optional, Type, TypeVar, Union

from omnibase_core.core.onex_container import ModelONEXContainer as ONEXContainer
from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus
from omnibase_core.utils.generation.utility_schema_loader import UtilitySchemaLoader

T = TypeVar("T")


class InfrastructureEventBusRedPanda:
    """RedPanda/Kafka event bus implementation for infrastructure services."""
    
    def __init__(self):
        """Initialize RedPanda event bus with configuration from environment."""
        # Import aiokafka for RedPanda integration
        try:
            from aiokafka import AIOKafkaProducer
            import json
            import os
            self._kafka = AIOKafkaProducer
            self._json = json
            self._producer = None
            
            # RedPanda connection configuration
            self._bootstrap_servers = [f"localhost:{os.getenv('REDPANDA_PORT', '9092')}"]
            
            print(f"RedPanda event bus initialized with servers: {self._bootstrap_servers}")
        except ImportError:
            print("WARNING: aiokafka not available, falling back to mock event bus")
            self._kafka = None
            self._json = None
            self._producer = None
            self._bootstrap_servers = []
    
    async def publish_original(self, topic: str, event_data: dict, correlation_id: str = None, partition_key: str = None):
        """
        Publish event to RedPanda topic.
        
        Args:
            topic: RedPanda topic name (e.g., "dev.omnibase.onex.evt.postgres-query-completed.v1")
            event_data: Event data dictionary (typically envelope.model_dump())
            correlation_id: Correlation ID for tracking
            partition_key: Partition key for consistent routing
        """
        if not self._kafka:
            # Mock publishing for testing without aiokafka
            print(f"MOCK: Publishing to topic '{topic}' with correlation_id={correlation_id}")
            return
        
        try:
            # Initialize producer if needed
            if not self._producer:
                self._producer = self._kafka(
                    bootstrap_servers=self._bootstrap_servers,
                    value_serializer=lambda x: self._json.dumps(x).encode('utf-8')
                )
                await self._producer.start()
                print(f"RedPanda producer started for servers: {self._bootstrap_servers}")
            
            # Publish to RedPanda topic
            await self._producer.send_and_wait(
                topic=topic,
                value=event_data,
                key=partition_key.encode('utf-8') if partition_key else None
            )
            
            print(f"Published event to RedPanda topic: {topic} (correlation_id={correlation_id})")
            
        except Exception as e:
            # Log error but don't fail (fire-and-forget pattern)
            print(f"RedPanda publishing failed: {str(e)}")
    
    async def close(self):
        """Close RedPanda producer connection."""
        if self._producer:
            await self._producer.stop()
            self._producer = None
    
    # Legacy ProtocolEventBus interface compatibility
    def subscribe(self, callback, event_type=None):
        """Subscribe compatibility (not implemented for RedPanda publisher)."""
        print(f"Subscribe called with event_type={event_type} (not implemented)")
    
    def unsubscribe(self, callback):
        """Unsubscribe compatibility (not implemented for RedPanda publisher)."""
        print("Unsubscribe called (not implemented)")
        
    def publish(self, *args, **kwargs):
        """
        Compatibility publish method with flexible signature.
        
        Handles different calling patterns from NodeEffectService base class:
        - publish() - no args (compatibility)
        - publish(topic, event_data, ...) - standard RedPanda publishing
        """
        # If no arguments, this is likely a compatibility call - ignore
        if not args and not kwargs:
            print("COMPATIBILITY: Empty publish() call - ignoring")
            return
            
        # If we have arguments, delegate to the async publish method
        if args or kwargs:
            # Extract common parameters
            topic = args[0] if args else kwargs.get('topic')
            event_data = args[1] if len(args) > 1 else kwargs.get('event_data')
            correlation_id = kwargs.get('correlation_id')
            partition_key = kwargs.get('partition_key')
            
            # If we're missing critical parameters, log and return
            if not topic or not event_data:
                print(f"COMPATIBILITY: Incomplete publish call - topic={topic}, event_data={bool(event_data)}")
                return
                
            # Create a simple async wrapper for sync calls
            import asyncio
            try:
                # Try to run in existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule as task if loop is running
                    loop.create_task(self.publish_async(topic, event_data, correlation_id, partition_key))
                else:
                    # Run directly if no loop is running
                    loop.run_until_complete(self.publish_async(topic, event_data, correlation_id, partition_key))
            except Exception as e:
                print(f"COMPATIBILITY: Sync publish failed: {e}")
    
    async def publish_async(self, topic: str, event_data: dict, correlation_id: str = None, partition_key: str = None):
        """Async publish method (the original implementation)."""
        return await self.publish_original(topic, event_data, correlation_id, partition_key)


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

    # Event Bus - RedPanda implementation for infrastructure services  
    event_bus = InfrastructureEventBusRedPanda()
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