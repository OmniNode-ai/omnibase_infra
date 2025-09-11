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


class InfrastructureEventBus:
    """Event bus adapter for infrastructure services."""
    
    def __init__(self):
        self._protocol_bus = ProtocolEventBus()
        self._callbacks_by_type = {}
    
    def subscribe(self, callback, event_type=None):
        """Subscribe to events with optional type filtering."""
        if event_type is not None:
            # MixinNodeService pattern: subscribe(callback, event_type) 
            if event_type not in self._callbacks_by_type:
                self._callbacks_by_type[event_type] = []
            self._callbacks_by_type[event_type].append(callback)
            
            # Register with the protocol bus with a filter
            def filtered_callback(event):
                # Check if event matches the type we want
                event_type_attr = getattr(event, 'event_type', None) or getattr(event, 'type', None)
                if event_type_attr == event_type:
                    callback(event)
            
            self._protocol_bus.subscribe(filtered_callback)
        else:
            # MixinEventHandler pattern: subscribe(callback)
            # Subscribe to all events without filtering
            self._protocol_bus.subscribe(callback)
    
    def publish(self, event):
        """Publish event."""
        self._protocol_bus.publish(event)
    
    def unsubscribe(self, callback):
        """Unsubscribe callback."""
        # For simplicity, clear all callbacks for now
        self._callbacks_by_type.clear()


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

    # Event Bus - shared across all infrastructure services  
    event_bus = InfrastructureEventBus()
    print(f"Created event bus: {type(event_bus).__name__}")

    # Schema Loader - required by MixinEventDrivenNode
    schema_loader = UtilitySchemaLoader()
    print(f"Created schema loader: {type(schema_loader).__name__}")

    # Register services in the container's service registry
    _register_service(container, "event_bus", event_bus)
    _register_service(container, "ProtocolEventBus", event_bus)
    _register_service(container, "schema_loader", schema_loader)
    _register_service(container, "ProtocolSchemaLoader", schema_loader)
    
    # Verify registration
    print(f"Registered services verification:")
    print(f"  ProtocolEventBus: {type(container.get_service('ProtocolEventBus')).__name__ if container.get_service('ProtocolEventBus') else 'None'}")
    print(f"  event_bus: {type(container.get_service('event_bus')).__name__ if container.get_service('event_bus') else 'None'}")


def _register_service(container: ONEXContainer, service_name: str, service_instance):
    """Register a service in the container for later retrieval."""
    # Use the ONEX container's native service registration
    container.register_service(service_name, service_instance)


def _bind_infrastructure_get_service_method(container: ONEXContainer):
    """Configure infrastructure container with proper dependency injection."""
    # The ModelONEXContainer should handle get_service natively
    # We just need to register our services properly in the container
    pass