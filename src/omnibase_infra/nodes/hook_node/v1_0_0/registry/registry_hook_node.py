"""Hook Node Registry - Container Injection Setup.

Defines dependency injection configuration for Hook Node following ONEX container patterns.
Registers protocol dependencies and service bindings required for webhook notification operations.

Dependencies:
- ProtocolHttpClient: HTTP client for webhook delivery
- ProtocolEventBus: Event bus for infrastructure events
- NodeHookEffect: Main hook node service implementation

Registration Pattern:
- Protocol-based dependency injection (no isinstance usage)
- Container service resolution with proper error handling
- Lazy loading of dependencies for optimal performance
"""

from typing import Dict, Any, Optional

from omnibase_core.core.onex_container import ONEXContainer
from omnibase_core.protocol.protocol_http_client import ProtocolHttpClient
from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus

from ..node import NodeHookEffect


class HookNodeRegistry:
    """
    Registry for Hook Node dependency injection and service configuration.

    Follows ONEX container patterns for protocol-based dependency resolution.
    Ensures all required services are properly registered and available
    for Hook Node operations.
    """

    @staticmethod
    def register_dependencies(container: ONEXContainer) -> None:
        """
        Register Hook Node dependencies in the ONEX container.

        This method should be called during application startup to ensure
        all required protocols and services are properly registered.

        Args:
            container: ONEX container instance for service registration
        """
        # Hook Node requires both HTTP client and Event Bus protocols
        # These are typically registered by the infrastructure layer

        # Validate that required protocols are available
        http_client = container.get_service("ProtocolHttpClient")
        if http_client is None:
            raise RuntimeError(
                "ProtocolHttpClient not registered - required for Hook Node webhook delivery"
            )

        event_bus = container.get_service("ProtocolEventBus")
        if event_bus is None:
            raise RuntimeError(
                "ProtocolEventBus not registered - required for Hook Node event integration"
            )

        # Register Hook Node service itself
        container.register_singleton("NodeHookEffect", lambda c: NodeHookEffect(c))

    @staticmethod
    def get_service_metadata() -> Dict[str, Any]:
        """
        Get metadata about Hook Node service and its dependencies.

        Returns:
            Dict containing service metadata for introspection and health checks
        """
        return {
            "service_name": "NodeHookEffect",
            "service_type": "EFFECT",
            "domain": "infrastructure",
            "dependencies": [
                {
                    "name": "ProtocolHttpClient",
                    "type": "protocol",
                    "required": True,
                    "description": "HTTP client for webhook delivery"
                },
                {
                    "name": "ProtocolEventBus",
                    "type": "protocol",
                    "required": True,
                    "description": "Event bus for infrastructure event integration"
                }
            ],
            "capabilities": [
                "webhook_notification_delivery",
                "retry_policy_support",
                "circuit_breaker_protection",
                "multiple_auth_methods",
                "structured_logging",
                "performance_metrics"
            ],
            "supported_destinations": [
                "slack_webhooks",
                "discord_webhooks",
                "generic_http_webhooks",
                "custom_api_endpoints"
            ]
        }

    @staticmethod
    def validate_configuration(container: ONEXContainer) -> bool:
        """
        Validate that Hook Node is properly configured in the container.

        Args:
            container: ONEX container to validate

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Check HTTP client protocol
            http_client = container.get_service("ProtocolHttpClient")
            if not isinstance(http_client, ProtocolHttpClient):
                return False

            # Check event bus protocol
            event_bus = container.get_service("ProtocolEventBus")
            if not isinstance(event_bus, ProtocolEventBus):
                return False

            # Attempt to create Hook Node instance
            hook_node = NodeHookEffect(container)
            if hook_node is None:
                return False

            return True

        except Exception:
            return False


def register_hook_node(container: ONEXContainer) -> None:
    """
    Convenience function to register Hook Node with all dependencies.

    Args:
        container: ONEX container for service registration
    """
    HookNodeRegistry.register_dependencies(container)


def get_hook_node_metadata() -> Dict[str, Any]:
    """
    Convenience function to get Hook Node metadata.

    Returns:
        Dict containing service metadata
    """
    return HookNodeRegistry.get_service_metadata()