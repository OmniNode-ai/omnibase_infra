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

from typing import Dict, Optional, Union, List

from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_spi.protocols.core import ProtocolHttpClient
from omnibase_spi.protocols.event_bus import ProtocolEventBus

from ..node import NodeHookEffect


class HookNodeRegistry:
    """
    Registry for Hook Node dependency injection and service configuration.

    Follows ONEX container patterns for protocol-based dependency resolution.
    Ensures all required services are properly registered and available
    for Hook Node operations.
    """

    @staticmethod
    def register_dependencies(container: ModelONEXContainer) -> None:
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
    def get_service_metadata() -> Dict[str, Union[str, List[Dict[str, Union[str, bool]]], List[str]]]:
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
    def validate_configuration(container: ModelONEXContainer) -> bool:
        """
        Validate that Hook Node is properly configured in the container.
        Uses protocol duck typing following ONEX patterns - no isinstance usage.

        Args:
            container: ONEX container to validate

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Check HTTP client protocol using duck typing
            http_client = container.get_service("ProtocolHttpClient")
            if http_client is None:
                return False

            # Validate HTTP client has required protocol methods (ONEX-compliant duck typing)
            if not (hasattr(http_client, 'request') and callable(getattr(http_client, 'request'))):
                return False

            # Check event bus protocol using duck typing
            event_bus = container.get_service("ProtocolEventBus")
            if event_bus is None:
                return False

            # Validate event bus has required protocol methods (ONEX-compliant duck typing)
            if not (hasattr(event_bus, 'publish') and callable(getattr(event_bus, 'publish'))):
                return False

            # Attempt to create Hook Node instance
            hook_node = NodeHookEffect(container)
            if hook_node is None:
                return False

            return True

        except Exception:
            return False


def register_hook_node(container: ModelONEXContainer) -> None:
    """
    Convenience function to register Hook Node with all dependencies.

    Args:
        container: ONEX container for service registration
    """
    HookNodeRegistry.register_dependencies(container)


def get_hook_node_metadata() -> Dict[str, Union[str, List[Dict[str, Union[str, bool]]], List[str]]]:
    """
    Convenience function to get Hook Node metadata.

    Returns:
        Dict containing service metadata
    """
    return HookNodeRegistry.get_service_metadata()