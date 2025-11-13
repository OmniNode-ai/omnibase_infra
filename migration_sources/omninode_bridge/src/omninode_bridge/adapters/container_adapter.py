"""Container adapters for omnibase_core container classes.

This module provides adapter classes that make omnibase_core container classes
protocol-compliant for duck typing until omnibase_core natively implements
the protocols defined in omnibase_spi.

Author: OmniNode Bridge Team
Created: 2025-10-30
"""

from typing import Any, Optional

# For MVP, use stub implementation only
# ModelONEXContainer from omnibase_core has a significantly different interface
# (uses dependency_injector, Configuration providers, protocol-based service registry)
# which would require a more complex adapter. This will be addressed upstream.
from ..nodes.orchestrator.v1_0_0._stubs import ModelContainer as ModelONEXContainer

USING_OMNIBASE_CORE = False

# Note: To use real ModelONEXContainer, upstream changes are required
# See UPSTREAM_CHANGES.md for details


class AdapterModelContainer:
    """
    Adapter making container classes protocol-compliant.

    This adapter wraps either ModelONEXContainer (from omnibase_core) or the
    stub ModelContainer to provide a consistent interface that implements
    ProtocolContainer.

    TODO [UPSTREAM omnibase_core v0.2.0]:
    - Make ModelONEXContainer implement ProtocolContainer natively
    - Standardize container interface across omnibase_core containers
    - Add @runtime_checkable decorator
    - Provide unified container factory

    Usage:
        >>> container = AdapterModelContainer.create(config={"key": "value"})
        >>> container.register_service("my_service", service_instance)
        >>> service = container.get_service("my_service")

    Type Checking:
        >>> from .protocols import ProtocolContainer
        >>> container: ProtocolContainer = AdapterModelContainer.create()
        >>> assert isinstance(container, ProtocolContainer)  # True
    """

    def __init__(
        self,
        wrapped_container: Any,
    ) -> None:
        """Initialize adapter with a container instance.

        Args:
            wrapped_container: The actual container instance to wrap
                (ModelONEXContainer or stub ModelContainer)
        """
        self._container = wrapped_container

    @classmethod
    def create(
        cls,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "AdapterModelContainer":
        """Create a new container instance with configuration.

        This factory method handles differences between omnibase_core
        ModelONEXContainer and stub ModelContainer initialization.

        Args:
            config: Configuration dictionary
            **kwargs: Additional container initialization parameters

        Returns:
            AdapterModelContainer wrapping the created container

        Example:
            >>> container = AdapterModelContainer.create(
            ...     config={
            ...         "metadata_stamping_service_url": "http://localhost:8053",
            ...         "kafka_broker_url": "localhost:9092"
            ...     }
            ... )
        """
        # Currently only using stub implementation
        # Stub ModelContainer accepts config directly
        wrapped = ModelONEXContainer(config=config or {}, **kwargs)

        return cls(wrapped)

    def get_service(self, name: str) -> Any:
        """Retrieve a registered service by name.

        Delegates to the wrapped container's get_service method.

        Args:
            name: The service identifier

        Returns:
            The registered service instance

        Raises:
            KeyError: If service not found (behavior depends on wrapped container)
        """
        # Stub ModelContainer uses get_service(service_name)
        return self._container.get_service(name)

    def register_service(self, name: str, service: Any) -> None:
        """Register a service with the container.

        Delegates to the wrapped container's registration mechanism.

        Args:
            name: The service identifier
            service: The service instance to register
        """
        # Stub ModelContainer has register_service
        self._container.register_service(name, service)

    @property
    def config(self) -> dict:
        """Get the container configuration.

        Returns:
            Configuration dictionary
        """
        # Try different config locations based on container type
        if hasattr(self._container, "config"):
            return self._container.config
        elif hasattr(self._container, "_config"):
            return self._container._config
        elif hasattr(self._container, "_adapter_config"):
            return self._container._adapter_config
        else:
            # Return empty dict as fallback
            return {}

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped container.

        This allows the adapter to transparently expose all methods
        and properties of the wrapped container.

        Args:
            name: Attribute name

        Returns:
            Attribute value from wrapped container

        Raises:
            AttributeError: If attribute not found
        """
        return getattr(self._container, name)

    def __repr__(self) -> str:
        """String representation of adapter."""
        return f"AdapterModelContainer(using_core={USING_OMNIBASE_CORE}, wrapped={self._container})"
