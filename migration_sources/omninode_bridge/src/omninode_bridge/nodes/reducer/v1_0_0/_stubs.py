#!/usr/bin/env python3
"""
Stub implementations for omnibase_core dependencies.

Used when omnibase_core is not available (e.g., in bridge/demo environments).
These stubs provide minimal API compatibility for testing without full ONEX infrastructure.
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

if TYPE_CHECKING:
    # Forward references for type checking without circular imports
    pass


# ============================================================================
# Logging Stubs
# ============================================================================


class LogLevel(str, Enum):
    """Stub log level enum."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def emit_log_event(
    level: LogLevel,
    message: str,
    context: Optional[dict[str, Any]] = None,
) -> None:
    """
    Stub structured logging function.

    In stub mode, this just uses Python's standard logging.
    In production with omnibase_core, this would use structured logging.

    Args:
        level: Log level
        message: Log message
        context: Additional context dict
    """
    logger = logging.getLogger("omninode_bridge.reducer")
    log_method = getattr(logger, level.value.lower(), logger.info)
    if context:
        log_method(f"{message} | context={context}")
    else:
        log_method(message)


class NodeReducer:
    """Stub base class for Reducer nodes."""

    def __init__(self, container: "ModelONEXContainer") -> None:
        """Initialize reducer with container.

        Args:
            container: ONEX DI container with service resolution
        """
        # Call super().__init__() to maintain cooperative inheritance MRO chain
        super().__init__()
        self.container = container
        self.node_id = str(uuid4())

    async def execute_reduction(
        self, contract: "ModelContractReducer"
    ) -> "ModelContractReducer":
        """Execute reduction logic (stub implementation).

        Args:
            contract: Reducer contract with input state and stream data

        Returns:
            Updated contract with reduction results

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement execute_reduction")


class _ConfigValue:
    """Wrapper for config values to support from_value() method."""

    def __init__(self, value: Any = None):
        self._value = value

    def from_value(self, value: Any) -> None:
        """Set the config value (dependency_injector compatibility)."""
        self._value = value

    def __call__(self) -> Any:
        """Get the config value when called."""
        return self._value

    def __str__(self) -> str:
        """String representation."""
        return str(self._value)

    def __repr__(self) -> str:
        """String representation."""
        return repr(self._value)


class _ConfigDict(dict):
    """
    Config dict with from_dict() method for compatibility.

    This allows stub container config to work with both dict-style access
    (config.get()) and dependency_injector-style initialization (config.from_dict()).
    """

    def from_dict(self, data: dict[str, Any]) -> None:
        """Update config from dict (dependency_injector compatibility)."""
        self.update(data)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get config value with default."""
        value = super().get(key, default)
        # Unwrap _ConfigValue objects
        if isinstance(value, _ConfigValue):
            return value._value
        return value

    def __getattr__(self, name: str) -> _ConfigValue:
        """Get config value as attribute (dependency_injector compatibility)."""
        if name.startswith("_"):
            # Don't interfere with internal attributes
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        # Return a _ConfigValue wrapper that supports from_value()
        if name not in self:
            # Create empty config value if it doesn't exist
            self[name] = None
        value = self[name]
        # Wrap in _ConfigValue if not already wrapped
        if not isinstance(value, _ConfigValue):
            config_value = _ConfigValue(value)
            self[name] = config_value
            return config_value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        """Set config value as attribute."""
        if name.startswith("_"):
            # Internal attributes use normal setattr
            super().__setattr__(name, value)
        else:
            # Config values go into dict
            if isinstance(value, _ConfigValue):
                self[name] = value
            else:
                self[name] = _ConfigValue(value)


class ModelONEXContainer:
    """Stub ONEX dependency injection container with service resolution."""

    _initialized: bool = False

    def __init__(
        self,
        name: str = "test_container",
        version: str = "1.0.0",
        services: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize DI container with service resolution capabilities."""
        self.name = name
        self.version = version
        self.services = services or {}
        # Use _ConfigDict for compatibility with both dict and dependency_injector APIs
        self.config = _ConfigDict(config or {})
        self._services: dict[str, Any] = services or {}

    def get(self, key: str, default: Optional[object] = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key to retrieve
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def get_service(self, service_name: str) -> Optional[object]:
        """Get service from container.

        Args:
            service_name: Name of the service to retrieve

        Returns:
            Service instance or None if not found
        """
        return self._services.get(service_name)

    def register_service(self, service_name: str, service: object) -> None:
        """Register service in container.

        Args:
            service_name: Name to register the service under
            service: Service instance to register
        """
        self._services[service_name] = service

    async def initialize(self) -> None:
        """
        Initialize all services (async initialization).

        Note: Uses print() for logging in stub mode. Production code
        should use structured logging when omnibase_core is available.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Create and register KafkaClient if not already present
        if "kafka_client" not in self._services:
            try:
                # Import KafkaClient
                from omninode_bridge.services.kafka_client import KafkaClient

                # Get Kafka bootstrap servers from config
                kafka_broker_url = self.config.get(
                    "kafka_broker_url", "omninode-bridge-redpanda:9092"
                )

                # Create KafkaClient instance
                kafka_client = KafkaClient(bootstrap_servers=kafka_broker_url)

                # Register in services
                self._services["kafka_client"] = kafka_client

                print(
                    f"KafkaClient registered with bootstrap_servers: {kafka_broker_url}"
                )
            except ImportError as e:
                print(f"Warning: Could not import KafkaClient: {e}")
            except Exception as e:
                print(f"Warning: Could not create KafkaClient: {e}")

        # Initialize any services that need async setup
        for service_name, service in self._services.items():
            if hasattr(service, "connect"):
                try:
                    await service.connect()
                    print(f"Service '{service_name}' connected successfully")
                except Exception as e:
                    print(f"Warning: Failed to connect service '{service_name}': {e}")

        self._initialized = True

    async def cleanup(self) -> None:
        """
        Cleanup all services.

        Note: Uses print() for logging in stub mode. Production code
        should use structured logging when omnibase_core is available.
        """
        for service in self._services.values():
            if hasattr(service, "disconnect"):
                await service.disconnect()

        if hasattr(self, "_initialized"):
            self._initialized = False


class ModelContractReducer:
    """Stub reducer contract."""

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        node_type: str,
        input_state: Optional[dict[str, Any]] = None,
        input_stream: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize reducer contract.

        Args:
            name: Contract name
            version: Contract version
            description: Contract description
            node_type: Node type (should be 'reducer')
            input_state: Current state to reduce with (optional)
            input_stream: Stream of items to reduce (optional)
            **kwargs: Additional dynamic attributes
        """
        self.name = name
        self.version = version
        self.description = description
        self.node_type = node_type
        self.input_state = input_state
        self.input_stream = input_stream
        # Store any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class ModelAggregationSubcontract:
    """Stub aggregation subcontract."""

    def __init__(self, aggregation_type: str = "namespace", **kwargs: Any) -> None:
        """Initialize aggregation subcontract."""
        self.aggregation_type = aggregation_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class ModelFSMSubcontract:
    """Stub FSM subcontract."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize FSM subcontract."""
        for key, value in kwargs.items():
            setattr(self, key, value)


class ModelStateManagementSubcontract:
    """Stub state management subcontract."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize state management subcontract."""
        for key, value in kwargs.items():
            setattr(self, key, value)


# Backward compatibility alias (ModelContainer renamed to ModelONEXContainer)
ModelContainer = ModelONEXContainer
