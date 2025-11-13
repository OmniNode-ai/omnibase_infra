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
# Error Handling Stubs
# ============================================================================


class EnumCoreErrorCode(str, Enum):
    """Stub error code enum."""

    OPERATION_FAILED = "OPERATION_FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class ModelOnexError(Exception):
    """Stub ONEX error class."""

    def __init__(
        self,
        code: Optional[EnumCoreErrorCode] = None,
        message: str = "",
        context: Optional[dict[str, Any]] = None,
        error_code: Optional[EnumCoreErrorCode] = None,
    ):
        """Initialize error with support for both code= and error_code= parameters."""
        super().__init__(message)
        # Support both old (code=) and new (error_code=) parameter names
        self.code = error_code or code
        self.error_code = error_code or code
        self.message = message
        self.context = context or {}


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
    logger = logging.getLogger("omninode_bridge.registry")
    log_method = getattr(logger, level.value.lower(), logger.info)
    if context:
        log_method(f"{message} | context={context}")
    else:
        log_method(message)


class NodeEffect:
    """Stub base class for Effect nodes."""

    def __init__(self, container: "ModelONEXContainer") -> None:
        """Initialize effect with container.

        Args:
            container: ONEX dependency injection container
        """
        # Call super().__init__() to maintain cooperative inheritance MRO chain
        super().__init__()
        self.container = container
        self.node_id = str(uuid4())

    async def execute_effect(
        self, contract: "ModelContractEffect"
    ) -> "ModelContractEffect":
        """Execute effect logic (stub implementation).

        Args:
            contract: Effect contract with I/O definition

        Returns:
            Updated contract with effect results

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement execute_effect")


class _ConfigValue:
    """
    Config value wrapper with from_value() method for compatibility.

    This wraps individual config values and provides the from_value() method
    that ModelONEXContainer uses.
    """

    def __init__(self, value: Any = None):
        """Initialize config value."""
        self._value = value

    def from_value(self, value: Any) -> None:
        """Set config value (dependency_injector compatibility)."""
        self._value = value

    def __call__(self) -> Any:
        """Get config value when called."""
        return self._value

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigValue({self._value})"


class _Config:
    """
    Config object with attribute access and from_dict() method.

    This provides both attribute-style access (config.kafka_broker_url)
    and dict-style initialization (config.from_dict()).
    """

    def __init__(
        self,
        data: Optional[dict[str, Any]] = None,
        container: Optional["ModelONEXContainer"] = None,
    ):
        """Initialize config."""
        self._data: dict[str, _ConfigValue] = {}
        self._container = container
        if data:
            self.from_dict(data)

    def from_dict(self, data: dict[str, Any]) -> None:
        """Update config from dict (dependency_injector compatibility)."""
        for key, value in data.items():
            # Wrap each value in _ConfigValue for from_value() support
            config_value = _ConfigValue(value)
            self._data[key] = config_value
            setattr(self, key, config_value)

        # Update container.value to sync with config changes
        if self._container is not None:
            # Convert _ConfigValue objects to raw values for container.value
            self._container.value = {k: v() for k, v in self._data.items()}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get config value with default."""
        config_value = self._data.get(key)
        if config_value is None:
            return default
        return config_value()

    def __getattr__(self, name: str) -> _ConfigValue:
        """Get config value by attribute."""
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        # Return existing or create new ConfigValue
        if name not in self._data:
            config_value: _ConfigValue = _ConfigValue(None)
            self._data[name] = config_value
            return config_value
        result: _ConfigValue = self._data[name]
        return result


class ModelONEXContainer:
    """Stub ONEX dependency injection container."""

    def __init__(
        self,
        name: str = "test_container",
        version: str = "1.0.0",
        services: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize container."""
        self.name = name
        self.version = version
        self.services = services or {}
        # Add value property for registry node compatibility
        self.value = config or {}
        # Use _Config for compatibility with attribute access and from_dict()
        # Pass self to _Config so it can update self.value when from_dict() is called
        self.config = _Config(config, container=self)
        self._services: dict[str, Any] = services or {}

    def get(self, key: str, default: Optional[object] = None) -> object | None:
        """Get configuration value.

        Args:
            key: Configuration key to retrieve
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        result: object | None = self.config.get(key, default)
        return result

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
        # Check initialization status
        initialized: bool = getattr(self, "_initialized", False)
        if initialized:
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

        self._initialized: bool = True

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


class ModelContractEffect:
    """Stub effect contract."""

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        node_type: str,
        io_operations: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize effect contract.

        Args:
            name: Contract name
            version: Contract version
            description: Contract description
            node_type: Node type (should be 'effect')
            io_operations: I/O operations definition (optional)
            **kwargs: Additional dynamic attributes
        """
        self.name = name
        self.version = version
        self.description = description
        self.node_type = node_type
        self.io_operations = io_operations
        # Store any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
