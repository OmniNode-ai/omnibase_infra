"""Protocol definitions for omnibase_core classes.

This module defines Protocol classes that omnibase_core classes should implement.
These protocols enable duck typing and type-safe interfaces until omnibase_spi
includes these protocols natively.

TODO [UPSTREAM omnibase_spi v0.2.0]:
- Move all protocols to omnibase_spi.protocols module
- Add @runtime_checkable decorators
- Ensure backward compatibility for existing code

Author: OmniNode Bridge Team
Created: 2025-10-30
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProtocolContainer(Protocol):
    """Protocol for dependency injection containers.

    This protocol defines the interface that container implementations should follow
    for service registration, retrieval, and configuration management.

    TODO [UPSTREAM omnibase_spi v0.2.0]: Add to omnibase_spi.protocols

    Expected Methods:
        get_service(name: str) -> Any: Retrieve a registered service by name
        register_service(name: str, service: Any) -> None: Register a service

    Expected Properties:
        config: dict - Configuration dictionary for the container

    Usage:
        >>> container: ProtocolContainer = AdapterModelContainer()
        >>> container.register_service("my_service", my_service_instance)
        >>> service = container.get_service("my_service")
    """

    def get_service(self, name: str) -> Any:
        """Retrieve a registered service by name.

        Args:
            name: The service identifier

        Returns:
            The registered service instance

        Raises:
            KeyError: If service not found
        """
        ...

    def register_service(self, name: str, service: Any) -> None:
        """Register a service with the container.

        Args:
            name: The service identifier
            service: The service instance to register
        """
        ...

    @property
    def config(self) -> dict:
        """Get the container configuration.

        Returns:
            Configuration dictionary
        """
        ...


@runtime_checkable
class ProtocolNode(Protocol):
    """Protocol for ONEX nodes.

    This protocol defines the interface that all ONEX nodes should implement,
    regardless of node type (Effect, Compute, Reducer, Orchestrator).

    TODO [UPSTREAM omnibase_spi v0.2.0]: Add to omnibase_spi.protocols

    Expected Methods:
        process(input_data: Any) -> Any: Main processing method (async)
        get_contract() -> Any: Retrieve the node's contract

    Usage:
        >>> node: ProtocolNode = AdapterNodeEffect()
        >>> result = await node.process(input_data)
        >>> contract = node.get_contract()
    """

    async def process(self, input_data: Any) -> Any:
        """Process input data according to the node's logic.

        Args:
            input_data: Input data to process

        Returns:
            Processed output data

        Raises:
            ModelOnexError: On processing failures
        """
        ...

    def get_contract(self) -> Any:
        """Retrieve the node's contract definition.

        Returns:
            The node's contract model
        """
        ...


@runtime_checkable
class ProtocolOnexError(Protocol):
    """Protocol for ONEX errors.

    This protocol defines the interface for standardized error handling
    in the ONEX ecosystem, ensuring consistent error structure and metadata.

    TODO [UPSTREAM omnibase_spi v0.2.0]: Add to omnibase_spi.protocols

    Expected Attributes:
        code: str - Error code (e.g., "VALIDATION_ERROR")
        message: str - Human-readable error message
        details: dict - Additional error context and metadata

    Usage:
        >>> error: ProtocolOnexError = ModelOnexError(
        ...     code="VALIDATION_ERROR",
        ...     message="Invalid input",
        ...     details={"field": "username"}
        ... )
    """

    code: str
    message: str
    details: dict


@runtime_checkable
class ProtocolContract(Protocol):
    """Protocol for ONEX contracts.

    This protocol defines the interface for node contracts, which specify
    the node's capabilities, inputs, outputs, and metadata.

    TODO [UPSTREAM omnibase_spi v0.2.0]: Add to omnibase_spi.protocols

    Expected Attributes:
        name: str - Contract name
        version: str - Contract version (semver)
        node_type: str - Node type (effect, compute, reducer, orchestrator)

    Expected Methods:
        validate() -> bool: Validate contract structure
        to_dict() -> dict: Serialize to dictionary

    Usage:
        >>> contract: ProtocolContract = ModelContractEffect(...)
        >>> if contract.validate():
        ...     config = contract.to_dict()
    """

    name: str
    version: str
    node_type: str

    def validate(self) -> bool:
        """Validate the contract structure.

        Returns:
            True if contract is valid

        Raises:
            ModelOnexError: If validation fails
        """
        ...

    def to_dict(self) -> dict:
        """Serialize contract to dictionary.

        Returns:
            Dictionary representation of the contract
        """
        ...


# Type aliases for convenience
Container = ProtocolContainer
Node = ProtocolNode
OnexError = ProtocolOnexError
Contract = ProtocolContract
