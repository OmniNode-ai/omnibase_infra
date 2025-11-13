"""
Persistence Layer Protocols.

Defines Protocol interfaces for type-safe persistence operations without
circular dependencies. These protocols enable strong typing of database
adapter nodes in CRUD functions.

ONEX v2.0 Compliance:
- Protocol-based typing for dependency inversion
- Strongly-typed method signatures
- Optional logger support via hasattr pattern
"""

from typing import Any, Protocol

from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_database_operation_input import (
    ModelDatabaseOperationInput,
)
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.outputs.model_database_operation_output import (
    ModelDatabaseOperationOutput,
)


class DatabaseAdapterProtocol(Protocol):
    """
    Protocol for database adapter nodes.

    This protocol defines the interface required by persistence layer CRUD
    functions. Any node implementing this protocol can be passed to CRUD
    functions without requiring concrete type imports (avoiding circular
    dependencies).

    Required Methods:
        process: Execute database operations with strongly-typed input/output

    Optional Attributes:
        _logger: Logger instance for operation logging (checked via hasattr)

    Usage Example:
        >>> from persistence.protocols import DatabaseAdapterProtocol
        >>> from persistence.bridge_state_crud import create_bridge_state
        >>>
        >>> # No more node: Any - strong typing!
        >>> async def my_function(node: DatabaseAdapterProtocol):
        ...     bridge_state = await create_bridge_state(
        ...         bridge_id=uuid4(),
        ...         namespace="production",
        ...         current_fsm_state="IDLE",
        ...         node=node,  # Type-safe!
        ...         correlation_id=uuid4()
        ...     )

    Type Safety Benefits:
        - mypy validates node has process() method
        - IDE autocomplete for process() signature
        - Runtime Protocol check if needed via isinstance()
        - No circular import issues (Protocol doesn't import concrete classes)

    Protocol Pattern:
        Protocols use structural subtyping (duck typing) rather than nominal
        subtyping. Any object with a compatible process() method satisfies
        this protocol, even without explicit inheritance.
    """

    async def process(
        self,
        operation_input: ModelDatabaseOperationInput,
    ) -> ModelDatabaseOperationOutput:
        """
        Process database operation with strongly-typed input/output.

        Args:
            operation_input: Strongly-typed database operation request with
                entity data, query filters, and correlation tracking

        Returns:
            Operation result with success status, execution metrics, and
            optional result data (e.g., query results, generated IDs)

        Raises:
            OnexError: For validation failures, database errors, or
                unexpected exceptions during operation execution
        """
        ...

    @property
    def _logger(self) -> Any:  # Type as Any since logger is optional
        """
        Optional logger instance for operation logging.

        This property is checked via hasattr() pattern in CRUD functions:
            if hasattr(node, "_logger") and node._logger:
                node._logger.log_operation_complete(...)

        Returns:
            Logger instance or None if not available
        """
        ...
