"""Node adapters for omnibase_core node classes.

This module provides adapter classes that make omnibase_core node classes
protocol-compliant for duck typing until omnibase_core natively implements
the protocols defined in omnibase_spi.

Author: OmniNode Bridge Team
Created: 2025-10-30
"""

from typing import Any

# Try to import from omnibase_core, fallback to local stubs
try:
    from omnibase_core.nodes.node_compute import NodeCompute
    from omnibase_core.nodes.node_effect import NodeEffect
    from omnibase_core.nodes.node_orchestrator import NodeOrchestrator
    from omnibase_core.nodes.node_reducer import NodeReducer

    USING_OMNIBASE_CORE = True
except ImportError:
    # Fallback to stub implementations
    from ..nodes.orchestrator.v1_0_0._stubs import NodeOrchestrator
    from ..nodes.reducer.v1_0_0._stubs import NodeReducer

    # Create minimal stubs for Effect and Compute
    class NodeEffect:
        """Stub NodeEffect for fallback."""

        pass

    class NodeCompute:
        """Stub NodeCompute for fallback."""

        pass

    USING_OMNIBASE_CORE = False


class AdapterNodeOrchestrator(NodeOrchestrator):
    """
    Adapter making NodeOrchestrator protocol-compliant.

    This adapter extends NodeOrchestrator to ensure it implements ProtocolNode,
    providing a consistent interface for duck typing with type checkers.

    TODO [UPSTREAM omnibase_core v0.2.0]:
    - Make NodeOrchestrator implement ProtocolNode natively
    - Add explicit @runtime_checkable decorator
    - Standardize node interface across all node types
    - Add process() as main entry point (currently execute_orchestration)

    Usage:
        >>> from .protocols import ProtocolNode
        >>> node: ProtocolNode = AdapterNodeOrchestrator(container)
        >>> result = await node.process(input_data)

    Type Checking:
        >>> assert isinstance(node, ProtocolNode)  # True with runtime_checkable
    """

    async def process(self, input_data: Any) -> Any:
        """
        Process input data according to orchestrator logic.

        This method provides a unified interface for all node types,
        delegating to the specific execute_orchestration method.

        Args:
            input_data: Input contract or data to process

        Returns:
            Processed output data or contract

        Raises:
            ModelOnexError: On processing failures

        Note:
            This is a wrapper around execute_orchestration() to provide
            a consistent interface across all node types. The actual
            implementation delegates to the base class method.
        """
        return await self.execute_orchestration(input_data)

    def get_contract(self) -> Any:
        """
        Retrieve the node's contract definition.

        TODO [UPSTREAM omnibase_core v0.2.0]:
        - Add get_contract() to base NodeOrchestrator class
        - Store contract during initialization
        - Return contract metadata for introspection

        Returns:
            The node's contract model

        Raises:
            NotImplementedError: Until upstream implements contract storage
        """
        if hasattr(self, "_contract"):
            return self._contract
        raise NotImplementedError(
            "Contract retrieval not yet implemented in base NodeOrchestrator. "
            "TODO: Add upstream support for contract storage and retrieval."
        )


class AdapterNodeReducer(NodeReducer):
    """
    Adapter making NodeReducer protocol-compliant.

    This adapter extends NodeReducer to ensure it implements ProtocolNode,
    providing a consistent interface for duck typing with type checkers.

    TODO [UPSTREAM omnibase_core v0.2.0]:
    - Make NodeReducer implement ProtocolNode natively
    - Add explicit @runtime_checkable decorator
    - Standardize node interface across all node types
    - Add process() as main entry point (currently execute_reduction)

    Usage:
        >>> from .protocols import ProtocolNode
        >>> node: ProtocolNode = AdapterNodeReducer(container)
        >>> result = await node.process(input_data)
    """

    async def process(self, input_data: Any) -> Any:
        """
        Process input data according to reducer logic.

        This method provides a unified interface for all node types,
        delegating to the specific execute_reduction method.

        Args:
            input_data: Input contract or data to process

        Returns:
            Processed output data or contract

        Raises:
            ModelOnexError: On processing failures
        """
        return await self.execute_reduction(input_data)

    def get_contract(self) -> Any:
        """
        Retrieve the node's contract definition.

        Returns:
            The node's contract model

        Raises:
            NotImplementedError: Until upstream implements contract storage
        """
        if hasattr(self, "_contract"):
            return self._contract
        raise NotImplementedError(
            "Contract retrieval not yet implemented in base NodeReducer. "
            "TODO: Add upstream support for contract storage and retrieval."
        )


class AdapterNodeEffect(NodeEffect):
    """
    Adapter making NodeEffect protocol-compliant.

    This adapter extends NodeEffect to ensure it implements ProtocolNode,
    providing a consistent interface for duck typing with type checkers.

    TODO [UPSTREAM omnibase_core v0.2.0]:
    - Make NodeEffect implement ProtocolNode natively
    - Add explicit @runtime_checkable decorator
    - Standardize node interface across all node types
    - Add process() as main entry point (currently execute_effect)

    Usage:
        >>> from .protocols import ProtocolNode
        >>> node: ProtocolNode = AdapterNodeEffect(container)
        >>> result = await node.process(input_data)
    """

    async def process(self, input_data: Any) -> Any:
        """
        Process input data according to effect logic.

        This method provides a unified interface for all node types,
        delegating to the specific execute_effect method.

        Args:
            input_data: Input contract or data to process

        Returns:
            Processed output data or contract

        Raises:
            ModelOnexError: On processing failures
        """
        return await self.execute_effect(input_data)

    def get_contract(self) -> Any:
        """
        Retrieve the node's contract definition.

        Returns:
            The node's contract model

        Raises:
            NotImplementedError: Until upstream implements contract storage
        """
        if hasattr(self, "_contract"):
            return self._contract
        raise NotImplementedError(
            "Contract retrieval not yet implemented in base NodeEffect. "
            "TODO: Add upstream support for contract storage and retrieval."
        )


class AdapterNodeCompute(NodeCompute):
    """
    Adapter making NodeCompute protocol-compliant.

    This adapter extends NodeCompute to ensure it implements ProtocolNode,
    providing a consistent interface for duck typing with type checkers.

    TODO [UPSTREAM omnibase_core v0.2.0]:
    - Make NodeCompute implement ProtocolNode natively
    - Add explicit @runtime_checkable decorator
    - Standardize node interface across all node types
    - Add process() as main entry point (currently execute_compute)

    Usage:
        >>> from .protocols import ProtocolNode
        >>> node: ProtocolNode = AdapterNodeCompute(container)
        >>> result = await node.process(input_data)
    """

    async def process(self, input_data: Any) -> Any:
        """
        Process input data according to compute logic.

        This method provides a unified interface for all node types,
        delegating to the specific execute_compute method.

        Args:
            input_data: Input contract or data to process

        Returns:
            Processed output data or contract

        Raises:
            ModelOnexError: On processing failures
        """
        return await self.execute_compute(input_data)

    def get_contract(self) -> Any:
        """
        Retrieve the node's contract definition.

        Returns:
            The node's contract model

        Raises:
            NotImplementedError: Until upstream implements contract storage
        """
        if hasattr(self, "_contract"):
            return self._contract
        raise NotImplementedError(
            "Contract retrieval not yet implemented in base NodeCompute. "
            "TODO: Add upstream support for contract storage and retrieval."
        )


# Type aliases for convenience
OrchestratorNode = AdapterNodeOrchestrator
ReducerNode = AdapterNodeReducer
EffectNode = AdapterNodeEffect
ComputeNode = AdapterNodeCompute
