# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Base class for deterministic compute plugins.

This module provides PluginComputeBase, an abstract base class for implementing
deterministic compute plugins that integrate with ONEX Compute nodes. It provides
validation hooks and enforces the deterministic computation contract.

Architecture:
    PluginComputeBase implements ProtocolPluginCompute and provides:
    - Abstract execute() method for plugin-specific logic
    - Optional validation hooks for input/output verification
    - Documentation and examples for plugin developers
    - Integration with ONEX 4-node architecture

Determinism Guarantees:
    All compute plugins extending this base class MUST guarantee:
    1. Same inputs → Same outputs (reproducibility)
    2. No external state dependencies
    3. No side effects (no I/O, no mutation)
    4. No randomness (unless seeded via input)
    5. No time dependencies (unless time provided as input)

What Plugins MUST NOT Do:
    ❌ Network operations (HTTP, gRPC, WebSocket)
    ❌ Filesystem operations (read, write, delete)
    ❌ Database operations (queries, transactions)
    ❌ Random number generation (non-deterministic)
    ❌ Current time access (non-deterministic)
    ❌ Mutable shared state (class variables, globals)
    ❌ External service calls (message buses, caches)
    ❌ Environment variable access (unless explicitly allowed)
    ❌ Process/thread management
    ❌ Signal handling or system calls

What Plugins CAN Do:
    ✅ Pure data transformations
    ✅ Mathematical computations
    ✅ String processing
    ✅ Data structure operations
    ✅ Validation and schema checking
    ✅ Deterministic hashing
    ✅ Deterministic randomness (with seed from input)

Integration with ONEX Compute Nodes:
    Compute plugins are designed to work with NodeComputeService:

    ```python
    from omnibase_infra.plugins import PluginComputeBase

    # Plugin implementation
    class MyComputePlugin(PluginComputeBase):
        def execute(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
            # Pure computation logic
            result = self._process(input_data, context)
            return {"result": result}

        def validate_input(self, input_data: dict[str, Any]) -> None:
            # Optional input validation
            if "required_field" not in input_data:
                raise ValueError("required_field missing")

    # Node integration
    plugin = MyComputePlugin()
    result = plugin.execute(
        input_data={"value": 42, "required_field": "present"},
        context={"operation": "process"}
    )
    ```

Thread Safety:
    Plugin implementations should be stateless and thread-safe:
    - No instance variables modified during execute()
    - All state passed through input_data or context
    - Immutable configuration in __init__() is acceptable

See Also:
    - src/omnibase_infra/protocols/protocol_plugin_compute.py for protocol definition
    - ONEX 4-node architecture documentation
    - OMN-813 for compute plugin design specification
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PluginComputeBase(ABC):
    """Abstract base class for compute plugins.

    Provides optional validation hooks and enforces the execute() contract.
    All compute plugins should inherit from this class to ensure consistency.

    Subclasses must implement execute() to perform deterministic computation.

    Example:
        ```python
        class MyComputePlugin(PluginComputeBase):
            def execute(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
                # Deterministic computation
                result = self._process(input_data)
                return {"result": result}

            def validate_input(self, input_data: dict[str, Any]) -> None:
                # Optional: Validate required fields
                if "required_field" not in input_data:
                    raise ValueError("Missing required_field")
        ```

    Note:
        The base class does NOT enforce thread safety. Plugins should be
        designed to be stateless or use immutable state only.
    """

    @abstractmethod
    def execute(
        self, input_data: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute computation. MUST be deterministic.

        Given the same input_data and context, this method MUST return
        the same result every time it is called.

        Args:
            input_data: The input data to process
            context: Execution context (correlation_id, timestamps, etc.)

        Returns:
            Computation result as dictionary

        Raises:
            ValueError: If input validation fails
            OnexError: For computation errors (with proper error chaining)

        Note:
            Implementations MUST NOT:
            - Access network
            - Access file system
            - Query databases
            - Use random numbers (unless seeded from context)
            - Use current time (unless passed in context)
            - Maintain mutable state between calls
        """
        ...

    def validate_input(self, input_data: dict[str, Any]) -> None:
        """Optional input validation hook.

        Override this method to validate input_data before execution.
        This is called automatically by the registry/executor before execute().

        Args:
            input_data: The input data to validate

        Raises:
            ValueError: If validation fails
        """
        return  # Default: no validation

    def validate_output(self, output: dict[str, Any]) -> None:
        """Optional output validation hook.

        Override this method to validate computation results after execution.
        This is called automatically by the registry/executor after execute().

        Args:
            output: The output data to validate

        Raises:
            ValueError: If validation fails
        """
        return  # Default: no validation
