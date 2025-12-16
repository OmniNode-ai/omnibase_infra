# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definition for deterministic compute plugins.

This module defines the ProtocolPluginCompute interface for plugins that perform
pure data transformations without side effects. This protocol is designed for use
with ONEX Compute nodes that require deterministic, reproducible operations.

Protocol Contract:
    Compute plugins MUST guarantee deterministic behavior:
    - Same inputs ALWAYS produce the same outputs
    - No dependency on external state or randomness
    - No side effects (no I/O, no state mutation)
    - No access to current time unless explicitly provided as input

What Plugins MUST NOT Do:
    ❌ Network operations (HTTP requests, API calls, socket connections)
    ❌ Filesystem operations (file read/write, directory access)
    ❌ Database operations (queries, transactions, connection pooling)
    ❌ Random number generation (unless deterministic with provided seed)
    ❌ Current time access (unless time is passed as input parameter)
    ❌ Mutable shared state (global variables, class-level state)
    ❌ External service calls (Kafka, Redis, Consul, Vault)
    ❌ Environment variable access (unless explicitly allowed)
    ❌ Process/thread creation or management
    ❌ Signal handling or system calls

What Plugins CAN Do:
    ✅ Pure data transformations (mapping, filtering, aggregation)
    ✅ Mathematical computations (arithmetic, statistics, algorithms)
    ✅ String processing (parsing, formatting, validation)
    ✅ Data structure operations (sorting, searching, grouping)
    ✅ Validation and schema checking (Pydantic models, type checking)
    ✅ Deterministic hashing (with consistent input ordering)
    ✅ Deterministic randomness (with seed from input_data or context)

Integration with ONEX Compute Nodes:
    Compute nodes follow the 4-node architecture pattern:
    - EFFECT: External I/O (database, network, filesystem)
    - COMPUTE: Pure transformations (THIS PROTOCOL)
    - REDUCER: State aggregation and consolidation
    - ORCHESTRATOR: Workflow coordination

    Compute plugins integrate with NodeComputeService to provide deterministic
    processing capabilities. The node is responsible for I/O and state management,
    while plugins focus purely on data transformation logic.

Example Usage:
    ```python
    from omnibase_infra.protocols import ProtocolPluginCompute
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class ProtocolPluginCompute(Protocol):
        def execute(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
            '''Execute deterministic computation.'''
            ...

    # Example plugin implementation
    class JsonSchemaValidator:
        def execute(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
            '''Validate JSON data against schema.'''
            schema = context.get("schema", {})
            data = input_data.get("data", {})

            # Pure validation logic (no I/O)
            is_valid = self._validate_schema(data, schema)

            return {
                "valid": is_valid,
                "data": data,
                "errors": [] if is_valid else self._get_errors(data, schema),
            }

        def _validate_schema(self, data: dict, schema: dict) -> bool:
            # Pure computation - deterministic validation
            ...

        def _get_errors(self, data: dict, schema: dict) -> list:
            # Pure computation - deterministic error extraction
            ...
    ```

Type Checking:
    This protocol uses @runtime_checkable to enable isinstance() checks at runtime:

    ```python
    plugin = JsonSchemaValidator()
    assert isinstance(plugin, ProtocolPluginCompute)  # True
    ```

See Also:
    - src/omnibase_infra/plugins/plugin_compute_base.py for base implementation
    - ONEX 4-node architecture documentation
    - OMN-813 for complete compute plugin design
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProtocolPluginCompute(Protocol):
    """Protocol for deterministic compute plugins.

    This protocol defines the interface for plugins that perform pure data
    transformations without side effects. Implementations must guarantee
    deterministic behavior where the same inputs always produce the same outputs.

    Methods:
        execute: Perform deterministic computation on input data.

    Thread Safety:
        Plugin implementations should be thread-safe and stateless. All required
        state should be passed through input_data or context parameters.

    Performance:
        Plugins should be optimized for performance as they may be called
        frequently in high-throughput data processing pipelines.

    Example:
        ```python
        class DataNormalizer:
            def execute(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
                '''Normalize numeric data to [0, 1] range.'''
                values = input_data.get("values", [])
                min_val = context.get("min_value", 0.0)
                max_val = context.get("max_value", 1.0)

                # Pure computation - deterministic normalization
                normalized = [
                    (v - min_val) / (max_val - min_val)
                    for v in values
                ]

                return {
                    "normalized_values": normalized,
                    "min": min_val,
                    "max": max_val,
                }
        ```
    """

    def execute(
        self, input_data: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute deterministic computation on input data.

        This method must be deterministic: given the same input_data and context,
        it must always produce the same output. No side effects are allowed.

        Args:
            input_data: Input data dictionary containing values to process.
                Structure depends on plugin implementation.
            context: Context dictionary providing configuration and metadata.
                Should contain any external parameters needed for computation.

        Returns:
            Result dictionary containing computed output values.
            Structure depends on plugin implementation.

        Raises:
            ValueError: If input_data or context are invalid.
            TypeError: If input types are incorrect.
            Any plugin-specific exceptions for computation errors.

        Notes:
            - Must be deterministic (same inputs → same outputs)
            - Must not perform I/O operations
            - Must not access external state
            - Must not modify input_data or context
            - Should validate inputs before processing
            - Should include relevant metadata in output

        Example:
            ```python
            # Input
            input_data = {"values": [1, 2, 3, 4, 5]}
            context = {"operation": "sum"}

            # Execution
            result = plugin.execute(input_data, context)

            # Output
            {
                "result": 15,
                "operation": "sum",
                "count": 5
            }
            ```
        """
        ...
