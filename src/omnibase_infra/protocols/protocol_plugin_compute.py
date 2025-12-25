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
        def execute(self, input_data: PluginInputData, context: PluginContext) -> PluginOutputData:
            '''Execute deterministic computation.'''
            ...

    # Example plugin implementation
    class JsonSchemaValidator:
        def execute(self, input_data: PluginInputData, context: PluginContext) -> PluginOutputData:
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

        def _validate_schema(self, data: dict[str, object], schema: dict[str, object]) -> bool:
            # Pure computation - deterministic validation
            ...

        def _get_errors(self, data: dict[str, object], schema: dict[str, object]) -> list[dict[str, object]]:
            # Pure computation - deterministic error extraction
            ...
    ```

Protocol Verification:
    Per ONEX conventions, protocol compliance is verified via duck typing rather
    than isinstance checks. Verify the required method exists and is callable:

    ```python
    plugin = JsonSchemaValidator()

    # Duck typing verification (preferred)
    assert hasattr(plugin, 'execute') and callable(plugin.execute)

    # Or use hasattr with getattr for cleaner pattern
    execute_method = getattr(plugin, 'execute', None)
    assert execute_method is not None and callable(execute_method)
    ```

See Also:
    - src/omnibase_infra/plugins/plugin_compute_base.py for base implementation
    - ONEX 4-node architecture documentation
    - OMN-813 for complete compute plugin design
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

__all__ = [
    "PluginInputData",
    "PluginContext",
    "PluginOutputData",
    "ProtocolPluginCompute",
]


class PluginInputData(TypedDict, total=False):
    """Base structure for plugin input data.

    This TypedDict defines the expected structure for input_data parameter.
    Plugins may extend or specialize this structure based on their requirements.

    Attributes:
        All fields are optional by default (total=False).
        Concrete plugins should document their required fields.

    Example:
        ```python
        class JsonNormalizerInput(TypedDict):
            json: dict[str, object]  # Required field for JSON normalizer

        # Runtime validation in validate_input()
        def validate_input(self, input_data: PluginInputData) -> None:
            if "json" not in input_data:
                raise ValueError("Missing required field: json")
        ```

    Note:
        This is a base type hint. Runtime validation should be performed
        in the validate_input() method to ensure type safety.
    """


class PluginContext(TypedDict, total=False):
    """Base structure for plugin execution context.

    This TypedDict defines the expected structure for context parameter.
    Common context fields include correlation IDs, timestamps, and configuration.

    Common Fields:
        correlation_id: UUID for distributed tracing
        execution_timestamp: When execution started (deterministic if provided)
        plugin_config: Plugin-specific configuration parameters
        metadata: Additional metadata for observability

    Example:
        ```python
        context: PluginContext = {
            "correlation_id": uuid4(),
            "execution_timestamp": "2025-01-15T12:00:00Z",
            "plugin_config": {"max_depth": 10},
            "metadata": {"source": "api_gateway"},
        }
        ```

    Note:
        All fields are optional (total=False). Plugins should document
        which context fields they require in their validate_input() method.
    """


class PluginOutputData(TypedDict, total=False):
    """Base structure for plugin output data.

    This TypedDict defines the expected structure for return values.
    Plugins may extend or specialize this structure based on their outputs.

    Common Fields:
        result: Primary computation result
        metadata: Output metadata (execution time, version, etc.)
        errors: List of validation or computation errors
        warnings: List of non-fatal warnings

    Example:
        ```python
        output: PluginOutputData = {
            "result": {"normalized": {...}},
            "metadata": {
                "execution_time_ms": 15,
                "plugin_version": "1.0.0",
            },
            "errors": [],
            "warnings": [],
        }
        ```

    Note:
        This is a base type hint. Concrete plugins should define their
        own output structure TypedDict for stronger type safety.
    """


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
            def execute(self, input_data: PluginInputData, context: PluginContext) -> PluginOutputData:
                '''Normalize numeric data to [0, 1] range.'''
                values: list[float] = input_data.get("values", [])
                min_val: float = context.get("min_value", 0.0)
                max_val: float = context.get("max_value", 1.0)

                # Pure computation - deterministic normalization
                normalized: list[float] = [
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
        self, input_data: PluginInputData, context: PluginContext
    ) -> PluginOutputData:
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
            OnexError: For all computation failures (with proper error chaining).
                Plugin implementations must convert all exceptions to OnexError.
            ValueError: For invalid input_data or context (should be caught and wrapped).
            TypeError: For incorrect input types (should be caught and wrapped).

        Error Handling:
            All plugin implementations MUST follow ONEX error handling standards:

            1. **OnexError Chaining**: Convert all exceptions to OnexError with proper chaining
               using `raise OnexError(...) from original_exception`.

            2. **Correlation ID Propagation**: Always include correlation_id from context
               in error details for distributed tracing.

            3. **Never Suppress Errors**: All errors must be logged, handled, or escalated.
               Silent failures are strictly prohibited.

            4. **Context Preservation**: Maintain full error context for debugging.

            Example - Input Validation Error:
                ```python
                def execute(self, input_data: PluginInputData, context: PluginContext) -> PluginOutputData:
                    from omnibase_core.errors import OnexError
                    from omnibase_core.enums import CoreErrorCode

                    correlation_id = context.get("correlation_id", "unknown")

                    try:
                        # Validate required fields
                        if "required_field" not in input_data:
                            raise ValueError("Missing required field: required_field")

                        # Perform computation
                        result = self._compute(input_data)
                        return {"result": result}

                    except ValueError as e:
                        raise OnexError(
                            message=f"Invalid input data: {e}",
                            error_code=CoreErrorCode.INVALID_INPUT,
                            correlation_id=correlation_id,
                            plugin_name=self.__class__.__name__,
                        ) from e
                ```

            Example - Computation Error with Context:
                ```python
                def execute(self, input_data: PluginInputData, context: PluginContext) -> PluginOutputData:
                    from omnibase_core.errors import OnexError
                    from omnibase_core.enums import CoreErrorCode

                    correlation_id = context.get("correlation_id", "unknown")

                    try:
                        # Perform complex computation
                        values: list[float] = input_data.get("values", [])
                        result: float = sum(values) / len(values)  # May raise ZeroDivisionError

                        return {
                            "average": result,
                            "count": len(values),
                            "correlation_id": correlation_id,
                        }

                    except ZeroDivisionError as e:
                        raise OnexError(
                            message="Cannot compute average: empty values list",
                            error_code=CoreErrorCode.INVALID_INPUT,
                            correlation_id=correlation_id,
                            plugin_name=self.__class__.__name__,
                            input_size=len(input_data.get("values", [])),
                        ) from e

                    except Exception as e:
                        # Catch-all for unexpected errors
                        raise OnexError(
                            message=f"Computation failed: {e}",
                            error_code=CoreErrorCode.INTERNAL_ERROR,
                            correlation_id=correlation_id,
                            plugin_name=self.__class__.__name__,
                        ) from e
                ```

            Example - Type Validation Error:
                ```python
                def execute(self, input_data: PluginInputData, context: PluginContext) -> PluginOutputData:
                    from omnibase_core.errors import OnexError
                    from omnibase_core.enums import CoreErrorCode

                    correlation_id = context.get("correlation_id", "unknown")

                    try:
                        values = input_data.get("values")

                        # Validate type
                        if not isinstance(values, list):
                            raise TypeError(f"Expected list for 'values', got {type(values).__name__}")

                        # Compute result
                        return {"processed": [v * 2 for v in values]}

                    except TypeError as e:
                        raise OnexError(
                            message=f"Type validation failed: {e}",
                            error_code=CoreErrorCode.INVALID_INPUT,
                            correlation_id=correlation_id,
                            plugin_name=self.__class__.__name__,
                            expected_type="list",
                            actual_type=type(input_data.get("values")).__name__,
                        ) from e
                ```

        Error Recovery Patterns:
            Plugins should implement graceful degradation where appropriate:

            Example - Fallback Values:
                ```python
                def execute(self, input_data: PluginInputData, context: PluginContext) -> PluginOutputData:
                    from omnibase_core.errors import OnexError
                    from omnibase_core.enums import CoreErrorCode

                    correlation_id = context.get("correlation_id", "unknown")

                    try:
                        # Attempt primary computation
                        result = self._complex_computation(input_data)

                        return {
                            "result": result,
                            "fallback_used": False,
                            "correlation_id": correlation_id,
                        }

                    except Exception as e:
                        # Try fallback strategy
                        try:
                            fallback_result = self._simple_fallback(input_data)

                            return {
                                "result": fallback_result,
                                "fallback_used": True,
                                "warning": f"Primary computation failed: {e}",
                                "correlation_id": correlation_id,
                            }

                        except Exception as fallback_error:
                            # Both strategies failed - raise with full context
                            raise OnexError(
                                message="Both primary and fallback computations failed",
                                error_code=CoreErrorCode.INTERNAL_ERROR,
                                correlation_id=correlation_id,
                                plugin_name=self.__class__.__name__,
                                primary_error=str(e),
                                fallback_error=str(fallback_error),
                            ) from fallback_error
                ```

        Notes:
            - Must be deterministic (same inputs → same outputs)
            - Must not perform I/O operations
            - Must not access external state
            - Must not modify input_data or context
            - Should validate inputs before processing
            - Should include relevant metadata in output
            - MUST include correlation_id in all error contexts
            - MUST use proper OnexError chaining for all exceptions

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
