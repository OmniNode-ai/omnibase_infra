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

ONEX 4-Node Architecture Integration:
    Compute plugins belong exclusively to the COMPUTE layer of ONEX architecture.
    This separation ensures clear responsibilities and maintainable code.

    Architecture Overview:
        - EFFECT (NodeEffectService):
          * External I/O (database, network, filesystem, message bus)
          * Service integrations (Kafka, Consul, Vault, Redis, PostgreSQL)
          * Connection pooling and circuit breaker patterns
          * Examples: postgres_adapter, consul_adapter, kafka_adapter

        - COMPUTE (NodeComputeService):
          * Pure data transformations (THIS IS WHERE PLUGINS LIVE)
          * Deterministic algorithms and business logic
          * Stateless computation without side effects
          * Examples: data validation, JSON normalization, aggregation

        - REDUCER (NodeReducerService):
          * State aggregation from multiple sources
          * Event sourcing and state reconstruction
          * Multi-source data consolidation
          * Examples: infrastructure_reducer, state_aggregator

        - ORCHESTRATOR (NodeOrchestratorService):
          * Workflow coordination across multiple nodes
          * Multi-step process management
          * Service orchestration patterns
          * Examples: infrastructure_orchestrator, workflow_coordinator

    Why Plugins Belong in COMPUTE:
        1. Determinism: COMPUTE layer requires reproducible outputs
        2. No Side Effects: EFFECT layer handles all I/O operations
        3. Testability: Pure functions are trivially testable
        4. Composability: Plugins combine without coordination complexity
        5. Scalability: Stateless computation enables horizontal scaling

    Integration with NodeComputeService:
        ```python
        from omnibase_infra.plugins import PluginComputeBase
        from omnibase_infra.protocols.protocol_plugin_compute import (
            ModelPluginInputData,
            ModelPluginContext,
            ModelPluginOutputData,
        )

        # Step 1: Implement plugin (pure computation)
        class DataValidatorPlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # Pure validation logic (no I/O)
                is_valid = self._validate_schema(input_data)
                return {
                    "valid": is_valid,
                    "errors": [] if is_valid else self._get_errors(input_data),
                }

            def validate_input(self, input_data: ModelPluginInputData) -> None:
                if "schema" not in input_data:
                    raise ValueError("schema field required")

        # Step 2: Integrate with NodeComputeService (I/O wrapper)
        class ValidationNode(NodeComputeService):
            def __init__(self, container: ONEXContainer, plugin: ProtocolPluginCompute):
                super().__init__(container)
                # Plugin is injected via constructor (sync-friendly)
                # Resolution from container happens at initialization time:
                #   plugin = await container.service_registry.resolve_service(
                #       ProtocolPluginCompute
                #   )
                #   node = ValidationNode(container, plugin)
                self.plugin = plugin

            async def execute(self, input_model: ModelInput) -> ModelOutput:
                # Node handles I/O and state management
                input_data = input_model.model_dump()
                context = {"correlation_id": input_model.correlation_id}

                # Plugin performs pure computation
                result = self.plugin.execute(input_data, context)

                # Node handles output persistence (if needed)
                return ModelOutput(**result)

        # Complete initialization example:
        async def create_validation_node(container: ONEXContainer) -> ValidationNode:
            '''Factory function for creating ValidationNode with resolved plugin.'''
            plugin = await container.service_registry.resolve_service(
                ProtocolPluginCompute
            )
            return ValidationNode(container, plugin)
        ```

    When to Use Plugin vs Direct Node Implementation:
        Use PluginComputeBase (COMPUTE layer):
        ✅ Pure data transformations (JSON normalization, data validation)
        ✅ Deterministic algorithms (sorting, filtering, aggregation)
        ✅ Business logic without external dependencies
        ✅ Reusable computation across multiple nodes
        ✅ Pluggable behavior that may vary by deployment

        Use Direct Node Implementation:
        ❌ EFFECT: Operations requiring I/O or external service calls
        ❌ REDUCER: State aggregation requiring multiple data sources
        ❌ ORCHESTRATOR: Workflow coordination across multiple nodes
        ❌ Complex lifecycle (connection pooling, circuit breakers)
        ❌ Infrastructure service integration (Kafka, Consul, Vault)

    Architectural Benefits:
        - Clear separation of concerns (I/O vs computation)
        - Simplified testing (mock I/O, test computation independently)
        - Enhanced reusability (plugins shared across nodes)
        - Improved scalability (stateless plugins scale horizontally)
        - Better maintainability (computation changes don't affect I/O layer)

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

# Import directly from model submodules to avoid circular import
# (protocol_plugin_compute imports these same models, creating a cycle)
from omnibase_infra.plugins.models.model_plugin_context import ModelPluginContext
from omnibase_infra.plugins.models.model_plugin_input_data import ModelPluginInputData
from omnibase_infra.plugins.models.model_plugin_output_data import ModelPluginOutputData


class PluginComputeBase(ABC):
    """Abstract base class for compute plugins.

    Provides optional validation hooks and enforces the execute() contract.
    All compute plugins should inherit from this class to ensure consistency.

    Subclasses must implement execute() to perform deterministic computation.

    Thread Safety:
        The base class does NOT enforce thread safety. Plugins MUST be designed
        to be stateless or use immutable state only.

        Safe Patterns:
        - ✅ No instance variables modified during execute()
        - ✅ Configuration set in __init__() and never modified
        - ✅ All state passed through input_data or context
        - ✅ Use function-local variables only

        Unsafe Patterns:
        - ❌ self.counter += 1
        - ❌ self.results.append(item)
        - ❌ self.cache[key] = value

    Edge Cases to Handle:
        Plugins extending this base class should handle these scenarios:

        1. **Empty Inputs**: Handle {} and [] gracefully
        2. **None Values**: Treat None as empty/default or validate and raise error
        3. **Missing Keys**: Use .get() with defaults or validate required fields
        4. **Type Validation**: Check types before processing
        5. **Large Inputs**: Consider memory limits for inputs >10MB
        6. **Deep Nesting**: Limit recursion depth to prevent stack overflow
        7. **Special Float Values**: Handle NaN and Infinity explicitly
        8. **Unicode**: Handle UTF-8 strings and control characters
        9. **Circular References**: Track visited objects in recursive algorithms
        10. **Locale Independence**: Do not rely on system locale

    Common Pitfalls:
        - Using mutable default arguments: `def foo(x=[])`
        - Assuming dict key ordering (Python <3.7)
        - Not handling division by zero
        - Not validating input types before processing
        - Modifying input_data or context dictionaries
        - Using non-deterministic operations (time.time(), random())

    Memory Considerations:
        For inputs >10MB:
        - Use streaming/iterators instead of loading all data
        - Release intermediate results promptly
        - Consider using generators for large outputs
        - Monitor memory growth in production

    Example:
        ```python
        from omnibase_infra.protocols.protocol_plugin_compute import (
            ModelPluginInputData,
            ModelPluginContext,
            ModelPluginOutputData,
        )

        class MyComputePlugin(PluginComputeBase):
            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # Handle edge cases
                if not input_data:
                    return {"result": None, "warning": "Empty input"}

                # Validate required fields
                if "required_field" not in input_data:
                    raise ValueError("Missing required_field")

                # Validate types
                value = input_data.get("required_field")
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Expected numeric value, got {type(value).__name__}")

                # Deterministic computation
                result = self._process(input_data)
                return {"result": result}

            def validate_input(self, input_data: ModelPluginInputData) -> None:
                # Optional: Validate required fields upfront
                if "required_field" not in input_data:
                    raise ValueError("Missing required_field")

                # Validate types
                if not isinstance(input_data["required_field"], (int, float)):
                    raise TypeError("required_field must be numeric")
        ```
    """

    __slots__ = ()  # Enforce statelessness - no instance attributes

    @abstractmethod
    def execute(
        self, input_data: ModelPluginInputData, context: ModelPluginContext
    ) -> ModelPluginOutputData:
        """Execute computation. MUST be deterministic.

        Given the same input_data and context, this method MUST return
        the same result every time it is called.

        Args:
            input_data: The input data to process
            context: Execution context (correlation_id, timestamps, etc.)

        Returns:
            Computation result as dictionary

        Raises:
            OnexError: For all computation failures (with proper error chaining)
            ValueError: If input validation fails (should be wrapped in OnexError)
            TypeError: If input types are incorrect (should be wrapped in OnexError)

        Error Handling Requirements:
            All implementations MUST follow ONEX error handling standards:

            1. **OnexError Chaining**: Convert all exceptions to OnexError
               ```python
               from omnibase_core.errors import OnexError
               from omnibase_core.enums import CoreErrorCode

               try:
                   result = self._compute(input_data)
               except Exception as e:
                   raise OnexError(
                       message=f"Computation failed: {e}",
                       error_code=CoreErrorCode.INTERNAL_ERROR,
                       correlation_id=context.get("correlation_id", "unknown"),
                       plugin_name=self.__class__.__name__,
                   ) from e
               ```

            2. **Correlation ID Propagation**: Always extract and propagate correlation_id
               ```python
               correlation_id = context.get("correlation_id", "unknown")
               # Include in all OnexError instances and output
               ```

            3. **Never Suppress Errors**: All exceptions must be converted to OnexError
               ```python
               # NEVER do this:
               except Exception:
                   pass  # ❌ Silent failure prohibited

               # ALWAYS do this:
               except Exception as e:
                   raise OnexError(...) from e  # ✅ Proper error chaining
               ```

            4. **Context Preservation**: Include debugging context in OnexError
               ```python
               raise OnexError(
                   message="Validation failed",
                   error_code=CoreErrorCode.INVALID_INPUT,
                   correlation_id=correlation_id,
                   plugin_name=self.__class__.__name__,
                   input_keys=list(input_data.keys()),  # Additional context
                   expected_type="list",
                   actual_type=type(value).__name__,
               ) from e
               ```

        Common Error Patterns:
            See ProtocolPluginCompute.execute() documentation for detailed examples:
            - Input validation errors (missing fields, invalid types)
            - Computation errors (ZeroDivisionError, overflow, underflow)
            - Type validation errors (expected vs actual types)
            - Fallback strategies with graceful degradation

        Edge Cases to Handle:
            1. **Empty inputs**: `input_data == {}` or `input_data.get("key") == []`
            2. **None values**: `input_data is None` or `context is None`
            3. **Missing keys**: Use `.get()` with defaults or validate upfront
            4. **Type mismatches**: Validate types before processing
            5. **Division by zero**: Check denominators before division
            6. **NaN/Infinity**: Use `math.isnan()` and `math.isinf()` checks
            7. **Deep nesting**: Limit recursion depth (e.g., max_depth=100)
            8. **Large inputs**: Monitor memory for inputs >10MB
            9. **Unicode strings**: Handle UTF-8 and control characters
            10. **Circular references**: Track visited objects with `set()`

        What NOT to Do:
            Implementations MUST NOT:
            - ❌ Access network (HTTP, gRPC, WebSocket)
            - ❌ Access file system (read, write, delete)
            - ❌ Query databases (SQL, NoSQL)
            - ❌ Use random numbers (unless seeded from context)
            - ❌ Use current time (unless passed in context)
            - ❌ Maintain mutable state between calls
            - ❌ Modify input_data or context dictionaries
            - ❌ Use global variables or class-level mutable state

        Example - Handling Edge Cases:
            ```python
            from omnibase_infra.protocols.protocol_plugin_compute import (
                ModelPluginInputData,
                ModelPluginContext,
                ModelPluginOutputData,
            )

            def execute(
                self, input_data: ModelPluginInputData, context: ModelPluginContext
            ) -> ModelPluginOutputData:
                # Edge Case 1 & 2: Handle None/empty inputs
                if not input_data:
                    return {"result": None, "warning": "Empty input"}

                # Edge Case 3: Handle missing keys
                values = input_data.get("values", [])
                if not values:
                    return {"result": 0, "count": 0}

                # Edge Case 4: Validate types
                if not all(isinstance(v, (int, float)) for v in values):
                    raise TypeError("All values must be numeric")

                # Edge Case 6: Handle NaN/Infinity
                import math
                clean_values = [v for v in values if not math.isnan(v) and not math.isinf(v)]

                # Edge Case 5: Division by zero check
                count = len(clean_values)
                if count == 0:
                    return {"result": 0, "warning": "All values were NaN/Inf"}

                # Safe computation
                total = sum(clean_values)
                average = total / count  # Safe: count > 0

                return {
                    "result": average,
                    "count": count,
                    "filtered": len(values) - count,
                }
            ```
        """
        ...

    def validate_input(self, input_data: ModelPluginInputData) -> None:
        """Optional input validation hook.

        Override this method to validate input_data before execution.
        This is called automatically by the registry/executor before execute().

        Args:
            input_data: The input data to validate

        Raises:
            ValueError: If validation fails
        """
        return  # Default: no validation

    def validate_output(self, output: ModelPluginOutputData) -> None:
        """Optional output validation hook.

        Override this method to validate computation results after execution.
        This is called automatically by the registry/executor after execute().

        Args:
            output: The output data to validate

        Raises:
            ValueError: If validation fails
        """
        return  # Default: no validation
