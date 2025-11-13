"""
Transformer framework for O.N.E. v0.1 protocol.

This module provides the base transformer pattern with schema validation
and execution context for typed transformations.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Module-level schema registry cache
_module_schema_registry = None


def _get_schema_registry():
    """Get schema registry instance with multiple fallback strategies."""
    global _module_schema_registry

    # Return cached if available
    if _module_schema_registry is not None:
        return _module_schema_registry

    # Try multiple import strategies
    try:
        # Strategy 1: Relative import
        from .schema_registry import schema_registry

        _module_schema_registry = schema_registry
        return schema_registry
    except ImportError:
        pass

    try:
        # Strategy 2: Absolute import
        from omninode_bridge.services.metadata_stamping.execution.schema_registry import (
            schema_registry,
        )

        _module_schema_registry = schema_registry
        return schema_registry
    except ImportError:
        pass

    try:
        # Strategy 3: Module-level search
        import sys

        # Look for schema_registry in any loaded module
        for module_name, module in sys.modules.items():
            if hasattr(module, "schema_registry") and module_name.endswith(
                "schema_registry"
            ):
                _module_schema_registry = module.schema_registry
                return module.schema_registry
    except (AttributeError, KeyError, ImportError):
        # Module search failed due to missing attributes, module access errors, or import issues
        pass

    return None


class ExecutionContext(BaseModel):
    """Execution context for transformers."""

    execution_id: str
    input_schema: str
    output_schema: str
    simulation_mode: bool = False
    budget_limit: Optional[float] = None
    metadata: dict[str, Any] = {}
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BaseTransformer(ABC):
    """
    Base transformer for schema-first execution.

    Provides typed input/output with automatic validation.
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize base transformer.

        Args:
            name: Transformer name
            version: Transformer version
        """
        self.name = name
        self.version = version
        self.input_schema: Optional[type[BaseModel]] = None
        self.output_schema: Optional[type[BaseModel]] = None
        self.execution_count = 0
        self.total_execution_time_ms = 0.0

    @abstractmethod
    async def execute(
        self, input_data: BaseModel, context: ExecutionContext
    ) -> BaseModel:
        """
        Execute transformer with typed input/output.

        Args:
            input_data: Validated input data
            context: Execution context

        Returns:
            BaseModel: Validated output data
        """
        pass

    def set_schemas(
        self, input_schema: type[BaseModel], output_schema: type[BaseModel]
    ):
        """
        Set input and output schemas for validation.

        Args:
            input_schema: Input schema class
            output_schema: Output schema class
        """
        self.input_schema = input_schema
        self.output_schema = output_schema

        # Register schemas with the schema registry
        try:
            schema_registry = _get_schema_registry()

            if schema_registry is None:
                raise ImportError("Schema registry not available")

            # Register input schema
            input_schema_name = f"{self.name}_input"
            success1 = schema_registry.register_schema(
                input_schema_name,
                input_schema,
                self.version,
                {"transformer": self.name, "schema_type": "input"},
            )

            # Register output schema
            output_schema_name = f"{self.name}_output"
            success2 = schema_registry.register_schema(
                output_schema_name,
                output_schema,
                self.version,
                {"transformer": self.name, "schema_type": "output"},
            )

            if success1 and success2:
                logger.debug(
                    f"Registered schemas for {self.name}: {input_schema.__name__} -> {output_schema.__name__}"
                )
            else:
                logger.warning(f"Some schema registrations failed for {self.name}")

        except ImportError:
            logger.warning(f"Schema registry not available for {self.name}")
        except Exception as e:
            logger.warning(f"Failed to register schemas for {self.name}: {e}")

        logger.debug(
            f"Schemas set for {self.name}: {input_schema.__name__} -> {output_schema.__name__}"
        )

    async def validate_input(self, data: Any) -> BaseModel:
        """
        Validate input against schema.

        Args:
            data: Input data to validate

        Returns:
            BaseModel: Validated input model

        Raises:
            ValueError: If validation fails
        """
        if not self.input_schema:
            raise ValueError(f"No input schema defined for {self.name}")

        try:
            if isinstance(data, dict):
                validated = self.input_schema(**data)
            elif isinstance(data, self.input_schema):
                validated = data
            else:
                validated = self.input_schema.model_validate(data)

            logger.debug(f"Input validation successful for {self.name}")
            return validated

        except ValidationError as e:
            logger.error(f"Input validation failed for {self.name}: {e}")
            raise ValueError(f"Input validation failed: {e}")

    async def validate_output(self, data: Any) -> BaseModel:
        """
        Validate output against schema.

        Args:
            data: Output data to validate

        Returns:
            BaseModel: Validated output model

        Raises:
            ValueError: If validation fails
        """
        if not self.output_schema:
            raise ValueError(f"No output schema defined for {self.name}")

        try:
            if isinstance(data, dict):
                validated = self.output_schema(**data)
            elif isinstance(data, self.output_schema):
                validated = data
            else:
                validated = self.output_schema.model_validate(data)

            logger.debug(f"Output validation successful for {self.name}")
            return validated

        except ValidationError as e:
            logger.error(f"Output validation failed for {self.name}: {e}")
            raise ValueError(f"Output validation failed: {e}")

    async def execute_with_validation(
        self, input_data: Any, context: ExecutionContext
    ) -> BaseModel:
        """
        Execute with automatic validation.

        Args:
            input_data: Input data
            context: Execution context

        Returns:
            BaseModel: Validated output

        Raises:
            ValueError: If validation fails
        """
        import time

        start_time = time.perf_counter()
        context.started_at = datetime.now(UTC)

        try:
            # Validate input
            validated_input = await self.validate_input(input_data)

            # Execute transformer
            result = await self.execute(validated_input, context)

            # Validate output
            validated_output = await self.validate_output(result)

            # Update metrics
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self.execution_count += 1
            self.total_execution_time_ms += execution_time_ms

            context.completed_at = datetime.now(UTC)
            context.metadata["execution_time_ms"] = execution_time_ms

            logger.info(
                f"Transformer {self.name} executed successfully in {execution_time_ms:.2f}ms"
            )

            return validated_output

        except Exception as e:
            context.completed_at = datetime.now(UTC)
            context.metadata["error"] = str(e)
            logger.error(f"Transformer {self.name} execution failed: {e}")
            raise

    def get_metrics(self) -> dict[str, Any]:
        """
        Get transformer execution metrics.

        Returns:
            dict: Execution metrics
        """
        avg_time = (
            self.total_execution_time_ms / self.execution_count
            if self.execution_count > 0
            else 0.0
        )

        return {
            "name": self.name,
            "version": self.version,
            "execution_count": self.execution_count,
            "total_execution_time_ms": self.total_execution_time_ms,
            "average_execution_time_ms": avg_time,
            "input_schema": self.input_schema.__name__ if self.input_schema else None,
            "output_schema": (
                self.output_schema.__name__ if self.output_schema else None
            ),
        }


# Global transformer registry
_transformer_registry: dict[str, BaseTransformer] = {}


def transformer(
    input_schema: type[BaseModel],
    output_schema: type[BaseModel],
    name: Optional[str] = None,
    version: str = "1.0.0",
):
    """
    Decorator to create a transformer from a function.

    Args:
        input_schema: Input schema class
        output_schema: Output schema class
        name: Transformer name (defaults to function name)
        version: Transformer version

    Returns:
        BaseTransformer: Transformer instance
    """

    def decorator(func: Callable) -> BaseTransformer:
        transformer_name = name or func.__name__

        class FunctionTransformer(BaseTransformer):
            """Transformer created from decorated function."""

            def __init__(self):
                super().__init__(transformer_name, version)
                self.set_schemas(input_schema, output_schema)
                self.func = func

            async def execute(
                self, input_data: BaseModel, context: ExecutionContext
            ) -> BaseModel:
                """Execute the decorated function."""
                # Execute function
                if inspect.iscoroutinefunction(self.func):
                    result = await self.func(input_data, context)
                else:
                    result = self.func(input_data, context)

                # Return result (validation happens in execute_with_validation)
                return result

        # Create transformer instance
        transformer_instance = FunctionTransformer()

        # Register transformer
        _transformer_registry[transformer_name] = transformer_instance
        logger.info(f"Registered transformer: {transformer_name} v{version}")

        # Return the transformer instance (not the function)
        return transformer_instance

    return decorator


def get_transformer(name: str) -> Optional[BaseTransformer]:
    """
    Get transformer by name.

    Args:
        name: Transformer name

    Returns:
        BaseTransformer: Transformer instance or None
    """
    return _transformer_registry.get(name)


def list_transformers() -> dict[str, BaseTransformer]:
    """
    List all registered transformers.

    Returns:
        dict: Map of transformer names to instances
    """
    return _transformer_registry.copy()


def unregister_transformer(name: str) -> bool:
    """
    Unregister a transformer.

    Args:
        name: Transformer name

    Returns:
        bool: True if unregistered successfully
    """
    if name in _transformer_registry:
        del _transformer_registry[name]
        logger.info(f"Unregistered transformer: {name}")
        return True
    return False
