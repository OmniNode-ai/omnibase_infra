"""Shared test fixtures and helpers for all node tests.

This conftest.py provides common fixtures and helper functions that are
automatically available to all tests in node subdirectories.
"""

from typing import Any
from uuid import UUID, uuid4

from omnibase_core.enums.enum_node_type import EnumNodeType
from omnibase_core.models.contracts import (
    ModelContractEffect,
    ModelContractReducer,
    ModelIOOperationConfig,
)
from omnibase_core.primitives import ModelSemVer


def create_test_contract_effect(
    name: str = "test_effect",
    version: str = "1.0.0",
    description: str = "Test effect node",
    node_type: EnumNodeType = EnumNodeType.EFFECT,
    correlation_id: UUID | None = None,
    input_state: dict[str, Any] | None = None,
    output_state: dict[str, Any] | None = None,
) -> ModelContractEffect:
    """
    Create a properly initialized ModelContractEffect for testing.

    Args:
        name: Node name
        version: Semantic version string (e.g., "1.0.0")
        description: Node description
        node_type: Node type enum
        correlation_id: Optional correlation ID (generates UUID if None)
        input_state: Optional input state dict
        output_state: Optional output state dict

    Returns:
        Fully initialized ModelContractEffect instance
    """
    # Parse version string into major.minor.patch
    parts = version.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    return ModelContractEffect(
        name=name,
        version=ModelSemVer(major=major, minor=minor, patch=patch),
        description=description,
        node_type=node_type,
        input_model="dict[str, Any]",
        output_model="dict[str, Any]",
        io_operations=[
            ModelIOOperationConfig(
                operation_type="read",
                resource_type="memory",
                required=False,
            )
        ],
        correlation_id=correlation_id or uuid4(),
        input_state=input_state or {},
        output_state=output_state or {},
    )


def create_test_contract_reducer(
    name: str = "test_reducer",
    version: str = "1.0.0",
    description: str = "Test reducer node",
    correlation_id: UUID | None = None,
    input_state: dict[str, Any] | None = None,
    output_state: dict[str, Any] | None = None,
) -> ModelContractReducer:
    """
    Create a properly initialized ModelContractReducer for testing.

    Args:
        name: Node name
        version: Semantic version string (e.g., "1.0.0")
        description: Node description
        correlation_id: Optional correlation ID (generates UUID if None)
        input_state: Optional input state dict
        output_state: Optional output state dict

    Returns:
        Fully initialized ModelContractReducer instance
    """
    # Parse version string into major.minor.patch
    parts = version.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    return ModelContractReducer(
        name=name,
        version=ModelSemVer(major=major, minor=minor, patch=patch),
        description=description,
        input_model="dict[str, Any]",
        output_model="dict[str, Any]",
        correlation_id=correlation_id or uuid4(),
        input_state=input_state or {},
        output_state=output_state or {},
    )
