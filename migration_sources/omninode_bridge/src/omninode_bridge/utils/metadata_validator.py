"""Metadata validation utilities for JSONB fields.

This module provides validation functions for metadata dictionaries using JSON Schema.
It ensures that JSONB metadata fields conform to expected structures and prevents
malformed or malicious data from being inserted into the database.
"""

from typing import Any

import jsonschema
from jsonschema import ValidationError


def validate_metadata(metadata: dict[str, Any], schema: dict[str, Any]) -> None:
    """
    Validate metadata against a JSON schema.

    Args:
        metadata: Metadata dictionary to validate
        schema: JSON schema to validate against

    Raises:
        ValueError: If validation fails with clear error message

    Example:
        ```python
        from omninode_bridge.schemas import WORKFLOW_METADATA_SCHEMA
        from omninode_bridge.utils.metadata_validator import validate_metadata

        # Valid metadata
        metadata = {"priority": 5, "tags": ["urgent", "api"]}
        validate_metadata(metadata, WORKFLOW_METADATA_SCHEMA)  # No error

        # Invalid metadata
        metadata = {"priority": 15}  # Priority > 10
        validate_metadata(metadata, WORKFLOW_METADATA_SCHEMA)  # Raises ValueError
        ```
    """
    try:
        jsonschema.validate(instance=metadata, schema=schema)
    except ValidationError as e:
        # Extract meaningful error information
        error_path = ".".join(str(p) for p in e.path) if e.path else "root"
        error_msg = (
            f"Metadata validation failed at '{error_path}': {e.message}\n"
            f"Failed value: {e.instance}\n"
            f"Schema constraint: {e.validator}={e.validator_value}"
        )
        raise ValueError(error_msg) from e
    except jsonschema.SchemaError as e:
        # Schema itself is malformed (should not happen in production)
        raise ValueError(f"Invalid JSON schema provided: {e.message}") from e


def validate_workflow_metadata(metadata: dict[str, Any]) -> None:
    """
    Validate workflow execution metadata.

    Convenience function for validating workflow_executions.metadata fields.

    Args:
        metadata: Workflow metadata dictionary to validate

    Raises:
        ValueError: If validation fails

    Example:
        ```python
        from omninode_bridge.utils.metadata_validator import validate_workflow_metadata

        metadata = {
            "workflow_type": "metadata_stamping",
            "priority": 8,
            "tags": ["api", "production"],
            "user_id": "user-123"
        }
        validate_workflow_metadata(metadata)
        ```
    """
    from omninode_bridge.schemas import WORKFLOW_METADATA_SCHEMA

    validate_metadata(metadata, WORKFLOW_METADATA_SCHEMA)


def validate_node_registration_metadata(metadata: dict[str, Any]) -> None:
    """
    Validate node registration metadata.

    Convenience function for validating node_registrations.metadata fields.

    Args:
        metadata: Node registration metadata dictionary to validate

    Raises:
        ValueError: If validation fails

    Example:
        ```python
        from omninode_bridge.utils.metadata_validator import (
            validate_node_registration_metadata
        )

        metadata = {
            "environment": "prod",
            "region": "us-west-2",
            "version": "1.0.0",
            "tags": ["cryptography", "hashing"]
        }
        validate_node_registration_metadata(metadata)
        ```
    """
    from omninode_bridge.schemas import NODE_REGISTRATION_METADATA_SCHEMA

    validate_metadata(metadata, NODE_REGISTRATION_METADATA_SCHEMA)


def validate_bridge_state_metadata(metadata: dict[str, Any]) -> None:
    """
    Validate bridge state aggregation metadata.

    Convenience function for validating bridge_states.aggregation_metadata fields.

    Args:
        metadata: Bridge state metadata dictionary to validate

    Raises:
        ValueError: If validation fails

    Example:
        ```python
        from omninode_bridge.utils.metadata_validator import (
            validate_bridge_state_metadata
        )

        metadata = {
            "window_size_ms": 5000,
            "batch_size": 100,
            "aggregation_type": "NAMESPACE_GROUPING"
        }
        validate_bridge_state_metadata(metadata)
        ```
    """
    from omninode_bridge.schemas import BRIDGE_STATE_METADATA_SCHEMA

    validate_metadata(metadata, BRIDGE_STATE_METADATA_SCHEMA)
