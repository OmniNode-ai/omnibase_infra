"""JSON Schema definitions for JSONB metadata fields.

This module provides JSON Schema definitions for validating metadata fields
in database entities. Each schema validates the structure of JSONB data to
prevent malformed or malicious data from being inserted into the database.

The schemas are permissive (additionalProperties: True) to allow extensibility
while enforcing validation for known/common fields.
"""

from typing import Any

# JSON Schema for workflow_executions.metadata
WORKFLOW_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "workflow_type": {
            "type": "string",
            "description": "Type classification for the workflow",
        },
        "priority": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "Workflow priority level (1-10)",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tags for categorization and filtering",
        },
        "user_id": {
            "type": "string",
            "description": "User ID who initiated the workflow",
        },
        "steps_completed": {
            "type": "integer",
            "minimum": 0,
            "description": "Number of workflow steps completed",
        },
        "custom_data": {
            "type": "object",
            "description": "Custom workflow-specific data",
        },
    },
    "additionalProperties": True,  # Allow extensibility
}

# JSON Schema for node_registrations.metadata
NODE_REGISTRATION_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "environment": {
            "type": "string",
            "enum": ["dev", "staging", "prod"],
            "description": "Deployment environment",
        },
        "region": {
            "type": "string",
            "description": "Geographic region or availability zone",
        },
        "deployment_id": {
            "type": "string",
            "description": "Unique deployment identifier",
        },
        "version": {
            "type": "string",
            "pattern": r"^\d+\.\d+\.\d+$",
            "description": "Semantic version (x.y.z)",
        },
        "author": {
            "type": "string",
            "description": "Author or team name",
        },
        "description": {
            "type": "string",
            "description": "Human-readable node description",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tags for categorization",
        },
        "resource_limits": {
            "type": "object",
            "properties": {
                "cpu_cores": {"type": "number", "minimum": 0},
                "memory_mb": {"type": "integer", "minimum": 0},
                "max_connections": {"type": "integer", "minimum": 0},
            },
            "description": "Resource limits for the node",
        },
    },
    "additionalProperties": True,  # Allow extensibility
}

# JSON Schema for bridge_states.aggregation_metadata
BRIDGE_STATE_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "aggregation_window": {
            "type": "string",
            "description": "Aggregation time window specification",
        },
        "window_size_ms": {
            "type": "integer",
            "minimum": 0,
            "description": "Aggregation window size in milliseconds",
        },
        "batch_size": {
            "type": "integer",
            "minimum": 1,
            "description": "Number of items per batch",
        },
        "last_window_items": {
            "type": "integer",
            "minimum": 0,
            "description": "Items processed in last window",
        },
        "state_version": {
            "type": "integer",
            "minimum": 0,
            "description": "State version for conflict resolution",
        },
        "aggregation_type": {
            "type": "string",
            "enum": [
                "NAMESPACE_GROUPING",
                "TIME_WINDOW",
                "FILE_TYPE_GROUPING",
                "SIZE_BUCKETS",
                "WORKFLOW_GROUPING",
                "CUSTOM",
            ],
            "description": "Type of aggregation being performed",
        },
    },
    "additionalProperties": True,  # Allow extensibility
}
