"""Pydantic models for node registration and discovery.

This module provides validation models for the dynamic node registration system,
enabling type-safe node metadata, capabilities, and endpoint configuration.
"""

from datetime import datetime
from typing import Any, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omninode_bridge.schemas import NODE_REGISTRATION_METADATA_SCHEMA
from omninode_bridge.utils.metadata_validator import validate_metadata

from .database_models import EnumNodeType


class ModelNodeRegistration(BaseModel):
    """
    Node registration model for dynamic node discovery and orchestration.

    This model represents a registered node in the omninode ecosystem,
    containing metadata about its capabilities, endpoints, and health status.

    Attributes:
        id: Unique identifier for the registration record
        node_id: Unique identifier for the node (e.g., "metadata-stamping-service")
        node_type: ONEX node type classification (effect, compute, reducer, orchestrator)
        capabilities: Dictionary describing node capabilities and supported operations
        endpoints: Dictionary mapping operation names to endpoint URLs
        metadata: Additional node metadata (version, author, tags, etc.)
        health_endpoint: Optional health check endpoint URL
        last_heartbeat: Timestamp of last health check or heartbeat
        registered_at: Timestamp when node was first registered
        updated_at: Timestamp of last registration update

    Example:
        ```python
        registration = ModelNodeRegistration(
            node_id="metadata-stamping-v1",
            node_type="effect",
            capabilities={
                "operations": ["stamp", "validate", "hash"],
                "max_file_size": 10485760,
                "supported_formats": ["json", "yaml", "text"]
            },
            endpoints={
                "stamp": "http://metadata-service:8053/api/v1/stamp",
                "validate": "http://metadata-service:8053/api/v1/validate",
                "hash": "http://metadata-service:8053/api/v1/hash"
            },
            metadata={
                "version": "1.0.0",
                "author": "OmniNode Team",
                "tags": ["cryptography", "hashing", "verification"]
            },
            health_endpoint="http://metadata-service:8053/health"
        )
        ```
    """

    id: Optional[UUID] = Field(
        default=None, description="Unique identifier for the registration record"
    )
    node_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique identifier for the node",
        examples=["metadata-stamping-v1", "onextree-intelligence"],
    )
    node_type: Union[EnumNodeType, str] = Field(
        ...,
        description="ONEX node type classification",
        examples=[
            EnumNodeType.EFFECT,
            EnumNodeType.COMPUTE,
            EnumNodeType.REDUCER,
            EnumNodeType.ORCHESTRATOR,
        ],
    )
    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary describing node capabilities and supported operations",
        examples=[
            {
                "operations": ["stamp", "validate"],
                "max_file_size": 10485760,
                "supported_formats": ["json", "yaml"],
            }
        ],
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping operation names to endpoint URLs",
        examples=[
            {
                "stamp": "http://service:8053/api/v1/stamp",
                "validate": "http://service:8053/api/v1/validate",
            }
        ],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional node metadata",
        examples=[{"version": "1.0.0", "author": "OmniNode Team", "tags": ["hashing"]}],
    )
    health_endpoint: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional health check endpoint URL",
        examples=["http://service:8053/health"],
    )
    last_heartbeat: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last health check or heartbeat",
    )
    registered_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when node was first registered",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last registration update",
    )

    @field_validator("node_type", mode="before")
    @classmethod
    def validate_node_type(cls, v: Union[EnumNodeType, str]) -> EnumNodeType:
        """Validate node_type is one of the allowed ONEX types."""
        if isinstance(v, EnumNodeType):
            return v
        if isinstance(v, str):
            try:
                return EnumNodeType(v.lower())
            except ValueError:
                raise ValueError(
                    f"node_type must be one of {[t.value for t in EnumNodeType]}, got: {v}"
                )
        raise ValueError(f"node_type must be a string or EnumNodeType, got: {type(v)}")

    @field_validator("endpoints")
    @classmethod
    def validate_endpoints(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate endpoints dictionary has valid URLs."""
        if not v:
            return v

        for operation, url in v.items():
            if not url or not isinstance(url, str):
                raise ValueError(
                    f"Invalid endpoint URL for operation '{operation}': {url}"
                )
            if not url.startswith(("http://", "https://")):
                raise ValueError(
                    f"Endpoint URL must start with http:// or https:// for operation '{operation}': {url}"
                )
        return v

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate capabilities structure."""
        if not v:
            return v

        # Ensure capabilities is a dictionary
        if not isinstance(v, dict):
            raise ValueError("capabilities must be a dictionary")

        # Validate common capability fields if present
        if "operations" in v and not isinstance(v["operations"], list):
            raise ValueError("capabilities.operations must be a list")

        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata_field(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate metadata against JSON schema."""
        if v:
            validate_metadata(v, NODE_REGISTRATION_METADATA_SCHEMA)
        return v

    model_config = ConfigDict(
        from_attributes=True,  # Pydantic v2 syntax for ORM mode
        json_schema_extra={
            "example": {
                "node_id": "metadata-stamping-v1",
                "node_type": "effect",  # Will be converted to EnumNodeType.EFFECT
                "capabilities": {
                    "operations": ["stamp", "validate", "hash"],
                    "max_file_size": 10485760,
                    "supported_formats": ["json", "yaml", "text"],
                },
                "endpoints": {
                    "stamp": "http://metadata-service:8053/api/v1/stamp",
                    "validate": "http://metadata-service:8053/api/v1/validate",
                    "hash": "http://metadata-service:8053/api/v1/hash",
                },
                "metadata": {
                    "version": "1.0.0",
                    "author": "OmniNode Team",
                    "description": "Cryptographic metadata stamping service",
                    "tags": ["cryptography", "hashing", "verification"],
                },
                "health_endpoint": "http://metadata-service:8053/health",
            }
        },
    )


class ModelNodeRegistrationCreate(BaseModel):
    """
    Model for creating a new node registration.

    This is a simplified model for node registration requests,
    excluding auto-generated fields like id, registered_at, updated_at.
    """

    node_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique identifier for the node",
    )
    node_type: Union[EnumNodeType, str] = Field(
        ...,
        description="ONEX node type classification",
        examples=[
            EnumNodeType.EFFECT,
            EnumNodeType.COMPUTE,
            EnumNodeType.REDUCER,
            EnumNodeType.ORCHESTRATOR,
        ],
    )
    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        description="Node capabilities",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Operation endpoints",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    health_endpoint: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Health check endpoint",
    )

    @field_validator("node_type", mode="before")
    @classmethod
    def validate_node_type(cls, v: Union[EnumNodeType, str]) -> EnumNodeType:
        """Validate node_type is one of the allowed ONEX types."""
        if isinstance(v, EnumNodeType):
            return v
        if isinstance(v, str):
            try:
                return EnumNodeType(v.lower())
            except ValueError:
                raise ValueError(
                    f"node_type must be one of {[t.value for t in EnumNodeType]}, got: {v}"
                )
        raise ValueError(f"node_type must be a string or EnumNodeType, got: {type(v)}")


class ModelNodeRegistrationUpdate(BaseModel):
    """
    Model for updating an existing node registration.

    All fields are optional to allow partial updates.
    """

    capabilities: Optional[dict[str, Any]] = Field(
        default=None,
        description="Updated node capabilities",
    )
    endpoints: Optional[dict[str, str]] = Field(
        default=None,
        description="Updated operation endpoints",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Updated metadata",
    )
    health_endpoint: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Updated health endpoint",
    )
    last_heartbeat: Optional[datetime] = Field(
        default=None,
        description="Updated heartbeat timestamp",
    )
