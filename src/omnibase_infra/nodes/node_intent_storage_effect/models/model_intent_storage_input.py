# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Intent Storage Input Model for Intent Storage Operations.

This module provides ModelIntentStorageInput, representing the input
for storing a classified intent in the graph database.

Architecture:
    ModelIntentStorageInput is constructed from IntentClassifiedEvent payloads
    and contains:
    - intent_type: The classified intent type
    - session_id: Optional session identifier for grouping
    - payload: Intent-specific data to store as node properties
    - correlation_id: Required correlation ID for tracing (fail-fast validation)

    This model serves as the canonical input for the intent.store operation.

Related:
    - NodeIntentStorageEffect: Effect node that processes this input
    - ModelIntentStorageOutput: Output model containing storage result
    - HandlerIntent: Handler that executes storage operations
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.errors import OnexError
from omnibase_core.types import JsonType


class ModelIntentStorageInput(BaseModel):
    """Input model for intent storage operations.

    Defines the required fields for storing a classified intent as a
    graph node in Memgraph.

    Immutability:
        This model uses frozen=True to ensure inputs are immutable
        once created, enabling safe reuse and caching.

    Attributes:
        intent_type: The classified intent type (e.g., "query", "action").
        session_id: Optional session identifier for grouping related intents.
        payload: Intent-specific data to store as node properties.
        correlation_id: Required correlation ID for request tracing.

    Example:
        >>> from uuid import uuid4
        >>> input_model = ModelIntentStorageInput(
        ...     intent_type="query",
        ...     session_id="sess-12345",
        ...     payload={"query": "What is ONEX?", "confidence": 0.95},
        ...     correlation_id=uuid4(),
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: str = Field(
        ...,
        description="The classified intent type",
        min_length=1,
        max_length=256,
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for grouping related intents",
        max_length=256,
    )
    payload: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Intent-specific data to store as node properties",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for request tracing (required for observability)",
    )

    @field_validator("payload")
    @classmethod
    def validate_no_reserved_keys(cls, v: dict[str, JsonType]) -> dict[str, JsonType]:
        """Validate payload does not contain reserved keys.

        Reserved keys (intent_type, session_id, correlation_id) are used as
        structured fields and cannot appear in the payload to prevent conflicts
        during storage property merging.

        Args:
            v: The payload dictionary to validate.

        Returns:
            The validated payload dictionary.

        Raises:
            OnexError: If payload contains any reserved keys.
        """
        reserved_keys = {"intent_type", "session_id", "correlation_id"}
        conflicting = reserved_keys & v.keys()
        if conflicting:
            raise OnexError(
                message=f"Payload cannot contain reserved keys: {sorted(conflicting)}",
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
            )
        return v

    def has_session(self) -> bool:
        """Check if this input has a session identifier.

        Returns:
            True if session_id is provided.
        """
        return self.session_id is not None

    def to_storage_properties(self) -> dict[str, JsonType]:
        """Convert input to storage-friendly properties dict.

        Returns:
            Dictionary containing all intent properties for graph storage,
            including intent_type, correlation_id, and session_id merged with payload.

        Note:
            Reserved key validation (intent_type, session_id, correlation_id)
            is performed at model construction time via field_validator,
            so this method can safely merge payload with structured fields.
        """
        properties: dict[str, JsonType] = {
            "intent_type": self.intent_type,
            "correlation_id": str(self.correlation_id),
        }
        if self.session_id:
            properties["session_id"] = self.session_id

        properties.update(self.payload)
        return properties


__all__ = ["ModelIntentStorageInput"]
