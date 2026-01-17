# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Record Model for Storage Operations.

This module provides ModelRegistrationRecord, representing a complete node
registration record for storage in backend systems (e.g., PostgreSQL).

Architecture:
    ModelRegistrationRecord captures all information about a registered node:
    - Identity: node_id, node_type, node_version
    - Capabilities: Tuple of capabilities the node provides (immutable)
    - Endpoints: Service discovery endpoints
    - Metadata: Additional key-value metadata
    - Timestamps: created_at, updated_at for tracking

    This model is backend-agnostic and can be serialized to any storage format.

Security:
    The metadata field should NOT contain sensitive information.
    Secrets should be stored in Vault, not in registration records.

Related:
    - NodeRegistrationStorageEffect: Effect node that stores these records
    - ProtocolRegistrationStorageHandler: Protocol for storage backends
    - ModelStorageQuery: Query model for retrieving records
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.utils import validate_timezone_aware_datetime_optional


class ModelRegistrationRecord(BaseModel):
    """Registration record for node storage operations.

    Represents a complete node registration record that can be stored in
    any backend (PostgreSQL, etc.). This model is capability-oriented,
    focusing on what the node does rather than implementation details.

    Immutability:
        This model uses frozen=True to ensure records are immutable
        once created, supporting safe concurrent access and comparison.

    Note on correlation_id:
        The correlation_id field is optional and is used for distributed
        tracing. When not provided, handlers should auto-generate one
        for observability. This design choice allows callers to either:
        1. Pass an existing correlation_id for trace continuity
        2. Omit it and let the handler generate one

        The auto-generation approach ensures every operation has traceability
        even when callers don't provide explicit IDs.

    Attributes:
        node_id: Unique identifier for the registered node.
        node_type: Type of ONEX node (EnumNodeKind).
        node_version: Semantic version of the node.
        capabilities: Tuple of capability names the node provides (immutable).
        endpoints: Dict mapping endpoint type to URL.
        metadata: Additional key-value metadata (no secrets).
        created_at: Timestamp when the record was created (optional, backend may set).
        updated_at: Timestamp when the record was last updated (optional, backend may set).
        correlation_id: Optional correlation ID for distributed tracing.

    Example:
        >>> from datetime import UTC, datetime
        >>> from uuid import uuid4
        >>> from omnibase_core.enums.enum_node_kind import EnumNodeKind
        >>> from omnibase_core.models.primitives.model_semver import ModelSemVer
        >>> record = ModelRegistrationRecord(
        ...     node_id=uuid4(),
        ...     node_type=EnumNodeKind.EFFECT,
        ...     node_version=ModelSemVer(major=1, minor=0, patch=0),
        ...     capabilities=("registration.storage", "registration.storage.query"),
        ...     endpoints={"health": "http://localhost:8080/health"},
        ...     metadata={"team": "platform"},
        ...     created_at=datetime.now(UTC),
        ...     updated_at=datetime.now(UTC),
        ...     correlation_id=uuid4(),
        ... )
        >>> record.node_type
        <EnumNodeKind.EFFECT: 'effect'>
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: UUID = Field(
        ...,
        description="Unique identifier for the registered node",
    )
    node_type: EnumNodeKind = Field(
        ...,
        description="Type of ONEX node",
    )
    node_version: ModelSemVer = Field(
        ...,
        description="Semantic version of the node",
    )
    capabilities: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tuple of capability names the node provides (immutable)",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Dict mapping endpoint type to URL (e.g., {'health': 'http://...'})",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional key-value metadata (no secrets)",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Timestamp when the record was created (timezone-aware if provided, backend may auto-set)",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Timestamp when the record was last updated (timezone-aware if provided, backend may auto-set)",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing (auto-generated if not provided)",
    )

    @field_validator("capabilities", mode="before")
    @classmethod
    def _coerce_capabilities_to_tuple(cls, v: object) -> tuple[str, ...]:
        """Convert list/sequence to tuple for immutability.

        This validator ensures explicit handling of all input types rather than
        silent fallback, which could mask invalid input. In strict mode, all
        items must already be strings - no silent type coercion.

        Args:
            v: The input value to coerce.

        Returns:
            tuple[str, ...]: The coerced tuple of capability strings.

        Raises:
            ValueError: If input is not a valid sequence type or contains non-strings.

        Type coercion rules:
            - tuple: Validates contents are strings, returns as-is
            - list/set/frozenset: Validates contents are strings, converts to tuple
            - Non-string items: Raises ValueError (strict mode)
            - None: Raises ValueError (use default_factory=tuple instead)
            - Other types: Raises ValueError
        """
        if isinstance(v, tuple):
            # Validate tuple contents in strict mode
            for i, item in enumerate(v):
                if not isinstance(item, str):
                    raise ValueError(
                        f"capabilities[{i}] must be a string, got {type(item).__name__}"
                    )
            return v
        if isinstance(v, list | set | frozenset):
            # Convert sequence to tuple, validating contents
            result: list[str] = []
            for i, item in enumerate(v):
                if not isinstance(item, str):
                    raise ValueError(
                        f"capabilities[{i}] must be a string, got {type(item).__name__}"
                    )
                result.append(item)
            return tuple(result)
        if isinstance(v, Sequence) and not isinstance(v, str):
            # Handle other sequence types
            result_seq: list[str] = []
            for i, item in enumerate(v):
                if not isinstance(item, str):
                    raise ValueError(
                        f"capabilities[{i}] must be a string, got {type(item).__name__}"
                    )
                result_seq.append(item)
            return tuple(result_seq)
        raise ValueError(
            f"capabilities must be a tuple, list, set, or frozenset, got {type(v).__name__}"
        )

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamp_timezone_aware(cls, v: datetime | None) -> datetime | None:
        """Validate that timestamps are timezone-aware when provided.

        Delegates to shared utility for consistent validation across all models.
        """
        return validate_timezone_aware_datetime_optional(v)


__all__ = ["ModelRegistrationRecord"]
