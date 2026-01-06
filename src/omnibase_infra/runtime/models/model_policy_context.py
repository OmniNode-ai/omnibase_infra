# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pydantic model for policy evaluation context.

This module provides the ModelPolicyContext Pydantic BaseModel for passing
structured context to policy evaluation methods.

Design Notes:
    - Uses ConfigDict(extra="allow") to support arbitrary policy-specific fields
    - Supports dict-like access via __getitem__ for backwards compatibility
    - Can be instantiated from dicts using model_validate()
    - Follows ONEX naming convention: Model<Name>

This model replaces the former JsonType usage in ProtocolPolicy.evaluate()
and ProtocolPolicy.decide() method signatures, providing type safety while
maintaining flexibility for diverse policy contexts.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPolicyContext(BaseModel):
    """Base Pydantic model for policy evaluation context.

    This model provides structured context for policy evaluation including
    correlation IDs, attempt counts, and policy-specific parameters.

    Common Fields:
        correlation_id: UUID for distributed tracing
        attempt: Current retry/evaluation attempt number
        timestamp_ms: Evaluation timestamp in milliseconds
        metadata: Additional metadata for observability

    Configuration:
        - extra="allow": Accepts arbitrary additional fields for policy-specific data
        - frozen=False: Allows mutation (though policies should not mutate context)
        - populate_by_name=True: Allows field access by alias

    Example:
        ```python
        from uuid import uuid4

        # Create with known fields
        context = ModelPolicyContext(
            correlation_id=uuid4(),
            attempt=3,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0,
        )

        # Access with get (dict-like, backwards compatible)
        attempt = context.get("attempt", 0)

        # Access as attribute
        corr_id = context.correlation_id

        # Create from dict (backwards compatible with JsonType usage)
        context = ModelPolicyContext.model_validate({"attempt": 3, "error_type": "timeout"})
        ```

    Policy-Specific Fields:
        For orchestrator policies:
            - attempt: Current retry attempt number
            - error_type: Type of error that triggered evaluation
            - elapsed_ms: Time elapsed since operation started
            - base_delay_seconds: Base delay for backoff calculations
            - max_delay_seconds: Maximum delay cap

        For reducer policies:
            - current_state: Current aggregated state
            - event: Event to process
            - timestamp: Event timestamp
            - sequence_number: Event sequence for ordering

    Note:
        All fields are optional due to extra="allow". Specific policies should
        document which context fields they require for proper evaluation.
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=False,
        populate_by_name=True,
        from_attributes=True,
    )

    # Common context fields - all optional with defaults
    correlation_id: UUID | None = None
    attempt: int = 0
    timestamp_ms: int = 0
    metadata: dict[str, object] = Field(default_factory=dict)

    def get(self, key: str, default: object = None) -> object:
        """Get field value by key with optional default.

        Provides dict-like access for backwards compatibility with
        JsonType usage patterns in policy implementations.

        Args:
            key: Field name to retrieve
            default: Default value if field not found

        Returns:
            Field value or default

        Example:
            >>> context = ModelPolicyContext(attempt=3)
            >>> context.get("attempt", 0)
            3
            >>> context.get("missing_key", "default")
            'default'
        """
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> object:
        """Get field value by key using bracket notation.

        Provides dict-like access for backwards compatibility.

        Args:
            key: Field name to retrieve

        Returns:
            Field value

        Raises:
            KeyError: If field does not exist

        Example:
            >>> context = ModelPolicyContext(attempt=3)
            >>> context["attempt"]
            3
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Check if field exists in model.

        Provides dict-like membership testing.

        Args:
            key: Field name to check

        Returns:
            True if field exists and is not None

        Example:
            >>> context = ModelPolicyContext(attempt=3)
            >>> "attempt" in context
            True
            >>> "missing" in context
            False
        """
        return hasattr(self, key) and getattr(self, key) is not None


__all__: list[str] = ["ModelPolicyContext"]
