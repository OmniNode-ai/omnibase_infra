# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pydantic model for plugin execution context.

This module provides the ModelPluginContext Pydantic BaseModel that replaces
the former PluginContext TypedDict definition.

Design Notes:
    - Uses ConfigDict(extra="allow") to support arbitrary fields
    - Supports dict-like access via __getitem__ for flexible API usage
    - Can be instantiated from dicts using model_validate()
    - Follows ONEX naming convention: Model<Name>
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.mixins import MixinDictLikeAccessors


class ModelPluginContext(MixinDictLikeAccessors, BaseModel):
    """Base Pydantic model for plugin execution context.

    This model replaces PluginContext TypedDict and provides structured
    context for plugin execution including correlation IDs and metadata.

    Common Fields:
        correlation_id: UUID for distributed tracing
        execution_timestamp: When execution started (ISO format string)
        plugin_config: Plugin-specific configuration parameters
        metadata: Additional metadata for observability
        random_seed: Seed for deterministic randomness (optional)

    Configuration:
        - extra="allow": Accepts arbitrary additional fields
        - frozen=False: Allows mutation (though plugins should not mutate)
        - populate_by_name=True: Allows field access by alias

    Example:
        ```python
        context = ModelPluginContext(
            correlation_id="test-uuid-123",
            execution_timestamp="2025-01-15T12:00:00Z",
            plugin_config={"max_depth": 10},
        )

        # Access with get (dict-like)
        corr_id = context.get("correlation_id", "unknown")
        ```

    Note:
        All fields are optional (extra="allow"). Plugins should document
        which context fields they require.
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=False,
        populate_by_name=True,
        from_attributes=True,  # pytest-xdist compatibility
    )

    # Common context fields - all optional with defaults
    # Note: execution_timestamp uses str for ISO format timestamps
    # Integer epoch timestamps should be converted to ISO strings
    # Using empty string/dict defaults to reduce union count while maintaining optionality
    correlation_id: str = ""
    execution_timestamp: str = ""
    random_seed: int | None = None  # Must stay nullable - 0 is a valid seed
    plugin_config: dict[str, JsonType] = Field(default_factory=dict)
    metadata: dict[str, JsonType] = Field(default_factory=dict)


__all__: list[str] = ["ModelPluginContext"]
