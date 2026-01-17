# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration cache entry model.

.. versionadded:: 0.8.0
    Initial implementation for OMN-765.

This module provides the ModelConfigCacheEntry for internal cache entries
in the BindingConfigResolver.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_binding_config import ModelBindingConfig


class ModelConfigCacheEntry(BaseModel):
    """Internal cache entry for resolved configurations.

    Attributes:
        config: The resolved binding configuration.
        expires_at: When this cache entry expires.
        source: Description of the configuration source (for debugging).
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    config: ModelBindingConfig = Field(
        ...,
        description="The resolved binding configuration.",
    )

    expires_at: datetime = Field(
        ...,
        description="When this cache entry expires.",
    )

    source: str = Field(
        ...,
        description="Description of the configuration source (for debugging).",
    )

    def is_expired(self) -> bool:
        """Check if this cache entry has expired.

        Returns:
            True if expired, False otherwise.
        """
        return datetime.now(UTC) > self.expires_at


__all__: list[str] = ["ModelConfigCacheEntry"]
