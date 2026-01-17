# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration cache entry model.

.. versionadded:: 0.8.0
    Initial implementation for OMN-765.

This module provides the ModelConfigCacheEntry for internal cache entries
in the BindingConfigResolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.runtime.models.model_binding_config import ModelBindingConfig


@dataclass
class ModelConfigCacheEntry:
    """Internal cache entry for resolved configurations.

    Attributes:
        config: The resolved binding configuration.
        expires_at: When this cache entry expires.
        source: Description of the configuration source (for debugging).
    """

    config: ModelBindingConfig
    expires_at: datetime
    source: str

    def is_expired(self) -> bool:
        """Check if this cache entry has expired.

        Returns:
            True if expired, False otherwise.
        """
        return datetime.now(UTC) > self.expires_at


__all__: list[str] = ["ModelConfigCacheEntry"]
