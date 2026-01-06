# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared type aliases for omnibase_infra models.

This module re-exports core types from omnibase_core.types for convenience.

Note: JsonPrimitive and Any are imported from omnibase_core.types.
"""

from omnibase_core.types import Any, JsonPrimitive

__all__ = [
    "JsonPrimitive",
    "Any",
]
