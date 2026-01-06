# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared type aliases for omnibase_infra models.

This module defines JSON type aliases for use throughout omnibase_infra.
These types were previously imported from omnibase_core.types but are now
defined locally after omnibase_core 0.6.2 removed the types module.

Note on Pydantic Compatibility:
    The recursive type definition `JsonValue = ... | dict[str, "JsonValue"] | list["JsonValue"]`
    causes infinite recursion in Pydantic's schema generation. To maintain Pydantic
    compatibility, we use `object` for nested values which still provides type safety
    at runtime while avoiding the recursion issue.

    For TYPE_CHECKING blocks (static analysis only), the full recursive definition
    could be used, but for runtime Pydantic models, this simplified version works.
"""

from __future__ import annotations

# JSON type definitions
# JsonValue and JsonPrimitive represent JSON-compatible values.
# Using typing.Any as the underlying type because JSON values can be any of:
# str, int, float, bool, None, dict, list (recursive structure).
# This matches the original omnibase_core.types definition and is required
# for compatibility with existing code that uses .get(), indexing, .items(), etc.
# Note: ONEX normally discourages Any, but this is a semantic type alias for
# "any JSON-compatible value" which is fundamentally dynamic.
from typing import Any

JsonPrimitive = Any  # Represents atomic JSON values (str, int, float, bool, None)
JsonValue = Any  # Represents any valid JSON value (primitive, dict, or list)

__all__ = [
    "JsonPrimitive",
    "JsonValue",
]
