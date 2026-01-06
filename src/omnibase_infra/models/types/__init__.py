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
# Using `object` as the underlying type per ONEX guidelines (Any is forbidden).
# This provides type safety while allowing JSON's dynamic nature.
# Note: `object` is the ONEX-preferred alternative to Any for generic payloads.

JsonPrimitive = object  # Represents atomic JSON values (str, int, float, bool, None)
JsonValue = object  # Represents any valid JSON value (primitive, dict, or list)

# JsonDict is a more specific type for functions that work with JSON objects.
# Use this when you need dict operations like .get(), indexing, or `in` checks.
# This is compatible with mypy while still using `object` for nested values.
JsonDict = dict[str, object]

__all__ = [
    "JsonDict",
    "JsonPrimitive",
    "JsonValue",
]
