# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared type aliases for omnibase_infra models.

This module re-exports JSON type aliases from omnibase_core.types and adds
infrastructure-specific type aliases.

Types from omnibase_core:
    - JsonPrimitive: Atomic JSON values (str, int, float, bool, None, UUID, datetime)
    - JsonType: Any valid JSON value (recursive union)

Local types:
    - JsonDict: dict[str, object] for JSON object operations
    - JsonValue: Alias for JsonType (backwards compatibility)
"""

from omnibase_core.types import JsonPrimitive, JsonType

# JsonDict is a more specific type for functions that work with JSON objects.
# Use this when you need dict operations like .get(), indexing, or `in` checks.
JsonDict = dict[str, object]

# Backwards compatibility alias
JsonValue = JsonType

__all__ = [
    "JsonDict",
    "JsonPrimitive",
    "JsonType",
    "JsonValue",
]
