# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared type aliases for omnibase_infra models.

This module re-exports JSON type aliases from omnibase_core.types and adds
infrastructure-specific type aliases.

Types from omnibase_core:
    - JsonPrimitive: Atomic JSON values (str, int, float, bool, None, UUID, datetime)

Local types:
    - JsonDict: dict[str, object] for JSON object operations
    - JsonValue: object alias for generic JSON-like data in function signatures

NOTE: JsonType from omnibase_core causes Pydantic 2.x recursion issues.
For Pydantic model fields, use `Any` with the required NOTE: comment.
For function signatures, use `object`. See ADR: adr-any-type-pydantic-workaround.md
"""

from omnibase_core.types import JsonPrimitive

# JsonDict is a more specific type for functions that work with JSON objects.
# Use this when you need dict operations like .get(), indexing, or `in` checks.
JsonDict = dict[str, object]

# ONEX: Using object instead of Any per ADR guidelines.
# JsonValue is an alias for generic JSON-like data in function signatures.
# For Pydantic model fields requiring JSON serialization, use `Any` directly
# with the required NOTE: comment (see ADR: adr-any-type-pydantic-workaround.md)
JsonValue = object

__all__ = [
    "JsonDict",
    "JsonPrimitive",
    "JsonValue",
]
