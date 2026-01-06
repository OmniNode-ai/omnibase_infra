# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared type aliases for omnibase_infra models.

This module re-exports JSON type aliases from omnibase_core.types and adds
infrastructure-specific type aliases.

Types from omnibase_core:
    - JsonPrimitive: Atomic JSON values (str, int, float, bool, None, UUID, datetime)

Local types:
    - JsonDict: dict[str, object] for JSON object operations

NOTE: JsonType from omnibase_core causes Pydantic 2.x recursion issues.
Use `Any` from typing as a workaround. See ADR: adr-any-type-pydantic-workaround.md
"""

from typing import Any

from omnibase_core.types import JsonPrimitive

# JsonDict is a more specific type for functions that work with JSON objects.
# Use this when you need dict operations like .get(), indexing, or `in` checks.
JsonDict = dict[str, object]

# Backwards compatibility alias - use Any due to JsonType recursion issues
JsonValue = Any

__all__ = [
    "Any",
    "JsonDict",
    "JsonPrimitive",
    "JsonValue",
]
