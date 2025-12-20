# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Type aliases for Registry Effect Node protocol definitions.

Re-exports centralized JSON type definitions from omnibase_infra.models.types.
This module maintains backward compatibility while reducing type duplication.

Type Aliases:
    JsonPrimitive: Base JSON primitive types (str, int, float, bool, None)
    JsonValue: Recursive JSON value type including primitives, lists, and dicts
    EnvelopeDict: Dictionary type for operation envelopes
    ResultDict: Dictionary type for operation results

These types represent valid JSON-serializable data without using Any.
The structure is validated at runtime by envelope executor implementations.
For strongly-typed data exchange, prefer explicit Pydantic models over raw JSON.
"""

from omnibase_infra.models.types import (
    EnvelopeDict,
    JsonPrimitive,
    JsonValue,
    ResultDict,
)

__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "EnvelopeDict",
    "ResultDict",
]
