# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Type aliases for Registry Effect Node protocol definitions.

These type aliases define JSON-serializable value types and envelope/result
dictionary structures used by protocol methods.

Type Aliases:
    JsonPrimitive: Base JSON primitive types (str, int, float, bool, None)
    JsonValue: Recursive JSON value type including primitives, lists, and dicts
    EnvelopeDict: Dictionary type for operation envelopes
    ResultDict: Dictionary type for operation results

These types represent valid JSON-serializable data without using Any.
The structure is validated at runtime by envelope executor implementations.
For strongly-typed data exchange, prefer explicit Pydantic models over raw JSON.

Note: Uses PEP 695 type statement syntax for proper Pydantic recursive type support.
"""

# JSON-serializable types for protocol boundaries using PEP 695 syntax.
# These provide type safety while allowing arbitrary JSON structures.
type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]

# Envelope dictionary types for protocol methods.
# Values are JSON-serializable, validated at runtime by executor implementations.
type EnvelopeDict = dict[str, JsonValue]
type ResultDict = dict[str, JsonValue]

__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "EnvelopeDict",
    "ResultDict",
]
