# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""JSON type aliases for type-safe JSON handling.

These type aliases define JSON-serializable value types used throughout
omnibase_infra for protocol boundaries, error details, and metadata.

Type Aliases:
    JsonPrimitive: Base JSON primitive types (str, int, float, bool, None)
    JsonValue: Recursive JSON value type including primitives, lists, and dicts
    EnvelopeDict: Dictionary type for operation envelopes
    ResultDict: Dictionary type for operation results

These types represent valid JSON-serializable data without using Any.
The structure is validated at runtime by implementing components.
For strongly-typed data exchange, prefer explicit Pydantic models over raw JSON.

Note: Uses PEP 695 type statement syntax for proper Pydantic recursive type support.

Design Rationale:
    Centralizing these type definitions reduces union count across the codebase
    and ensures consistent JSON type handling. Previously, these types were
    duplicated in multiple locations (protocol_types.py, model_dispatch_result.py).
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
