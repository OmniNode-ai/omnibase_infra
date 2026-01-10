# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared type aliases for omnibase_infra models.

This module re-exports JSON type aliases from omnibase_core.types and adds
infrastructure-specific type aliases.

Types from omnibase_core:
    - JsonPrimitive: Atomic JSON values (str, int, float, bool, None, UUID, datetime)
    - JsonType: Full recursive JSON type (now fixed via TypeAlias pattern)

Local types:
    - JsonDict: ``dict[str, object]`` for JSON object operations
    - JsonValue: ``object`` alias for generic JSON-like data in function signatures

JsonType Recursion Fix (OMN-1274) - Type Selection Guide:
    The original ``JsonType`` recursive type alias caused Pydantic 2.x recursion
    errors at class definition time. This has been fixed in omnibase_core v0.6.3+
    using the TypeAlias pattern. Here is how to choose the right type:

    **Use JsonType** (from ``omnibase_core.types``) when:
    - The field can be ANY valid JSON value (object, array, string, number, etc.)
    - Example: HTTP response body, generic JSON payloads
    - ``from omnibase_core.types import JsonType``

    **Use dict[str, object]** when:
    - The field is specifically a dictionary with string keys
    - Example: Error details, database rows, configuration dicts
    - Provides correct semantics without recursion

    **Use object** when:
    - In function signatures for generic payloads
    - NEVER use ``Any`` - per ONEX guidelines, use ``object`` instead
    - ``def process(data: object) -> object:``

    **Technical Background:**
    The original ``JsonType`` was defined as::

        JsonType = dict[str, "JsonType"] | list["JsonType"] | str | int | float | bool | None

    Pydantic 2.x performs eager schema generation at class definition time,
    causing infinite recursion when expanding this type alias. The fix in
    omnibase_core v0.6.3 uses ``TypeAlias`` to prevent this::

        JsonType: TypeAlias = "dict[str, JsonType] | list[JsonType] | str | ..."

See Also:
    - ADR: ``docs/decisions/adr-any-type-pydantic-workaround.md`` (historical)
    - Pydantic issue: https://github.com/pydantic/pydantic/issues/3278
"""

from omnibase_core.types import JsonPrimitive, JsonType

# JsonDict is a more specific type for functions that work with JSON objects.
# Use this when you need dict operations like .get(), indexing, or `in` checks.
# Preferred over JsonType when the value is known to be a dictionary.
JsonDict = dict[str, object]

# JsonValue is an alias for generic JSON-like data in function signatures.
# Use this instead of Any per ONEX guidelines - object is the proper "unknown type"
# marker in Python's type system.
# For Pydantic model fields requiring full JSON support, use JsonType from
# omnibase_core.types instead.
JsonValue = object

__all__ = [
    "JsonDict",
    "JsonPrimitive",
    "JsonType",
    "JsonValue",
]
