# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mixin providing dict-like accessors for Pydantic models.

This mixin provides `get`, `__getitem__`, and `__contains__` methods
that enable dict-like access patterns for Pydantic BaseModel subclasses.

Use Cases:
    - Flexible access patterns where field names are computed or unknown
    - Interoperability with dict-based interfaces and APIs
    - Generic processing where field names are dynamic

Design Notes:
    - Works with any Pydantic BaseModel subclass
    - Uses getattr/hasattr for attribute access (compatible with extra="allow")
    - `__contains__` returns False for None values (presence semantics)

Example:
    >>> class MyModel(MixinDictLikeAccessors, BaseModel):
    ...     model_config = ConfigDict(extra="allow")
    ...     name: str = ""
    ...
    >>> m = MyModel(name="test", custom_field="value")
    >>> m.get("name")
    'test'
    >>> m["custom_field"]
    'value'
    >>> "name" in m
    True

.. versionadded:: 0.6.0
"""

from __future__ import annotations


class MixinDictLikeAccessors:
    """Mixin providing dict-like access methods for Pydantic models.

    Provides three dict-like methods:
        - `get(key, default=None)`: Safe access with optional default
        - `__getitem__(key)`: Bracket notation access (raises KeyError if missing)
        - `__contains__(key)`: Membership testing (returns False if value is None)

    This mixin is designed for use with Pydantic BaseModel subclasses,
    especially those with `extra="allow"` config that accept arbitrary fields.

    Attributes:
        None. This is a pure mixin that relies on getattr/hasattr.

    Example:
        >>> from pydantic import BaseModel, ConfigDict
        >>> class FlexibleModel(MixinDictLikeAccessors, BaseModel):
        ...     model_config = ConfigDict(extra="allow")
        ...     known_field: str = ""
        ...
        >>> model = FlexibleModel(known_field="value", extra_field=42)
        >>> model.get("known_field")
        'value'
        >>> model.get("extra_field")
        42
        >>> model.get("missing", "default")
        'default'
        >>> model["known_field"]
        'value'
        >>> "extra_field" in model
        True
    """

    def get(self, key: str, default: object = None) -> object:
        """Get field value by key with optional default.

        Provides dict-like access for interoperability with dict-based
        interfaces and APIs.

        Args:
            key: Field name to retrieve.
            default: Default value if field not found. Defaults to None.

        Returns:
            Field value if present, otherwise the default value.

        Example:
            >>> model = FlexibleModel(status="ok")
            >>> model.get("status")
            'ok'
            >>> model.get("missing_key", "fallback")
            'fallback'
        """
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> object:
        """Get field value by key using bracket notation.

        Enables dict-like bracket access: `model["field"]`.

        Args:
            key: Field name to retrieve.

        Returns:
            Field value.

        Raises:
            KeyError: If field does not exist on the model.

        Example:
            >>> model = FlexibleModel(count=5)
            >>> model["count"]
            5
            >>> model["nonexistent"]
            Traceback (most recent call last):
                ...
            KeyError: 'nonexistent'
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Check if field exists in model with a non-None value.

        Enables membership testing: `"field" in model`.

        Note:
            Returns False if the field exists but its value is None.
            This provides "presence" semantics rather than pure existence.

        Args:
            key: Field name to check.

        Returns:
            True if field exists and value is not None, False otherwise.

        Example:
            >>> model = FlexibleModel(active=True, empty=None)
            >>> "active" in model
            True
            >>> "empty" in model  # None values return False
            False
            >>> "missing" in model
            False
        """
        return hasattr(self, key) and getattr(self, key) is not None


__all__: list[str] = ["MixinDictLikeAccessors"]
