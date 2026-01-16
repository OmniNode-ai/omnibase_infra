# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul intent payload model for registration orchestrator.

This module provides the typed payload model for Consul registration intents.

Thread Safety:
    This model is fully immutable (frozen=True) with immutable field types.
    - ``tags`` uses tuple instead of list
    - ``meta`` uses tuple of key-value pairs instead of dict

    For dict-like access to metadata, use the ``meta_dict`` property which
    returns a MappingProxyType (read-only view).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConsulIntentPayload(BaseModel):
    """Payload for Consul registration intents.

    Used by the Consul adapter to register nodes in service discovery.

    This model is fully immutable to support thread-safe concurrent access.
    All fields use immutable types (tuple instead of list/dict).

    Attributes:
        service_name: Name to register the node as in Consul.
        tags: Immutable tuple of service tags for Consul filtering.
        meta: Immutable tuple of (key, value) pairs for metadata.
            Use the ``meta_dict`` property for dict-like read access.

    Example:
        >>> payload = ModelConsulIntentPayload(
        ...     service_name="my-service",
        ...     tags=["production", "v1"],
        ...     meta={"env": "prod", "region": "us-west"},
        ... )
        >>> payload.tags
        ('production', 'v1')
        >>> payload.meta_dict["env"]
        'prod'
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    service_name: str = Field(
        ...,
        min_length=1,
        description="Service name to register in Consul",
    )
    tags: tuple[str, ...] = Field(
        default=(),
        description="Immutable tuple of service tags for Consul filtering",
    )
    meta: tuple[tuple[str, str], ...] = Field(
        default=(),
        description="Immutable tuple of (key, value) pairs for metadata",
    )

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags_to_tuple(cls, v: object) -> tuple[str, ...]:
        """Convert list/sequence to tuple for immutability.

        This validator ensures explicit handling of all input types rather than
        silent fallback, which could mask invalid input. In strict mode, all
        items must already be strings - no silent type coercion.

        Args:
            v: The input value to coerce. Must be a tuple, list, set, or frozenset
                containing only string elements.

        Returns:
            A tuple of string tags.

        Raises:
            ValueError: If the input is not a recognized sequence type or if
                any element is not a string. This ensures invalid input types
                are explicitly rejected rather than silently coerced.

        Edge Cases:
            - ``None``: Raises ValueError (explicit rejection)
            - Empty sequence ``[]``, ``()``: Returns empty tuple
            - Invalid types (int, str, dict): Raises ValueError
            - Non-string elements: Raises ValueError (strict mode)
            - Valid sequences of strings: Converts to tuple
        """
        if isinstance(v, tuple):
            # Validate tuple contents in strict mode
            for i, item in enumerate(v):
                if not isinstance(item, str):
                    raise ValueError(
                        f"tags[{i}] must be a string, got {type(item).__name__}"
                    )
            return v  # type: ignore[return-value]  # NOTE: runtime type validated by Pydantic
        if isinstance(v, list | set | frozenset):
            # Validate and convert to tuple - strict mode requires string elements
            result: list[str] = []
            for i, item in enumerate(v):
                if not isinstance(item, str):
                    raise ValueError(
                        f"tags[{i}] must be a string, got {type(item).__name__}"
                    )
                result.append(item)
            return tuple(result)
        raise ValueError(
            f"tags must be a tuple, list, set, or frozenset, got {type(v).__name__}"
        )

    @field_validator("meta", mode="before")
    @classmethod
    def _coerce_meta_to_tuple(cls, v: object) -> tuple[tuple[str, str], ...]:
        """Convert dict/mapping to tuple of pairs for immutability.

        This validator ensures explicit handling of all input types rather than
        silent fallback to empty tuple, which could mask invalid input. In strict
        mode, all keys and values must already be strings - no silent type coercion.

        Args:
            v: The input value to coerce. Must be either a tuple of (key, value)
                pairs or a Mapping (dict-like object) with string keys and values.

        Returns:
            A tuple of (key, value) string pairs.

        Raises:
            ValueError: If the input is neither a tuple nor a Mapping type, or if
                any key/value is not a string. This ensures invalid input types
                are explicitly rejected rather than silently coerced.

        Edge Cases:
            - ``None``: Raises ValueError (explicit rejection)
            - Empty Mapping ``{}``: Returns empty tuple
            - Empty tuple ``()``: Passed through (same as default)
            - Invalid types (list, int, str): Raises ValueError
            - Non-string keys/values: Raises ValueError (strict mode)
            - Non-empty Mapping with strings: Converts to tuple of (key, value) pairs
        """
        if isinstance(v, tuple):
            # Validate tuple contents in strict mode
            for i, item in enumerate(v):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError(
                        f"meta[{i}] must be a (key, value) tuple, got {type(item).__name__}"
                    )
                key, val = item
                if not isinstance(key, str):
                    raise ValueError(
                        f"meta[{i}][0] (key) must be a string, got {type(key).__name__}"
                    )
                if not isinstance(val, str):
                    raise ValueError(
                        f"meta[{i}][1] (value) must be a string, got {type(val).__name__}"
                    )
            return v  # type: ignore[return-value]  # NOTE: runtime type validated by Pydantic
        if isinstance(v, Mapping):
            # Validate and convert to tuple - strict mode requires string keys/values
            result: list[tuple[str, str]] = []
            for key, val in v.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"meta key must be a string, got {type(key).__name__}"
                    )
                if not isinstance(val, str):
                    raise ValueError(
                        f"meta[{key!r}] value must be a string, got {type(val).__name__}"
                    )
                result.append((key, val))
            return tuple(result)
        raise ValueError(f"meta must be a tuple or Mapping, got {type(v).__name__}")

    @property
    def meta_dict(self) -> MappingProxyType[str, str]:
        """Return a read-only dict view of the metadata.

        Returns:
            MappingProxyType providing dict-like read access to metadata.
        """
        return MappingProxyType(dict(self.meta))


__all__ = [
    "ModelConsulIntentPayload",
]
