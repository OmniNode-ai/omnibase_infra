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
from typing import Any

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
    def _coerce_tags_to_tuple(cls, v: Any) -> tuple[str, ...]:
        """Convert list/sequence to tuple for immutability."""
        if isinstance(v, tuple):
            return v  # type: ignore[return-value]  # Runtime validated by Pydantic
        if isinstance(v, list | set | frozenset):
            return tuple(str(item) for item in v)
        # For unrecognized types, wrap in tuple and let Pydantic validate contents
        return (v,) if v is not None else ()

    @field_validator("meta", mode="before")
    @classmethod
    def _coerce_meta_to_tuple(cls, v: Any) -> tuple[tuple[str, str], ...]:
        """Convert dict/mapping to tuple of pairs for immutability."""
        if isinstance(v, tuple):
            return v  # type: ignore[return-value]  # Runtime validated by Pydantic
        if isinstance(v, Mapping):
            return tuple((str(k), str(val)) for k, val in v.items())
        # For unrecognized types, return empty tuple (Pydantic will validate)
        return ()

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
