# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Payload Query Mixin.

Provides query methods for the RegistryPayload class.
These methods are thread-safe and require the registry to be frozen.

Design Principles:
    - All query methods require frozen state
    - Thread-safe concurrent access
    - O(1) resolve performance

Related:
    - OMN-2036: ProtocolPayloadRegistry implementation
    - RegistryPayload: Main class that uses this mixin

.. versionadded:: 0.7.0
"""

from __future__ import annotations

__all__ = [
    "MixinPayloadQuery",
]

import threading

from pydantic import BaseModel

from omnibase_infra.errors.error_payload_registry import PayloadRegistryError


class MixinPayloadQuery:
    """
    Mixin providing query methods for payload registry.

    This mixin implements:
        - Payload type resolution
        - Existence checking
        - Type listing

    Requires the following attributes to be defined by the host class:
        - _entries: dict[tuple[str, str], type[BaseModel]]
        - _lock: threading.Lock
        - _frozen: bool
        - _require_frozen(method_name: str) -> None

    .. versionadded:: 0.7.0
    """

    _entries: dict[tuple[str, str], type[BaseModel]]
    _lock: threading.Lock
    _frozen: bool

    def _require_frozen(self, method_name: str) -> None:
        """Raise if registry is not frozen. Must be provided by host class."""
        raise NotImplementedError(
            f"_require_frozen() must be provided by the host class. "
            f"Called from {method_name}()."
        )

    def resolve(
        self,
        payload_type: str,
        version: str,
    ) -> type[BaseModel]:
        """
        Resolve a (payload_type, version) tuple to its registered model class.

        Args:
            payload_type: The payload type identifier.
            version: The semantic version string.

        Returns:
            The registered Pydantic model class.

        Raises:
            PayloadRegistryError: If registry is not frozen
            PayloadRegistryError: If (payload_type, version) is not registered

        Thread Safety:
            Safe for concurrent access after freeze().

        .. versionadded:: 0.7.0
        """
        self._require_frozen("resolve")

        key = (payload_type, version)
        if key not in self._entries:
            registered = sorted(self._entries.keys())
            registered_str = (
                ", ".join(f"({pt!r}, {v!r})" for pt, v in registered)
                if registered
                else "(none)"
            )
            raise PayloadRegistryError(
                f"Unregistered payload type ({payload_type!r}, {version!r}). "
                f"Registered types: {registered_str}",
                payload_type=payload_type,
                version=version,
            )

        return self._entries[key]

    def has(
        self,
        payload_type: str,
        version: str,
    ) -> bool:
        """
        Check if a (payload_type, version) tuple is registered.

        Args:
            payload_type: The payload type identifier.
            version: The semantic version string.

        Returns:
            True if registered, False otherwise.

        Raises:
            PayloadRegistryError: If registry is not frozen

        Thread Safety:
            Safe for concurrent access after freeze().

        .. versionadded:: 0.7.0
        """
        self._require_frozen("has")
        return (payload_type, version) in self._entries

    def list_types(self) -> list[tuple[str, str]]:
        """
        List all registered (payload_type, version) tuples.

        Returns:
            List of (payload_type, version) tuples, sorted lexicographically.

        Raises:
            PayloadRegistryError: If registry is not frozen

        Thread Safety:
            Safe for concurrent access after freeze().

        .. versionadded:: 0.7.0
        """
        self._require_frozen("list_types")
        return sorted(self._entries.keys())
