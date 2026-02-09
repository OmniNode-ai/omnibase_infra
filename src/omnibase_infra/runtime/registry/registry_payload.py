# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Payload Registry Implementation.

Provides the RegistryPayload class that maps (payload_type, version) tuples
to Pydantic model classes for dynamic deserialization.

Design Principles:
    - Freeze-after-init pattern for thread-safe concurrent access
    - Clear error messages for unregistered types
    - Domain-agnostic: any Pydantic BaseModel subclass can be registered
    - Thread-safe registration with locking
    - Mixin-based decomposition for method count compliance

Thread Safety:
    RegistryPayload follows the freeze-after-init pattern:
    1. **Registration Phase** (thread-safe): Register payload types with lock
    2. **Freeze**: Call freeze() to lock the registry
    3. **Query Phase** (multi-threaded safe): Thread-safe lookups via frozen dict

Performance Characteristics:
    - Registration: O(1) per (payload_type, version) tuple
    - Resolve: O(1) dictionary access
    - List types: O(n log n) due to sorting

Ownership Boundary:
    omnibase_infra provides this registry primitive. Application repos
    register their own payload models at startup time.

Related:
    - OMN-2036: ProtocolPayloadRegistry implementation
    - ProtocolPayloadRegistry: Protocol interface definition
    - MixinPayloadRegistration: Registration methods
    - MixinPayloadQuery: Query methods

.. versionadded:: 0.7.0
"""

from __future__ import annotations

__all__ = [
    "RegistryPayload",
]

import warnings

from omnibase_infra.runtime.registry.mixin_payload_query import MixinPayloadQuery
from omnibase_infra.runtime.registry.mixin_payload_registration import (
    MixinPayloadRegistration,
)


class RegistryPayload(MixinPayloadRegistration, MixinPayloadQuery):
    """
    Payload Registry mapping (payload_type, version) to Pydantic model classes.

    Maps (payload_type, version) tuples to concrete Pydantic BaseModel subclasses,
    enabling dynamic deserialization of typed event payloads at runtime.

    Key Features:
        - **Tuple-keyed mapping**: (payload_type, version) -> model class
        - **Freeze-after-init**: Thread-safe concurrent reads after freeze
        - **Clear error reporting**: Descriptive errors for unregistered types
        - **Domain-agnostic**: Any BaseModel subclass can be registered

    Thread Safety:
        Follows the freeze-after-init pattern:
        1. **Registration Phase**: Thread-safe registration via lock
        2. **Freeze**: Validation and locking
        3. **Query Phase**: Thread-safe concurrent lookups

    Example:
        >>> from omnibase_infra.runtime.registry import RegistryPayload
        >>> from pydantic import BaseModel
        >>>
        >>> class ModelClaudeHookEvent(BaseModel):
        ...     event_type: str
        ...     payload: dict
        ...
        >>> registry = RegistryPayload()
        >>> registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)
        >>> registry.freeze()
        >>>
        >>> # Resolve (thread-safe after freeze)
        >>> cls = registry.resolve("ModelClaudeHookEvent", "1.0.0")
        >>> assert cls is ModelClaudeHookEvent

    See Also:
        - :class:`ProtocolPayloadRegistry`: Protocol interface
        - :class:`MixinPayloadRegistration`: Registration methods
        - :class:`MixinPayloadQuery`: Query methods

    .. versionadded:: 0.7.0
    """

    def __init__(self) -> None:
        """Initialize RegistryPayload with empty registry."""
        self._init_payload_state()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_frozen(self) -> bool:
        """
        Check if the registry is frozen.

        Returns:
            True if frozen and registration is disabled.

        .. versionadded:: 0.7.0
        """
        return self._frozen

    @property
    def entry_count(self) -> int:
        """
        Get the number of registered payload entries.

        Returns:
            Number of registered (payload_type, version) tuples.

        .. versionadded:: 0.7.0
        """
        return len(self._entries)

    # =========================================================================
    # Testing Utility
    # =========================================================================

    def clear(self) -> None:
        """Clear all registered types.

        Warning:
            This method is intended for **testing purposes only**.
            Calling it in production code will emit a warning.
            It breaks the immutability guarantee after startup.

        Thread Safety:
            This method is protected by the instance lock to ensure
            thread-safe clearing of the registry.

        .. versionadded:: 0.7.0
        """
        warnings.warn(
            "RegistryPayload.clear() is intended for testing only. "
            "Do not use in production code.",
            UserWarning,
            stacklevel=2,
        )
        with self._lock:
            self._entries.clear()
            self._frozen = False

    # =========================================================================
    # Dunder Methods
    # =========================================================================

    def __len__(self) -> int:
        """Return the number of registered payload types.

        Raises:
            PayloadRegistryError: If registry is not frozen.
        """
        self._require_frozen("__len__")
        return len(self._entries)

    def __contains__(self, key: tuple[str, str]) -> bool:
        """Check if (payload_type, version) is registered using 'in' operator.

        Raises:
            PayloadRegistryError: If registry is not frozen.
        """
        self._require_frozen("__contains__")
        return key in self._entries

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"RegistryPayload[entries={len(self._entries)}, frozen={self._frozen}]"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        types = sorted(self._entries.keys())[:10]
        type_repr = (
            repr(types)
            if len(self._entries) <= 10
            else f"<{len(self._entries)} entries>"
        )
        return f"RegistryPayload(entries={type_repr}, frozen={self._frozen})"
