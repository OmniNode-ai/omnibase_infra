# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Protocol definition for the Payload Registry.

Defines the interface contract for payload registry implementations that map
(payload_type, version) tuples to Pydantic model classes for dynamic
deserialization.

Design Principles:
    - Protocol-based interface for flexibility and testability
    - Runtime-checkable for isinstance() validation
    - Freeze-after-init pattern for thread-safe concurrent access
    - Clear error reporting for unregistered types
    - Domain-agnostic: any Pydantic BaseModel subclass can be registered

Ownership Boundary:
    omnibase_infra provides the registry primitive. Application repos
    register their own payload models at startup.

Related:
    - OMN-2036: ProtocolPayloadRegistry implementation
    - RegistryPayload: Primary implementation of this protocol

.. versionadded:: 0.7.0
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class ProtocolPayloadRegistry(Protocol):
    """
    Protocol for payload registry implementations.

    Defines the interface contract for registries that map
    (payload_type, version) tuples to Pydantic model classes,
    enabling dynamic deserialization of typed event payloads.

    Implementations must follow the freeze-after-init pattern:
    1. Registration phase: All register() calls available
    2. Freeze: Call freeze() to lock the registry
    3. Query phase: Only resolve/has/list methods available

    Thread Safety Requirements:
        - Registration methods must be thread-safe during registration phase
        - After freeze(), all query methods must be safe for concurrent access
        - Implementations should use appropriate locking strategies

    Example Implementation:
        .. code-block:: python

            class MyPayloadRegistry:
                '''Custom payload registry.'''

                def register(
                    self,
                    payload_type: str,
                    version: str,
                    model_class: type[BaseModel],
                ) -> None:
                    ...

                def resolve(
                    self,
                    payload_type: str,
                    version: str,
                ) -> type[BaseModel]:
                    ...

                # ... implement all protocol methods

            # Verify protocol compliance
            registry: ProtocolPayloadRegistry = MyPayloadRegistry()

    See Also:
        - :class:`RegistryPayload`: Primary implementation

    .. versionadded:: 0.7.0
    """

    # =========================================================================
    # Registration Methods (available before freeze)
    # =========================================================================

    def register(
        self,
        payload_type: str,
        version: str,
        model_class: type[BaseModel],
    ) -> None:
        """
        Register a payload model class for a (payload_type, version) tuple.

        Associates a Pydantic model class with a specific payload type and
        version, enabling dynamic deserialization at runtime.

        Args:
            payload_type: The payload type identifier (e.g., "ModelClaudeHookEvent").
                Must be a non-empty string.
            version: The semantic version string (e.g., "1.0.0").
                Must be a non-empty string.
            model_class: The Pydantic BaseModel subclass to register.
                Must be a concrete class (not BaseModel itself).

        Raises:
            PayloadRegistryError: If registry is frozen
            PayloadRegistryError: If (payload_type, version) is already registered
            PayloadRegistryError: If model_class is not a BaseModel subclass

        Thread Safety:
            Must be thread-safe during registration phase.

        .. versionadded:: 0.7.0
        """
        ...

    def freeze(self) -> None:
        """
        Freeze the registry to prevent further modifications.

        Once frozen, register() will raise PayloadRegistryError.
        This enables thread-safe concurrent access during query phase.

        Idempotent: Calling freeze() multiple times has no additional effect.

        Thread Safety:
            Must be thread-safe.

        .. versionadded:: 0.7.0
        """
        ...

    # =========================================================================
    # Query Methods (available after freeze)
    # =========================================================================

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
            Must be safe for concurrent access after freeze().

        .. versionadded:: 0.7.0
        """
        ...

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
            Must be safe for concurrent access after freeze().

        .. versionadded:: 0.7.0
        """
        ...

    def list_types(self) -> list[tuple[str, str]]:
        """
        List all registered (payload_type, version) tuples.

        Returns:
            List of (payload_type, version) tuples, sorted lexicographically.

        Raises:
            PayloadRegistryError: If registry is not frozen

        Thread Safety:
            Must be safe for concurrent access after freeze().

        .. versionadded:: 0.7.0
        """
        ...

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
        ...

    @property
    def entry_count(self) -> int:
        """
        Get the number of registered payload entries.

        Returns:
            Number of registered (payload_type, version) tuples.

        .. versionadded:: 0.7.0
        """
        ...


__all__ = ["ProtocolPayloadRegistry"]
