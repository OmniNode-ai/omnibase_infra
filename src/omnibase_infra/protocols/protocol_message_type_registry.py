# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Protocol definition for the Message Type Registry.

Defines the interface contract for message type registry implementations.
Enables duck typing and test mocking while ensuring consistent API.

Design Principles:
    - Protocol-based interface for flexibility and testability
    - Runtime-checkable for isinstance() validation
    - Comprehensive API covering registration, query, and validation
    - Thread-safety documented in implementation requirements

Related:
    - OMN-937: Central Message Type Registry implementation
    - RegistryMessageType: Primary implementation of this protocol

.. versionadded:: 0.5.0
"""

from typing import Protocol, runtime_checkable

from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.models.registry.model_message_type_entry import (
    ModelMessageTypeEntry,
)


@runtime_checkable
class ProtocolMessageTypeRegistry(Protocol):
    """
    Protocol for message type registry implementations.

    Defines the interface contract for registries that map message types
    to handler implementations with category and domain constraints.

    Implementations must follow the freeze-after-init pattern:
    1. Registration phase: All register_* methods available
    2. Freeze: Call freeze() to lock the registry
    3. Query phase: Only lookup methods available

    Thread Safety Requirements:
        - Registration methods must be thread-safe during registration phase
        - After freeze(), all query methods must be safe for concurrent access
        - Implementations should use appropriate locking strategies

    Example Implementation:
        .. code-block:: python

            class MyRegistry:
                '''Custom registry implementation.'''

                def register_message_type(
                    self,
                    entry: ModelMessageTypeEntry,
                ) -> None:
                    ...

                def get_handlers(
                    self,
                    message_type: str,
                    topic_category: EnumMessageCategory,
                    topic_domain: str,
                ) -> list[str]:
                    ...

                # ... implement all protocol methods

            # Verify protocol compliance
            registry: ProtocolMessageTypeRegistry = MyRegistry()

    See Also:
        - :class:`RegistryMessageType`: Primary implementation

    .. versionadded:: 0.5.0
    """

    # =========================================================================
    # Registration Methods (available before freeze)
    # =========================================================================

    def register_message_type(
        self,
        entry: ModelMessageTypeEntry,
    ) -> None:
        """
        Register a message type with its handler mappings.

        Associates a message type with handler(s) and defines constraints
        for valid usage. Supports fan-out by registering multiple handlers
        for the same message type.

        Args:
            entry: The message type entry containing handler mappings
                and constraints.

        Raises:
            ModelOnexError: If registry is frozen (INVALID_STATE)
            ModelOnexError: If entry validation fails (VALIDATION_ERROR)

        Thread Safety:
            Must be thread-safe during registration phase.

        .. versionadded:: 0.5.0
        """
        ...

    def freeze(self) -> None:
        """
        Freeze the registry to prevent further modifications.

        Once frozen, registration methods will raise ModelOnexError.
        This enables thread-safe concurrent access during query phase.

        Idempotent: Calling freeze() multiple times has no additional effect.

        Raises:
            ModelOnexError: If validation fails during freeze (e.g., missing
                handler references)

        Thread Safety:
            Must be thread-safe.

        .. versionadded:: 0.5.0
        """
        ...

    # =========================================================================
    # Query Methods (available after freeze)
    # =========================================================================

    def get_handlers(
        self,
        message_type: str,
        topic_category: EnumMessageCategory,
        topic_domain: str,
    ) -> list[str]:
        """
        Get handler IDs for a message type with validation.

        Validates that the message type is allowed in the given category
        and that the topic domain matches domain constraints.

        Args:
            message_type: The message type to look up.
            topic_category: The category inferred from the topic.
            topic_domain: The domain extracted from the topic.

        Returns:
            List of handler IDs that can process this message type.
            Empty list if no handlers registered.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)
            ModelOnexError: If category constraint violated (VALIDATION_ERROR)
            ModelOnexError: If domain constraint violated (VALIDATION_ERROR)

        Thread Safety:
            Must be safe for concurrent access after freeze().

        .. versionadded:: 0.5.0
        """
        ...

    def get_entry(self, message_type: str) -> ModelMessageTypeEntry | None:
        """
        Get the registry entry for a message type.

        Args:
            message_type: The message type to look up.

        Returns:
            The registry entry if found, None otherwise.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        Thread Safety:
            Must be safe for concurrent access after freeze().

        .. versionadded:: 0.5.0
        """
        ...

    def has_message_type(self, message_type: str) -> bool:
        """
        Check if a message type is registered.

        Args:
            message_type: The message type to check.

        Returns:
            True if registered, False otherwise.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        Thread Safety:
            Must be safe for concurrent access after freeze().

        .. versionadded:: 0.5.0
        """
        ...

    def list_message_types(
        self,
        category: EnumMessageCategory | None = None,
        domain: str | None = None,
    ) -> list[str]:
        """
        List registered message types with optional filtering.

        Args:
            category: Optional filter by allowed category.
            domain: Optional filter by owning domain.

        Returns:
            List of message type names matching the filters.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        Thread Safety:
            Must be safe for concurrent access after freeze().

        .. versionadded:: 0.5.0
        """
        ...

    def list_domains(self) -> list[str]:
        """
        List all domains that have registered message types.

        Returns:
            List of unique domain names, sorted alphabetically.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        Thread Safety:
            Must be safe for concurrent access after freeze().

        .. versionadded:: 0.5.0
        """
        ...

    # =========================================================================
    # Validation Methods
    # =========================================================================

    def validate_startup(self) -> list[str]:
        """
        Perform startup-time validation and return any errors.

        Validates registry consistency:
        - All handler references are valid
        - No conflicting category constraints
        - Domain constraints are properly configured

        This method should be called after freeze() to ensure fail-fast
        behavior before consumers start processing messages.

        Returns:
            List of validation error messages. Empty list if valid.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        Thread Safety:
            Must be safe for concurrent access after freeze().

        .. versionadded:: 0.5.0
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

        .. versionadded:: 0.5.0
        """
        ...

    @property
    def entry_count(self) -> int:
        """
        Get the number of registered message type entries.

        Returns:
            Number of registered message types.

        .. versionadded:: 0.5.0
        """
        ...


__all__ = ["ProtocolMessageTypeRegistry"]
