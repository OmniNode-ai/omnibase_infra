# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Message Type Query Mixin.

Provides query methods for the RegistryMessageType class.
These methods are thread-safe and require the registry to be frozen.

Design Principles:
    - All query methods require frozen state
    - Thread-safe concurrent access
    - O(1) handler lookup performance

Related:
    - OMN-937: Central Message Type Registry implementation
    - RegistryMessageType: Main class that uses this mixin

.. versionadded:: 0.6.0
"""

from __future__ import annotations

import logging

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.errors import MessageTypeRegistryError
from omnibase_infra.models.errors import ModelMessageTypeRegistryErrorContext
from omnibase_infra.models.registry.model_message_type_entry import (
    ModelMessageTypeEntry,
)

__all__ = [
    "MixinMessageTypeQuery",
]

logger = logging.getLogger(__name__)


class MixinMessageTypeQuery:
    """
    Mixin providing query methods for message type registry.

    This mixin implements:
        - Handler lookup with constraint validation
        - Unchecked handler lookup for introspection
        - Entry retrieval and existence checking
        - List methods for message types, domains, and handlers

    Requires the following attributes to be defined by the host class:
        - _entries: dict[str, ModelMessageTypeEntry]
        - _domains: set[str]
        - _handler_references: set[str]
        - _category_index: dict[EnumMessageCategory, list[str]]
        - _domain_index: dict[str, list[str]]
        - _frozen: bool
        - _logger: logging.Logger

    .. versionadded:: 0.6.0
    """

    # Type hints for mixin attributes (defined by host class)
    _entries: dict[str, ModelMessageTypeEntry]
    _domains: set[str]
    _handler_references: set[str]
    _category_index: dict[EnumMessageCategory, list[str]]
    _domain_index: dict[str, list[str]]
    _frozen: bool
    _logger: logging.Logger

    def _require_frozen(self, method_name: str) -> None:
        """Raise error if registry is not frozen."""
        if not self._frozen:
            raise ModelOnexError(
                message=f"{method_name}() called before freeze(). "
                f"Registration MUST complete and freeze() MUST be called "
                f"before queries. This is required for thread safety.",
                error_code=EnumCoreErrorCode.INVALID_STATE,
            )

    def get_handlers(
        self,
        message_type: str,
        topic_category: EnumMessageCategory,
        topic_domain: str,
    ) -> list[str]:
        """
        Get handler IDs for a message type with constraint validation.

        Validates that:
        1. The message type is registered
        2. The topic category is allowed for this message type
        3. The topic domain matches domain constraints

        Args:
            message_type: The message type to look up.
            topic_category: The category inferred from the topic.
            topic_domain: The domain extracted from the topic.

        Returns:
            List of handler IDs that can process this message type.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)
            MessageTypeRegistryError: If message type not registered
            MessageTypeRegistryError: If category constraint violated
            MessageTypeRegistryError: If domain constraint violated

        Example:
            >>> handlers = registry.get_handlers(
            ...     message_type="UserCreated",
            ...     topic_category=EnumMessageCategory.EVENT,
            ...     topic_domain="user",
            ... )
            >>> # Returns ["user-handler", "audit-handler"] etc.

        .. versionadded:: 0.5.0
        """
        self._require_frozen("get_handlers")

        # Look up entry
        entry = self._entries.get(message_type)
        if entry is None:
            registered = sorted(self._entries.keys())[:10]
            suffix = "..." if len(self._entries) > 10 else ""
            raise MessageTypeRegistryError(
                f"No handler mapping for message type '{message_type}'. "
                f"Registered types: {registered}{suffix}",
                registry_context=ModelMessageTypeRegistryErrorContext(
                    message_type=message_type,
                ),
                registered_types=registered,
            )

        # Check if entry is enabled
        if not entry.enabled:
            raise MessageTypeRegistryError(
                f"Message type '{message_type}' is registered but disabled.",
                registry_context=ModelMessageTypeRegistryErrorContext(
                    message_type=message_type,
                ),
            )

        # Validate category constraint
        category_outcome = entry.validate_category(topic_category)
        if not category_outcome:
            raise MessageTypeRegistryError(
                category_outcome.error_message
                or f"Category validation failed for '{message_type}'",
                registry_context=ModelMessageTypeRegistryErrorContext(
                    message_type=message_type,
                    category=topic_category,
                ),
            )

        # Validate domain constraint
        domain_outcome = entry.domain_constraint.validate_consumption(
            topic_domain,
            message_type,
        )
        if not domain_outcome:
            raise MessageTypeRegistryError(
                domain_outcome.error_message
                or f"Domain validation failed for '{message_type}'",
                registry_context=ModelMessageTypeRegistryErrorContext(
                    message_type=message_type,
                    domain=topic_domain,
                ),
            )

        return list(entry.handler_ids)

    def get_handlers_unchecked(
        self,
        message_type: str,
    ) -> list[str] | None:
        """
        Get handler IDs for a message type without constraint validation.

        Use this method when you need to look up handlers without
        performing category or domain validation (e.g., for introspection).

        Note:
            This method returns None for disabled entries as well as
            unregistered message types. Use get_entry() if you need to
            distinguish between these cases.

        Args:
            message_type: The message type to look up.

        Returns:
            List of handler IDs if registered and enabled, None if not found
            or if the entry is disabled.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        .. versionadded:: 0.5.0
        """
        self._require_frozen("get_handlers_unchecked")

        entry = self._entries.get(message_type)
        if entry is None or not entry.enabled:
            return None
        return list(entry.handler_ids)

    def get_entry(self, message_type: str) -> ModelMessageTypeEntry | None:
        """
        Get the registry entry for a message type.

        Args:
            message_type: The message type to look up.

        Returns:
            The registry entry if found, None otherwise.

        Note:
            Unlike get_handlers_unchecked(), this method returns entries
            regardless of their enabled status. Check entry.enabled if you
            need to distinguish between enabled and disabled entries.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        .. versionadded:: 0.5.0
        """
        self._require_frozen("get_entry")
        return self._entries.get(message_type)

    def has_message_type(self, message_type: str) -> bool:
        """
        Check if a message type is registered.

        Args:
            message_type: The message type to check.

        Returns:
            True if registered and enabled, False otherwise.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        .. versionadded:: 0.5.0
        """
        self._require_frozen("has_message_type")
        entry = self._entries.get(message_type)
        return entry is not None and entry.enabled

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
            List of message type names matching the filters, sorted.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        Example:
            >>> # List all event message types
            >>> registry.list_message_types(category=EnumMessageCategory.EVENT)
            ['OrderCreated', 'UserCreated', 'UserUpdated']
            >>> # List all message types in user domain
            >>> registry.list_message_types(domain="user")
            ['UserCreated', 'UserUpdated']

        .. versionadded:: 0.5.0
        """
        self._require_frozen("list_message_types")

        # Apply filters
        if category is not None and domain is not None:
            # Both filters: intersection of category and domain indexes
            cat_types = set(self._category_index.get(category, []))
            dom_types = set(self._domain_index.get(domain, []))
            result = cat_types & dom_types
        elif category is not None:
            result = set(self._category_index.get(category, []))
        elif domain is not None:
            result = set(self._domain_index.get(domain, []))
        else:
            result = set(self._entries.keys())

        # Filter to enabled entries only
        enabled_result = [mt for mt in result if self._entries[mt].enabled]

        return sorted(enabled_result)

    def list_domains(self) -> list[str]:
        """
        List all domains that have registered message types.

        Returns:
            List of unique domain names, sorted alphabetically.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        .. versionadded:: 0.5.0
        """
        self._require_frozen("list_domains")
        return sorted(self._domains)

    def list_handlers(self) -> list[str]:
        """
        List all handler IDs referenced in the registry.

        Returns:
            List of unique handler IDs, sorted alphabetically.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        .. versionadded:: 0.5.0
        """
        self._require_frozen("list_handlers")
        return sorted(self._handler_references)
