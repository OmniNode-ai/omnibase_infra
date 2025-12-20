# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Central Message Type Registry Implementation.

Provides the MessageTypeRegistry class that maps message types to handler
implementations and enforces topic category constraints and domain ownership.

Design Principles:
    - Freeze-after-init pattern for thread-safe concurrent access
    - Startup-time validation with fail-fast behavior
    - Domain ownership enforcement derived from topic names
    - Clear error messages for configuration issues
    - Extensibility for new domains

Thread Safety:
    MessageTypeRegistry follows the freeze-after-init pattern:
    1. **Registration Phase** (single-threaded): Register message types
    2. **Freeze**: Call freeze() to validate and lock the registry
    3. **Query Phase** (multi-threaded safe): Thread-safe lookups

Performance Characteristics:
    - Registration: O(1) per message type
    - Handler lookup: O(1) dictionary access
    - Domain validation: O(1) constraint check
    - Startup validation: O(n) where n = number of entries

Related:
    - OMN-937: Central Message Type Registry implementation
    - OMN-934: Message Dispatch Engine (uses this registry)
    - ProtocolMessageTypeRegistry: Interface definition

.. versionadded:: 0.5.0
"""

from __future__ import annotations

__all__ = [
    "MessageTypeRegistry",
    "MessageTypeRegistryError",
]

import logging
import re
import threading
from collections import defaultdict

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.models.errors.model_onex_error import ModelOnexError

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.runtime.registry.model_domain_constraint import (
    ModelDomainConstraint,
)
from omnibase_infra.runtime.registry.model_message_type_entry import (
    ModelMessageTypeEntry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Registry Error
# =============================================================================


class MessageTypeRegistryError(RuntimeHostError):
    """Error raised when message type registry operations fail.

    Used for:
    - Missing message type mappings
    - Category constraint violations
    - Domain constraint violations
    - Registration validation failures

    Extends RuntimeHostError for consistency with infrastructure error patterns.

    Example:
        >>> try:
        ...     handlers = registry.get_handlers("UnknownType", category, domain)
        ... except MessageTypeRegistryError as e:
        ...     print(f"Handler not found: {e}")

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        message: str,
        message_type: str | None = None,
        domain: str | None = None,
        category: EnumMessageCategory | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize MessageTypeRegistryError.

        Args:
            message: Human-readable error message
            message_type: The message type that caused the error (if applicable)
            domain: The domain involved in the error (if applicable)
            category: The category involved in the error (if applicable)
            **extra_context: Additional context information
        """
        # Build extra context dict
        extra: dict[str, object] = dict(extra_context)
        if message_type is not None:
            extra["message_type"] = message_type
        if domain is not None:
            extra["domain"] = domain
        if category is not None:
            extra["category"] = category.value

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.VALIDATION_FAILED,
            context=None,  # No ModelInfraErrorContext needed for registry errors
            **extra,
        )


# =============================================================================
# Topic Domain Extraction
# =============================================================================

# Pattern for extracting domain from topics
# Handles both ONEX format (onex.<domain>.<type>) and environment-aware format
# (<env>.<domain>.<category>.<version>)
_DOMAIN_PATTERN = re.compile(
    r"^(?:onex\.)?(?P<env>[a-zA-Z0-9_-]+)\.(?P<domain>[a-zA-Z0-9_-]+)\.",
    re.IGNORECASE,
)


def extract_domain_from_topic(topic: str) -> str | None:
    """
    Extract the domain from a topic string.

    Domain extraction follows ONEX topic naming conventions:
    - ONEX Kafka format: "onex.<domain>.<type>" -> domain
    - Environment-aware format: "<env>.<domain>.<category>.<version>" -> domain

    Args:
        topic: The topic string to extract domain from.

    Returns:
        The extracted domain string, or None if extraction fails.

    Examples:
        >>> extract_domain_from_topic("onex.registration.events")
        'registration'
        >>> extract_domain_from_topic("dev.user.events.v1")
        'user'
        >>> extract_domain_from_topic("prod.order.commands.v2")
        'order'
        >>> extract_domain_from_topic("invalid")
        None

    .. versionadded:: 0.5.0
    """
    if not topic:
        return None

    # Try to match the pattern
    match = _DOMAIN_PATTERN.match(topic)
    if not match:
        # Fallback: try simple split on dots and take second segment
        parts = topic.split(".")
        if len(parts) >= 2:
            return parts[1]
        return None

    # For "onex.<domain>.*" format, the domain is the first captured group
    # For "<env>.<domain>.*" format, the domain is the second captured group
    groups = match.groupdict()
    env_or_domain = groups.get("env", "")
    domain = groups.get("domain", "")

    # If first segment is "onex", return second segment
    if env_or_domain.lower() == "onex":
        return domain

    # For environment-aware format, second segment is the domain
    return domain


# =============================================================================
# Message Type Registry
# =============================================================================


class MessageTypeRegistry:
    """
    Central Message Type Registry for ONEX runtime dispatch.

    Maps message types to handler implementations and enforces topic category
    constraints and domain ownership rules. This registry is the single source
    of truth for message type routing configuration.

    Key Features:
        - **Message Type Mapping**: Maps message types to handler ID(s)
        - **Fan-out Support**: Multiple handlers can process the same message type
        - **Category Constraints**: Validates message types against topic categories
        - **Domain Ownership**: Enforces domain isolation with opt-in cross-domain
        - **Startup Validation**: Fail-fast behavior before message processing
        - **Extensibility**: Easy to add new domains and message types

    Thread Safety:
        Follows the freeze-after-init pattern:
        1. **Registration Phase**: Single-threaded registration
        2. **Freeze**: Validation and locking
        3. **Query Phase**: Thread-safe concurrent lookups

    Example:
        >>> from omnibase_infra.runtime.registry import (
        ...     MessageTypeRegistry,
        ...     ModelMessageTypeEntry,
        ...     ModelDomainConstraint,
        ... )
        >>> from omnibase_infra.enums import EnumMessageCategory
        >>>
        >>> # Create registry and register message types
        >>> registry = MessageTypeRegistry()
        >>> entry = ModelMessageTypeEntry(
        ...     message_type="UserCreated",
        ...     handler_ids=("user-handler",),
        ...     allowed_categories=frozenset([EnumMessageCategory.EVENT]),
        ...     domain_constraint=ModelDomainConstraint(owning_domain="user"),
        ... )
        >>> registry.register_message_type(entry)
        >>>
        >>> # Freeze and validate
        >>> registry.freeze()
        >>> errors = registry.validate_startup()
        >>> if errors:
        ...     raise RuntimeError(f"Validation failed: {errors}")
        >>>
        >>> # Query handlers (thread-safe after freeze)
        >>> handlers = registry.get_handlers(
        ...     message_type="UserCreated",
        ...     topic_category=EnumMessageCategory.EVENT,
        ...     topic_domain="user",
        ... )

    Attributes:
        _entries: Dictionary mapping message_type -> ModelMessageTypeEntry
        _domains: Set of all registered domains
        _handler_references: Set of all referenced handler IDs
        _category_index: Index of message types by category
        _domain_index: Index of message types by domain
        _frozen: If True, registration is disabled
        _lock: Lock protecting registration operations

    See Also:
        - :class:`ModelMessageTypeEntry`: Entry model definition
        - :class:`ModelDomainConstraint`: Domain constraint model
        - :class:`ProtocolMessageTypeRegistry`: Protocol interface

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        logger_instance: logging.Logger | None = None,
    ) -> None:
        """
        Initialize MessageTypeRegistry with empty registries.

        Creates empty message type registry. Register message types before
        freeze(). Call validate_startup() after freeze() to ensure fail-fast
        behavior.

        Args:
            logger_instance: Optional custom logger for structured logging.
                If not provided, uses module-level logger.
        """
        self._logger: logging.Logger = (
            logger_instance if logger_instance is not None else logger
        )

        # Primary storage: message_type -> entry
        self._entries: dict[str, ModelMessageTypeEntry] = {}

        # Domain tracking
        self._domains: set[str] = set()

        # Handler reference tracking (for validation)
        self._handler_references: set[str] = set()

        # Indexes for efficient queries
        self._category_index: dict[EnumMessageCategory, list[str]] = defaultdict(list)
        self._domain_index: dict[str, list[str]] = defaultdict(list)

        # Freeze state
        self._frozen: bool = False
        self._lock: threading.Lock = threading.Lock()

    # =========================================================================
    # Registration Methods
    # =========================================================================

    def register_message_type(
        self,
        entry: ModelMessageTypeEntry,
    ) -> None:
        """
        Register a message type with its handler mappings.

        Associates a message type with handler(s) and defines constraints.
        If the message type is already registered, handlers are merged
        (fan-out pattern).

        Args:
            entry: The message type entry containing handler mappings.

        Raises:
            ModelOnexError: If registry is frozen (INVALID_STATE)
            ModelOnexError: If entry is None (INVALID_PARAMETER)
            MessageTypeRegistryError: If entry validation fails

        Example:
            >>> registry.register_message_type(ModelMessageTypeEntry(
            ...     message_type="OrderCreated",
            ...     handler_ids=("order-handler",),
            ...     allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            ...     domain_constraint=ModelDomainConstraint(owning_domain="order"),
            ... ))

        .. versionadded:: 0.5.0
        """
        if entry is None:
            raise ModelOnexError(
                message="Cannot register None entry. ModelMessageTypeEntry is required.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        with self._lock:
            if self._frozen:
                raise ModelOnexError(
                    message="Cannot register message type: MessageTypeRegistry is frozen. "
                    "Registration is not allowed after freeze() has been called.",
                    error_code=EnumCoreErrorCode.INVALID_STATE,
                )

            message_type = entry.message_type

            # If already registered, merge handlers (fan-out support)
            if message_type in self._entries:
                existing = self._entries[message_type]
                # Validate constraints match
                if existing.allowed_categories != entry.allowed_categories:
                    raise MessageTypeRegistryError(
                        f"Category constraint mismatch for message type "
                        f"'{message_type}': existing={existing.allowed_categories}, "
                        f"new={entry.allowed_categories}. "
                        f"All registrations for a message type must have the same "
                        f"allowed categories.",
                        message_type=message_type,
                    )
                if (
                    existing.domain_constraint.owning_domain
                    != entry.domain_constraint.owning_domain
                ):
                    raise MessageTypeRegistryError(
                        f"Domain constraint mismatch for message type "
                        f"'{message_type}': existing domain="
                        f"'{existing.domain_constraint.owning_domain}', "
                        f"new domain='{entry.domain_constraint.owning_domain}'. "
                        f"All registrations for a message type must have the same "
                        f"owning domain.",
                        message_type=message_type,
                        domain=entry.domain_constraint.owning_domain,
                    )

                # Merge handlers
                for handler_id in entry.handler_ids:
                    if handler_id not in existing.handler_ids:
                        self._entries[message_type] = existing.with_additional_handler(
                            handler_id
                        )
                        self._handler_references.add(handler_id)
                        existing = self._entries[message_type]

                self._logger.debug(
                    "Merged handlers for message type '%s': %s",
                    message_type,
                    self._entries[message_type].handler_ids,
                )
            else:
                # New registration
                self._entries[message_type] = entry

                # Track domain
                domain = entry.domain_constraint.owning_domain
                self._domains.add(domain)

                # Track handler references
                for handler_id in entry.handler_ids:
                    self._handler_references.add(handler_id)

                # Update indexes
                for category in entry.allowed_categories:
                    self._category_index[category].append(message_type)
                self._domain_index[domain].append(message_type)

                self._logger.debug(
                    "Registered message type '%s' with handlers %s "
                    "(domain=%s, categories=%s)",
                    message_type,
                    entry.handler_ids,
                    domain,
                    [c.value for c in entry.allowed_categories],
                )

    def register_simple(
        self,
        message_type: str,
        handler_id: str,
        category: EnumMessageCategory,
        domain: str,
        *,
        description: str | None = None,
        allow_cross_domains: frozenset[str] | None = None,
    ) -> None:
        """
        Convenience method to register a message type with minimal parameters.

        Creates a ModelMessageTypeEntry internally with sensible defaults.

        Args:
            message_type: The message type name (e.g., "UserCreated").
            handler_id: The handler ID to process this type.
            category: The allowed message category.
            domain: The owning domain.
            description: Optional description of the message type.
            allow_cross_domains: Optional set of additional domains to allow.

        Raises:
            ModelOnexError: If registry is frozen (INVALID_STATE)
            MessageTypeRegistryError: If validation fails

        Example:
            >>> registry.register_simple(
            ...     message_type="UserCreated",
            ...     handler_id="user-handler",
            ...     category=EnumMessageCategory.EVENT,
            ...     domain="user",
            ...     description="User creation event",
            ... )

        .. versionadded:: 0.5.0
        """
        constraint = ModelDomainConstraint(
            owning_domain=domain,
            allowed_cross_domains=allow_cross_domains or frozenset(),
        )

        entry = ModelMessageTypeEntry(
            message_type=message_type,
            handler_ids=(handler_id,),
            allowed_categories=frozenset([category]),
            domain_constraint=constraint,
            description=description,
        )

        self.register_message_type(entry)

    def freeze(self) -> None:
        """
        Freeze the registry to prevent further modifications.

        Once frozen, registration methods will raise ModelOnexError with
        INVALID_STATE. This enables thread-safe concurrent access during
        the query phase.

        Idempotent: Calling freeze() multiple times has no additional effect.

        Example:
            >>> registry.register_message_type(entry)
            >>> registry.freeze()
            >>> assert registry.is_frozen

        Note:
            This is a one-way operation. There is no unfreeze() method
            by design, as unfreezing would defeat thread-safety guarantees.

        .. versionadded:: 0.5.0
        """
        with self._lock:
            if self._frozen:
                # Idempotent - already frozen
                return

            self._frozen = True
            self._logger.info(
                "MessageTypeRegistry frozen with %d message types across %d domains",
                len(self._entries),
                len(self._domains),
            )

    # =========================================================================
    # Query Methods
    # =========================================================================

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
                message_type=message_type,
                registered_types=registered,
            )

        # Check if entry is enabled
        if not entry.enabled:
            raise MessageTypeRegistryError(
                f"Message type '{message_type}' is registered but disabled.",
                message_type=message_type,
            )

        # Validate category constraint
        is_valid, error_msg = entry.validate_category(topic_category)
        if not is_valid:
            raise MessageTypeRegistryError(
                error_msg or f"Category validation failed for '{message_type}'",
                message_type=message_type,
                category=topic_category,
            )

        # Validate domain constraint
        is_valid, error_msg = entry.domain_constraint.validate_consumption(
            topic_domain,
            message_type,
        )
        if not is_valid:
            raise MessageTypeRegistryError(
                error_msg or f"Domain validation failed for '{message_type}'",
                message_type=message_type,
                domain=topic_domain,
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

        Args:
            message_type: The message type to look up.

        Returns:
            List of handler IDs if registered, None if not found.

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

    # =========================================================================
    # Validation Methods
    # =========================================================================

    def validate_startup(
        self,
        available_handler_ids: set[str] | None = None,
    ) -> list[str]:
        """
        Perform startup-time validation and return any errors.

        Validates registry consistency:
        - All handler references point to available handlers (if provided)
        - No duplicate message types with conflicting constraints
        - Domain constraints are properly configured

        This method should be called after freeze() to ensure fail-fast
        behavior before consumers start processing messages.

        Args:
            available_handler_ids: Optional set of handler IDs that are
                actually registered with the dispatch engine. If provided,
                validates that all referenced handlers exist.

        Returns:
            List of validation error messages. Empty list if valid.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        Example:
            >>> registry.freeze()
            >>> errors = registry.validate_startup(
            ...     available_handler_ids={"user-handler", "order-handler"},
            ... )
            >>> if errors:
            ...     for error in errors:
            ...         print(f"Validation error: {error}")
            ...     raise RuntimeError("Registry validation failed")

        .. versionadded:: 0.5.0
        """
        self._require_frozen("validate_startup")

        errors: list[str] = []

        # Validate handler references if available handlers provided
        if available_handler_ids is not None:
            missing_handlers = self._handler_references - available_handler_ids
            if missing_handlers:
                for handler_id in sorted(missing_handlers):
                    # Find which message types reference this handler
                    referencing_types = [
                        mt
                        for mt, entry in self._entries.items()
                        if handler_id in entry.handler_ids
                    ]
                    errors.append(
                        f"Handler '{handler_id}' is referenced by message types "
                        f"{referencing_types} but is not registered with the "
                        f"dispatch engine."
                    )

        # Validate that enabled entries have at least one handler
        for message_type, entry in self._entries.items():
            if entry.enabled and len(entry.handler_ids) == 0:
                errors.append(
                    f"Message type '{message_type}' is enabled but has no handlers."
                )

        # Validate domain constraints are internally consistent
        for message_type, entry in self._entries.items():
            domain = entry.domain_constraint.owning_domain
            if not domain:
                errors.append(f"Message type '{message_type}' has empty owning_domain.")

        # Log validation result
        if errors:
            self._logger.warning(
                "MessageTypeRegistry startup validation failed with %d errors",
                len(errors),
            )
        else:
            self._logger.info(
                "MessageTypeRegistry startup validation passed "
                "(%d message types, %d handlers, %d domains)",
                len(self._entries),
                len(self._handler_references),
                len(self._domains),
            )

        return errors

    def validate_topic_message_type(
        self,
        topic: str,
        message_type: str,
    ) -> tuple[bool, str | None]:
        """
        Validate that a message type can appear on the given topic.

        Extracts domain and category from the topic and validates against
        the registered constraints for the message type.

        Args:
            topic: The full topic string (e.g., "dev.user.events.v1").
            message_type: The message type to validate.

        Returns:
            Tuple of (is_valid, error_message). If is_valid is True,
            error_message is None.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        Example:
            >>> is_valid, error = registry.validate_topic_message_type(
            ...     topic="dev.user.events.v1",
            ...     message_type="UserCreated",
            ... )
            >>> if not is_valid:
            ...     print(f"Validation failed: {error}")

        .. versionadded:: 0.5.0
        """
        self._require_frozen("validate_topic_message_type")

        # Extract category from topic
        category = EnumMessageCategory.from_topic(topic)
        if category is None:
            return (
                False,
                f"Cannot infer message category from topic '{topic}'. "
                f"Topic must contain .events, .commands, or .intents segment.",
            )

        # Extract domain from topic
        domain = extract_domain_from_topic(topic)
        if domain is None:
            return (
                False,
                f"Cannot extract domain from topic '{topic}'. "
                f"Topic format not recognized.",
            )

        # Look up entry
        entry = self._entries.get(message_type)
        if entry is None:
            return (
                False,
                f"Message type '{message_type}' is not registered.",
            )

        if not entry.enabled:
            return (
                False,
                f"Message type '{message_type}' is registered but disabled.",
            )

        # Validate category
        is_valid, error_msg = entry.validate_category(category)
        if not is_valid:
            return (False, error_msg)

        # Validate domain
        is_valid, error_msg = entry.domain_constraint.validate_consumption(
            domain,
            message_type,
        )
        if not is_valid:
            return (False, error_msg)

        return (True, None)

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
        return self._frozen

    @property
    def entry_count(self) -> int:
        """
        Get the number of registered message type entries.

        Returns:
            Number of registered message types.

        .. versionadded:: 0.5.0
        """
        return len(self._entries)

    @property
    def handler_count(self) -> int:
        """
        Get the number of unique handler IDs referenced.

        Returns:
            Number of unique handlers.

        .. versionadded:: 0.5.0
        """
        return len(self._handler_references)

    @property
    def domain_count(self) -> int:
        """
        Get the number of unique domains.

        Returns:
            Number of unique domains.

        .. versionadded:: 0.5.0
        """
        return len(self._domains)

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _require_frozen(self, method_name: str) -> None:
        """Raise error if registry is not frozen."""
        if not self._frozen:
            raise ModelOnexError(
                message=f"{method_name}() called before freeze(). "
                f"Registration MUST complete and freeze() MUST be called "
                f"before queries. This is required for thread safety.",
                error_code=EnumCoreErrorCode.INVALID_STATE,
            )

    # =========================================================================
    # Dunder Methods
    # =========================================================================

    def __len__(self) -> int:
        """Return the number of registered message types."""
        return len(self._entries)

    def __contains__(self, message_type: str) -> bool:
        """Check if message type is registered using 'in' operator."""
        if not self._frozen:
            return message_type in self._entries
        return self.has_message_type(message_type)

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"MessageTypeRegistry[entries={len(self._entries)}, "
            f"domains={len(self._domains)}, frozen={self._frozen}]"
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        types = sorted(self._entries.keys())[:10]
        type_repr = (
            repr(types) if len(self._entries) <= 10 else f"<{len(self._entries)} types>"
        )
        domains = sorted(self._domains)[:5]
        domain_repr = (
            repr(domains)
            if len(self._domains) <= 5
            else f"<{len(self._domains)} domains>"
        )

        return (
            f"MessageTypeRegistry("
            f"entries={type_repr}, "
            f"domains={domain_repr}, "
            f"frozen={self._frozen})"
        )
