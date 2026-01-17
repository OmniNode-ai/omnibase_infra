# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Central Message Type Registry Implementation.

Provides the RegistryMessageType class that maps message types to handler
implementations and enforces topic category constraints and domain ownership.

Design Principles:
    - Freeze-after-init pattern for thread-safe concurrent access
    - Startup-time validation with fail-fast behavior
    - Domain ownership enforcement derived from topic names
    - Clear error messages for configuration issues
    - Extensibility for new domains

Thread Safety:
    RegistryMessageType follows the freeze-after-init pattern:
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
    "RegistryMessageType",
    "MessageTypeRegistryError",
    "extract_domain_from_topic",
]

import logging
import re

from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.errors import MessageTypeRegistryError
from omnibase_infra.models.validation import ModelValidationOutcome
from omnibase_infra.runtime.registry.mixin_message_type_query import (
    MixinMessageTypeQuery,
)
from omnibase_infra.runtime.registry.mixin_message_type_registration import (
    MixinMessageTypeRegistration,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Topic Domain Extraction
# =============================================================================

# Pattern for ONEX format: onex.<domain>.<type>
# Example: onex.registration.events -> domain = "registration"
_ONEX_TOPIC_PATTERN = re.compile(
    r"^onex\.(?P<domain>[a-zA-Z0-9_-]+)\.",
    re.IGNORECASE,
)

# Pattern for environment-aware format: <env>.<domain>.<category>.<version>
# Example: dev.user.events.v1 -> domain = "user"
# Requires at least 3 segments (env.domain.rest) with trailing content
_ENV_TOPIC_PATTERN = re.compile(
    r"^(?P<env>[a-zA-Z0-9_-]+)\.(?P<domain>[a-zA-Z0-9_-]+)\.",
    re.IGNORECASE,
)


def extract_domain_from_topic(topic: str | None) -> str | None:
    """
    Extract the domain from a topic string.

    Domain extraction follows ONEX topic naming conventions:
    - ONEX Kafka format: "onex.<domain>.<type>" -> domain
    - Environment-aware format: "<env>.<domain>.<category>.<version>" -> domain

    Args:
        topic: The topic string to extract domain from. May be None, empty,
            or whitespace-only.

    Returns:
        The extracted domain string, or None if:
        - topic is None
        - topic is empty string
        - topic is whitespace-only
        - topic format is not recognized (no domain extractable)

    Examples:
        >>> extract_domain_from_topic("onex.registration.events")
        'registration'
        >>> extract_domain_from_topic("dev.user.events.v1")
        'user'
        >>> extract_domain_from_topic("prod.order.commands.v2")
        'order'
        >>> extract_domain_from_topic("invalid")
        None
        >>> extract_domain_from_topic(None)
        None
        >>> extract_domain_from_topic("")
        None
        >>> extract_domain_from_topic("   ")
        None

    Design Note:
        **Pattern Matching Order is Critical for Correctness**

        The ONEX pattern MUST be checked BEFORE the environment-aware pattern.
        This ordering is intentional and required to correctly handle topics
        where "onex" appears as the first segment.

        Consider the ambiguous topic ``onex.prod.events``:

        - **If env pattern checked first (WRONG)**:
          ``<env>.<domain>.<rest>`` would match with env="onex", domain="prod"
          This incorrectly interprets "onex" as an environment name.

        - **With ONEX pattern first (CORRECT)**:
          ``onex.<domain>.<type>`` matches with domain="prod"
          This correctly recognizes "onex" as the ONEX namespace prefix.

        The ONEX pattern is more specific (requires literal "onex" prefix),
        so it must take precedence over the generic environment pattern.

    .. versionadded:: 0.5.0
    """
    # Handle None, empty string, and whitespace-only inputs
    if topic is None or not topic.strip():
        return None

    # Strip whitespace for consistent matching
    topic = topic.strip()

    # -------------------------------------------------------------------------
    # CRITICAL: Pattern matching order matters!
    # -------------------------------------------------------------------------
    # The ONEX pattern MUST be checked BEFORE the environment-aware pattern.
    # Both patterns would match topics like "onex.prod.events", but with
    # different (and conflicting) domain extraction:
    #
    #   Topic: "onex.prod.events"
    #   - ONEX pattern:  domain = "prod"  (CORRECT - "onex" is namespace)
    #   - Env pattern:   domain = "prod"  (WRONG - "onex" treated as env)
    #
    # Wait, in this case both extract "prod"! The issue is with interpretation:
    # The env pattern would set env="onex", which is semantically wrong.
    # More critically, consider "onex.user.events" vs "prod.user.events":
    # - "onex.user.events" -> ONEX format, domain="user"
    # - "prod.user.events" -> Env format, env="prod", domain="user"
    #
    # If we checked env pattern first, "onex" would be misclassified as an
    # environment, breaking any code that relies on detecting ONEX-namespaced
    # topics. The ONEX pattern is more specific, so it takes precedence.
    # -------------------------------------------------------------------------

    # Step 1: Try ONEX format first: onex.<domain>.<type>
    match = _ONEX_TOPIC_PATTERN.match(topic)
    if match:
        return match.group("domain")

    # Step 2: Try environment-aware format: <env>.<domain>.<category>.<version>
    # Only reached if ONEX pattern did not match (topic doesn't start with "onex.")
    match = _ENV_TOPIC_PATTERN.match(topic)
    if match:
        return match.group("domain")

    # Fallback: try simple split on dots and take second segment
    # This handles edge cases like "a.b" (no trailing segment)
    parts = topic.split(".")
    if len(parts) >= 2:
        return parts[1]

    return None


# =============================================================================
# Message Type Registry
# =============================================================================


class RegistryMessageType(MixinMessageTypeRegistration, MixinMessageTypeQuery):
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
        ...     RegistryMessageType,
        ...     ModelMessageTypeEntry,
        ...     ModelDomainConstraint,
        ... )
        >>> from omnibase_infra.enums import EnumMessageCategory
        >>>
        >>> # Create registry and register message types
        >>> registry = RegistryMessageType()
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
        ...     from omnibase_infra.errors import ProtocolConfigurationError
        ...     raise ProtocolConfigurationError(
        ...         f"Validation failed: {errors}",
        ...         code="REGISTRY_VALIDATION_FAILED",
        ...     )
        >>>
        >>> # Query handlers (thread-safe after freeze)
        >>> handlers = registry.get_handlers(
        ...     message_type="UserCreated",
        ...     topic_category=EnumMessageCategory.EVENT,
        ...     topic_domain="user",
        ... )

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
        Initialize RegistryMessageType with empty registries.

        Creates empty message type registry. Register message types before
        freeze(). Call validate_startup() after freeze() to ensure fail-fast
        behavior.

        Args:
            logger_instance: Optional custom logger for structured logging.
                If not provided, uses module-level logger.
        """
        self._init_registration_state(logger_instance)

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
            ...     from omnibase_infra.errors import ProtocolConfigurationError
            ...     for error in errors:
            ...         print(f"Validation error: {error}")
            ...     raise ProtocolConfigurationError(
            ...         "Registry validation failed",
            ...         code="REGISTRY_VALIDATION_FAILED",
            ...         errors=errors,
            ...     )

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
                    f"Invalid message type configuration for '{message_type}': "
                    f"expected at least one handler, got 0"
                )

        # Validate domain constraints are internally consistent
        for message_type, entry in self._entries.items():
            domain = entry.domain_constraint.owning_domain
            if not domain:
                errors.append(f"Message type '{message_type}' has empty owning_domain.")

        # Log validation result
        if errors:
            self._logger.warning(
                "RegistryMessageType startup validation failed with %d errors",
                len(errors),
            )
        else:
            self._logger.info(
                "RegistryMessageType startup validation passed"
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
    ) -> ModelValidationOutcome:
        """
        Validate that a message type can appear on the given topic.

        Extracts domain and category from the topic and validates against
        the registered constraints for the message type.

        Args:
            topic: The full topic string (e.g., "dev.user.events.v1").
            message_type: The message type to validate.

        Returns:
            ModelValidationOutcome with is_valid=True if valid,
            or is_valid=False with error_message if invalid.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE)

        Example:
            >>> outcome = registry.validate_topic_message_type(
            ...     topic="dev.user.events.v1",
            ...     message_type="UserCreated",
            ... )
            >>> if not outcome:
            ...     print(f"Validation failed: {outcome.error_message}")

        .. versionadded:: 0.5.0
        .. versionchanged:: 0.6.0
            Return type changed from tuple[bool, str | None] to ModelValidationOutcome.
        """
        self._require_frozen("validate_topic_message_type")

        # Extract category from topic
        category = EnumMessageCategory.from_topic(topic)
        if category is None:
            return ModelValidationOutcome.failure(
                f"Cannot infer message category from topic '{topic}'. "
                f"Topic must contain .events, .commands, or .intents segment."
            )

        # Extract domain from topic
        domain = extract_domain_from_topic(topic)
        if domain is None:
            return ModelValidationOutcome.failure(
                f"Cannot extract domain from topic '{topic}'. "
                f"Topic format not recognized."
            )

        # Look up entry
        entry = self._entries.get(message_type)
        if entry is None:
            return ModelValidationOutcome.failure(
                f"Message type '{message_type}' is not registered."
            )

        if not entry.enabled:
            return ModelValidationOutcome.failure(
                f"Message type '{message_type}' is registered but disabled."
            )

        # Validate category
        category_outcome = entry.validate_category(category)
        if not category_outcome:
            return category_outcome

        # Validate domain
        domain_outcome = entry.domain_constraint.validate_consumption(
            domain,
            message_type,
        )
        if not domain_outcome:
            return domain_outcome

        return ModelValidationOutcome.success()

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
            f"RegistryMessageType[entries={len(self._entries)}, "
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
            f"RegistryMessageType("
            f"entries={type_repr}, "
            f"domains={domain_repr}, "
            f"frozen={self._frozen})"
        )
