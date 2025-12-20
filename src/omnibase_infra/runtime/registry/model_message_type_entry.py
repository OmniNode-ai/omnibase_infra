# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Message Type Entry Model.

Defines the registry entry for message type to handler mappings in the
Central Message Type Registry.

Design Principles:
    - Each entry maps a message type to one or more handler implementations
    - Topic category constraints define where message types can appear
    - Domain ownership is tracked for cross-domain validation
    - Immutable entries for thread-safe concurrent access

Related:
    - OMN-937: Central Message Type Registry implementation
    - ModelDomainConstraint: Domain ownership rules
    - MessageTypeRegistry: Uses entries for handler resolution

.. versionadded:: 0.5.0
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.registry.model_domain_constraint import (
    ModelDomainConstraint,
)


class ModelMessageTypeEntry(BaseModel):
    """
    Registry entry mapping a message type to handler(s) with constraints.

    Each entry represents a message type registration in the Central Message Type
    Registry. It defines which handlers process the message type, what categories
    are allowed, and domain ownership constraints.

    Attributes:
        message_type: The fully-qualified message type name.
            Example: "UserCreated", "CreateOrderCommand", "CheckoutIntent"
        handler_ids: List of handler IDs that can process this message type.
            Supports fan-out pattern with multiple handlers per type.
        allowed_categories: Message categories where this type can appear.
            Example: [EVENT] for domain events, [COMMAND] for commands.
        domain_constraint: Domain ownership and cross-domain rules.
        description: Human-readable description of the message type.
        enabled: Whether this entry is active. Disabled entries are skipped.
        registered_at: Timestamp when this entry was registered.
        metadata: Optional additional metadata.

    Fan-Out Support:
        Multiple handler_ids can be registered for the same message type to enable
        fan-out processing patterns. Each registered handler will receive the
        message independently.

    Example:
        >>> entry = ModelMessageTypeEntry(
        ...     message_type="UserCreated",
        ...     handler_ids=["user-event-handler", "audit-logger"],
        ...     allowed_categories=[EnumMessageCategory.EVENT],
        ...     domain_constraint=ModelDomainConstraint(owning_domain="user"),
        ...     description="User creation event",
        ... )
        >>> entry.supports_category(EnumMessageCategory.EVENT)
        True
        >>> entry.supports_category(EnumMessageCategory.COMMAND)
        False

    Thread Safety:
        This model is immutable (frozen=True) and thread-safe for concurrent access.

    .. versionadded:: 0.5.0
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # ---- Message Type Identity ----
    message_type: str = Field(
        ...,
        description="The fully-qualified message type name.",
        min_length=1,
        max_length=200,
    )

    # ---- Handler Mapping ----
    handler_ids: tuple[str, ...] = Field(
        ...,
        description=(
            "Handler IDs that can process this message type. "
            "Tuple for immutability and fan-out support."
        ),
        min_length=1,
    )

    # ---- Category Constraints ----
    allowed_categories: frozenset[EnumMessageCategory] = Field(
        ...,
        description=(
            "Message categories where this type can appear. Frozenset for immutability."
        ),
    )

    # ---- Domain Constraints ----
    domain_constraint: ModelDomainConstraint = Field(
        ...,
        description="Domain ownership and cross-domain consumption rules.",
    )

    # ---- Metadata ----
    description: str | None = Field(
        default=None,
        description="Human-readable description of the message type.",
        max_length=500,
    )

    enabled: bool = Field(
        default=True,
        description="Whether this entry is active. Disabled entries are skipped.",
    )

    registered_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when this entry was registered (UTC).",
    )

    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional additional metadata.",
    )

    def supports_category(self, category: EnumMessageCategory) -> bool:
        """
        Check if this message type is allowed in the given category.

        Args:
            category: The message category to check.

        Returns:
            True if the message type can appear in this category, False otherwise.

        Example:
            >>> entry = ModelMessageTypeEntry(
            ...     message_type="UserCreated",
            ...     handler_ids=("handler-1",),
            ...     allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            ...     domain_constraint=ModelDomainConstraint(owning_domain="user"),
            ... )
            >>> entry.supports_category(EnumMessageCategory.EVENT)
            True

        .. versionadded:: 0.5.0
        """
        return category in self.allowed_categories

    def validate_category(
        self,
        topic_category: EnumMessageCategory,
    ) -> tuple[bool, str | None]:
        """
        Validate if the message type can appear in the given topic category.

        Args:
            topic_category: The category inferred from the topic.

        Returns:
            Tuple of (is_valid, error_message). If is_valid is True,
            error_message is None.

        Example:
            >>> entry = ModelMessageTypeEntry(
            ...     message_type="UserCreated",
            ...     handler_ids=("handler-1",),
            ...     allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            ...     domain_constraint=ModelDomainConstraint(owning_domain="user"),
            ... )
            >>> entry.validate_category(EnumMessageCategory.COMMAND)
            (False, "Message type 'UserCreated' is not allowed in category ...")

        .. versionadded:: 0.5.0
        """
        if self.supports_category(topic_category):
            return (True, None)

        allowed_str = ", ".join(sorted(c.value for c in self.allowed_categories))
        return (
            False,
            f"Message type '{self.message_type}' is not allowed in category "
            f"'{topic_category.value}'. Allowed categories: [{allowed_str}].",
        )

    def with_additional_handler(self, handler_id: str) -> "ModelMessageTypeEntry":
        """
        Create a new entry with an additional handler ID.

        Used for fan-out registration where multiple handlers process
        the same message type.

        Args:
            handler_id: The handler ID to add.

        Returns:
            New ModelMessageTypeEntry with the additional handler.

        Example:
            >>> entry = ModelMessageTypeEntry(
            ...     message_type="UserCreated",
            ...     handler_ids=("handler-1",),
            ...     allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            ...     domain_constraint=ModelDomainConstraint(owning_domain="user"),
            ... )
            >>> updated = entry.with_additional_handler("handler-2")
            >>> updated.handler_ids
            ('handler-1', 'handler-2')

        .. versionadded:: 0.5.0
        """
        if handler_id in self.handler_ids:
            return self

        new_handlers = tuple(list(self.handler_ids) + [handler_id])
        return self.model_copy(update={"handler_ids": new_handlers})

    def with_enabled(self, enabled: bool) -> "ModelMessageTypeEntry":
        """
        Create a new entry with updated enabled status.

        Args:
            enabled: The new enabled status.

        Returns:
            New ModelMessageTypeEntry with updated status.

        .. versionadded:: 0.5.0
        """
        return self.model_copy(update={"enabled": enabled})


__all__ = ["ModelMessageTypeEntry"]
