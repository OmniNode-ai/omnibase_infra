# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatcher Registration Model.

Represents metadata about a registered dispatcher in the dispatch engine.
Dispatchers are the execution units that process messages after routing.

Design Pattern:
    ModelDispatcherRegistration is a pure data model that captures dispatcher metadata:
    - Dispatcher identity (unique ID, human-readable name)
    - Dispatcher capabilities (what message categories/types it can handle)
    - Dispatcher configuration (timeout, concurrency limits)
    - Health and status information

    This model is used by the dispatch engine to:
    1. Register dispatchers during startup
    2. Validate that routes reference valid dispatchers
    3. Track dispatcher health and availability
    4. Configure dispatcher execution parameters

Thread Safety:
    ModelDispatcherRegistration is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from omnibase_infra.models.dispatch import ModelDispatcherRegistration
    >>> from omnibase_infra.enums import EnumMessageCategory
    >>> from omnibase_core.enums.enum_node_kind import EnumNodeKind
    >>> from datetime import datetime, timezone
    >>>
    >>> # Register a dispatcher for user events
    >>> dispatcher = ModelDispatcherRegistration(
    ...     dispatcher_id="user-event-dispatcher",
    ...     dispatcher_name="User Event Dispatcher",
    ...     node_kind=EnumNodeKind.REDUCER,
    ...     supported_categories=[EnumMessageCategory.EVENT],
    ...     timeout_seconds=30,
    ...     max_concurrent=10,
    ...     registered_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    ... )

See Also:
    omnibase_infra.models.dispatch.ModelDispatchRoute: Uses dispatcher_id to reference dispatchers
    omnibase_infra.models.dispatch.ModelDispatchResult: Reports dispatcher execution results
    omnibase_core.enums.EnumNodeKind: Node type classification
"""

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_metadata import ModelDispatchMetadata


class ModelDispatcherRegistration(BaseModel):
    """
    Metadata about a registered dispatcher in the dispatch engine.

    Captures all information needed to configure and manage a dispatcher,
    including its capabilities, configuration, and health status.

    Attributes:
        dispatcher_id: Unique identifier for this dispatcher (referenced by routes).
        dispatcher_name: Human-readable name for the dispatcher.
        node_kind: The ONEX node kind this dispatcher represents.
        node_id: Optional UUID of the node instance implementing this dispatcher.
        supported_categories: Message categories this dispatcher can process.
        supported_message_types: Specific message types this dispatcher accepts.
            When empty, accepts all message types in supported categories.
        timeout_seconds: Maximum execution time before timeout.
        max_concurrent: Maximum concurrent executions allowed.
        max_retries: Maximum retry attempts for failed executions.
        enabled: Whether this dispatcher is currently accepting messages.
        healthy: Whether this dispatcher is currently healthy.
        last_health_check: Timestamp of the last health check.
        registered_at: Timestamp when this dispatcher was registered.
        version: Dispatcher version string for compatibility tracking.
        description: Human-readable description of the dispatcher's purpose.
        tags: Optional tags for categorization and filtering.
        metadata: Optional additional metadata about the dispatcher.

    Example:
        >>> from datetime import datetime, timezone
        >>> dispatcher = ModelDispatcherRegistration(
        ...     dispatcher_id="order-processor",
        ...     dispatcher_name="Order Processing Dispatcher",
        ...     node_kind=EnumNodeKind.ORCHESTRATOR,
        ...     supported_categories=[EnumMessageCategory.COMMAND],
        ...     supported_message_types=["CreateOrderCommand", "CancelOrderCommand"],
        ...     timeout_seconds=60,
        ...     max_concurrent=5,
        ...     description="Processes order commands and coordinates fulfillment",
        ...     registered_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # ---- Dispatcher Identity ----
    dispatcher_id: str = Field(
        ...,
        description="Unique identifier for this dispatcher (referenced by routes).",
        min_length=1,
        max_length=200,
    )
    dispatcher_name: str = Field(
        ...,
        description="Human-readable name for the dispatcher.",
        min_length=1,
        max_length=200,
    )

    # ---- Node Information ----
    node_kind: EnumNodeKind = Field(
        ...,
        description="The ONEX node kind this dispatcher represents.",
    )
    node_id: UUID | None = Field(
        default=None,
        description="Optional UUID of the node instance implementing this dispatcher.",
    )

    # ---- Capabilities ----
    supported_categories: list[EnumMessageCategory] = Field(
        ...,
        description="Message categories this dispatcher can process.",
        min_length=1,
    )
    supported_message_types: list[str] = Field(
        default_factory=list,
        description=(
            "Specific message types this dispatcher accepts. "
            "When empty, accepts all message types in supported categories."
        ),
    )

    # ---- Execution Configuration ----
    timeout_seconds: int = Field(
        default=30,
        description="Maximum execution time before timeout.",
        ge=1,
        le=3600,
    )
    max_concurrent: int = Field(
        default=10,
        description="Maximum concurrent executions allowed.",
        ge=1,
        le=1000,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed executions.",
        ge=0,
        le=10,
    )

    # ---- Status ----
    enabled: bool = Field(
        default=True,
        description="Whether this dispatcher is currently accepting messages.",
    )
    healthy: bool = Field(
        default=True,
        description="Whether this dispatcher is currently healthy.",
    )
    last_health_check: datetime | None = Field(
        default=None,
        description="Timestamp of the last health check (UTC).",
    )

    # ---- Registration Metadata ----
    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    registered_at: datetime = Field(
        ...,
        description="Timestamp when this dispatcher was registered (UTC, must be explicitly provided).",
    )
    version: str | None = Field(
        default=None,
        description="Dispatcher version string for compatibility tracking.",
        max_length=50,
    )

    # ---- Documentation ----
    description: str | None = Field(
        default=None,
        description="Human-readable description of the dispatcher's purpose.",
        max_length=1000,
    )

    # ---- Optional Metadata ----
    tags: list[str] | None = Field(
        default=None,
        description="Optional tags for categorization and filtering.",
    )
    metadata: ModelDispatchMetadata | None = Field(
        default=None,
        description="Optional additional metadata about the dispatcher.",
    )

    def supports_category(self, category: EnumMessageCategory) -> bool:
        """
        Check if this dispatcher supports the given message category.

        Args:
            category: The message category to check

        Returns:
            True if the dispatcher supports this category, False otherwise

        Example:
            >>> from datetime import datetime, timezone
            >>> dispatcher = ModelDispatcherRegistration(
            ...     dispatcher_id="test",
            ...     dispatcher_name="Test Dispatcher",
            ...     node_kind=EnumNodeKind.REDUCER,
            ...     supported_categories=[EnumMessageCategory.EVENT],
            ...     registered_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            ... )
            >>> dispatcher.supports_category(EnumMessageCategory.EVENT)
            True
            >>> dispatcher.supports_category(EnumMessageCategory.COMMAND)
            False
        """
        return category in self.supported_categories

    def supports_message_type(self, message_type: str) -> bool:
        """
        Check if this dispatcher supports the given message type.

        If supported_message_types is empty, accepts all message types.

        Args:
            message_type: The message type to check

        Returns:
            True if the dispatcher supports this message type, False otherwise

        Example:
            >>> from datetime import datetime, timezone
            >>> dispatcher = ModelDispatcherRegistration(
            ...     dispatcher_id="test",
            ...     dispatcher_name="Test Dispatcher",
            ...     node_kind=EnumNodeKind.REDUCER,
            ...     supported_categories=[EnumMessageCategory.EVENT],
            ...     supported_message_types=["UserCreated", "UserUpdated"],
            ...     registered_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            ... )
            >>> dispatcher.supports_message_type("UserCreated")
            True
            >>> dispatcher.supports_message_type("OrderCreated")
            False
        """
        if not self.supported_message_types:
            # Empty list means all message types are supported
            return True
        return message_type in self.supported_message_types

    def can_accept_message(
        self,
        category: EnumMessageCategory,
        message_type: str | None = None,
    ) -> bool:
        """
        Check if this dispatcher can accept a message with the given category and type.

        Args:
            category: The message category
            message_type: Optional message type

        Returns:
            True if the dispatcher can accept this message, False otherwise

        Example:
            >>> from datetime import datetime, timezone
            >>> dispatcher = ModelDispatcherRegistration(
            ...     dispatcher_id="test",
            ...     dispatcher_name="Test Dispatcher",
            ...     node_kind=EnumNodeKind.REDUCER,
            ...     supported_categories=[EnumMessageCategory.EVENT],
            ...     supported_message_types=["UserCreated"],
            ...     registered_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            ... )
            >>> dispatcher.can_accept_message(EnumMessageCategory.EVENT, "UserCreated")
            True
            >>> dispatcher.can_accept_message(EnumMessageCategory.EVENT, "OrderCreated")
            False
        """
        if not self.enabled:
            return False
        if not self.healthy:
            return False
        if not self.supports_category(category):
            return False
        if message_type is not None and not self.supports_message_type(message_type):
            return False
        return True

    def is_available(self) -> bool:
        """
        Check if this dispatcher is available to accept messages.

        A dispatcher is available if it is both enabled and healthy.

        Returns:
            True if the dispatcher is available, False otherwise
        """
        return self.enabled and self.healthy

    def with_health_status(
        self,
        healthy: bool,
        check_time: datetime | None = None,
    ) -> "ModelDispatcherRegistration":
        """
        Create a new registration with updated health status.

        Args:
            healthy: The new health status
            check_time: Optional timestamp for the health check (defaults to now)

        Returns:
            New ModelDispatcherRegistration with updated health status
        """
        return self.model_copy(
            update={
                "healthy": healthy,
                "last_health_check": check_time or datetime.now(UTC),
            }
        )

    def with_enabled(self, enabled: bool) -> "ModelDispatcherRegistration":
        """
        Create a new registration with updated enabled status.

        Args:
            enabled: The new enabled status

        Returns:
            New ModelDispatcherRegistration with updated enabled status
        """
        return self.model_copy(update={"enabled": enabled})


__all__ = ["ModelDispatcherRegistration"]
