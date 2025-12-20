# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Message Category Enumeration.

Defines the fundamental message categories in ONEX event-driven architecture:
- EVENT: Domain events representing facts that have occurred
- COMMAND: Instructions to perform an action
- INTENT: User intents requiring interpretation and routing
- PROJECTION: State projections for optimized read models

Thread Safety:
    All enums in this module are immutable and thread-safe.
    Enum values can be safely shared across threads without synchronization.
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class EnumMessageCategory(str, Enum):
    """
    Message category classification for ONEX event-driven routing.

    The four fundamental message categories determine how messages are
    processed and routed through the system:

    - **EVENT**: Facts that have occurred (past tense, immutable)
      - Published to event topics (*.events.*)
      - Consumed by reducers and projections
      - Example: UserCreatedEvent, OrderCompletedEvent

    - **COMMAND**: Instructions to perform an action (imperative)
      - Published to command topics (*.commands.*)
      - Consumed by command handlers
      - Example: CreateUserCommand, ProcessPaymentCommand

    - **INTENT**: User intents requiring interpretation (declarative)
      - Published to intent topics (*.intents.*)
      - Consumed by orchestrators for routing decisions
      - Example: UserWantsToCheckoutIntent, RequestPasswordResetIntent

    - **PROJECTION**: State projections for optimized read models
      - Used by reducers for state consolidation
      - Example: OrderSummaryProjection, UserProfileProjection

    Attributes:
        topic_suffix: The plural suffix used in topic names (e.g., "events")

    Example:
        >>> category = EnumMessageCategory.EVENT
        >>> category.topic_suffix
        'events'
        >>> EnumMessageCategory.from_topic("onex.user.events")
        <EnumMessageCategory.EVENT: 'event'>
        >>> EnumMessageCategory.from_topic("dev.order.commands.v1")
        <EnumMessageCategory.COMMAND: 'command'>
    """

    EVENT = "event"
    """Domain events representing facts that have occurred."""

    COMMAND = "command"
    """Instructions to perform an action."""

    INTENT = "intent"
    """User intents requiring interpretation and routing."""

    PROJECTION = "projection"
    """State projections for optimized read models."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    @property
    def topic_suffix(self) -> str:
        """
        Get the topic suffix for this category (plural form).

        The suffix is used in topic naming conventions:
        - EVENT -> "events" (e.g., onex.user.events)
        - COMMAND -> "commands" (e.g., onex.order.commands)
        - INTENT -> "intents" (e.g., onex.checkout.intents)
        - PROJECTION -> "projections" (e.g., onex.order.projections)

        Returns:
            The plural suffix string for topic names

        Example:
            >>> EnumMessageCategory.EVENT.topic_suffix
            'events'
            >>> EnumMessageCategory.COMMAND.topic_suffix
            'commands'
        """
        return _CATEGORY_TO_SUFFIX[self]

    @classmethod
    def from_topic(cls, topic: str) -> EnumMessageCategory | None:
        """
        Infer the message category from a topic string.

        Examines the topic for category keywords (events, commands, intents, projections)
        as complete segments and returns the corresponding category. Handles both
        ONEX Kafka format (onex.<domain>.<type>) and Environment-Aware format
        (<env>.<domain>.<category>.<version>).

        The matching is segment-based to prevent false positives. For example:
        - "onex.user.events" matches EVENT (segment "events" exists)
        - "dev.eventsource.data.v1" does NOT match ("eventsource" != "events")

        Args:
            topic: The topic string to analyze

        Returns:
            EnumMessageCategory if a category can be inferred, None otherwise

        Example:
            >>> EnumMessageCategory.from_topic("onex.user.events")
            <EnumMessageCategory.EVENT: 'event'>
            >>> EnumMessageCategory.from_topic("dev.order.commands.v1")
            <EnumMessageCategory.COMMAND: 'command'>
            >>> EnumMessageCategory.from_topic("prod.checkout.intents.v2")
            <EnumMessageCategory.INTENT: 'intent'>
            >>> EnumMessageCategory.from_topic("invalid.topic")
            None
            >>> EnumMessageCategory.from_topic("dev.eventsource.data.v1")
            None  # No false positive: "eventsource" is not the same as "events"
        """
        if not topic:
            return None

        # Split topic into segments and check for exact category suffix matches.
        # This prevents false positives where a segment merely contains the
        # category suffix as a substring (e.g., "eventsource" containing "events").
        segments = topic.lower().split(".")
        for suffix, category in _SUFFIX_TO_CATEGORY.items():
            if suffix in segments:
                return category

        return None

    @classmethod
    def from_suffix(cls, suffix: str) -> EnumMessageCategory | None:
        """
        Get the category from a topic suffix.

        Args:
            suffix: The suffix to look up (e.g., "events", "commands", "intents", "projections")

        Returns:
            EnumMessageCategory if the suffix is valid, None otherwise

        Example:
            >>> EnumMessageCategory.from_suffix("events")
            <EnumMessageCategory.EVENT: 'event'>
            >>> EnumMessageCategory.from_suffix("commands")
            <EnumMessageCategory.COMMAND: 'command'>
            >>> EnumMessageCategory.from_suffix("unknown")
            None
        """
        return _SUFFIX_TO_CATEGORY.get(suffix.lower())

    def is_event(self) -> bool:
        """Check if this is an event category."""
        return self == EnumMessageCategory.EVENT

    def is_command(self) -> bool:
        """Check if this is a command category."""
        return self == EnumMessageCategory.COMMAND

    def is_intent(self) -> bool:
        """Check if this is an intent category."""
        return self == EnumMessageCategory.INTENT

    def is_projection(self) -> bool:
        """Check if this is a projection category."""
        return self == EnumMessageCategory.PROJECTION


# Module-level constant mappings for better performance (avoid dict creation per call)
# Defined after enum class to enable proper type resolution
_CATEGORY_TO_SUFFIX: dict[EnumMessageCategory, str] = {
    EnumMessageCategory.EVENT: "events",
    EnumMessageCategory.COMMAND: "commands",
    EnumMessageCategory.INTENT: "intents",
    EnumMessageCategory.PROJECTION: "projections",
}
_SUFFIX_TO_CATEGORY: dict[str, EnumMessageCategory] = {
    "events": EnumMessageCategory.EVENT,
    "commands": EnumMessageCategory.COMMAND,
    "intents": EnumMessageCategory.INTENT,
    "projections": EnumMessageCategory.PROJECTION,
}


__all__ = ["EnumMessageCategory"]
