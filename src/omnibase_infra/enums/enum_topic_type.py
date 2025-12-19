# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Topic Type Enumeration.

Defines the valid topic types in ONEX topic taxonomy:
- EVENTS: Event topics for domain events
- COMMANDS: Command topics for action instructions
- INTENTS: Intent topics for user intents
- SNAPSHOTS: Snapshot topics for state snapshots

Thread Safety:
    All enums in this module are immutable and thread-safe.
    Enum values can be safely shared across threads without synchronization.
"""

from enum import Enum, unique


@unique
class EnumTopicType(str, Enum):
    """
    Topic type classification for ONEX topic taxonomy.

    Represents the valid topic suffixes in the ONEX naming convention.
    These types define what kind of messages flow through a topic.

    Values:
        EVENTS: Topic for domain events (facts that occurred)
        COMMANDS: Topic for commands (action instructions)
        INTENTS: Topic for intents (user intentions)
        SNAPSHOTS: Topic for state snapshots (materialized views)

    Example:
        >>> topic_type = EnumTopicType.EVENTS
        >>> str(topic_type)
        'events'
        >>> EnumTopicType.from_suffix("commands")
        <EnumTopicType.COMMANDS: 'commands'>
    """

    EVENTS = "events"
    """Topic for domain events representing facts that have occurred."""

    COMMANDS = "commands"
    """Topic for commands representing action instructions."""

    INTENTS = "intents"
    """Topic for intents representing user intentions."""

    SNAPSHOTS = "snapshots"
    """Topic for state snapshots and materialized views."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    @classmethod
    def from_suffix(cls, suffix: str) -> "EnumTopicType | None":
        """
        Get the topic type from a suffix string.

        Args:
            suffix: The suffix to look up (e.g., "events", "commands")

        Returns:
            EnumTopicType if the suffix is valid, None otherwise

        Example:
            >>> EnumTopicType.from_suffix("events")
            <EnumTopicType.EVENTS: 'events'>
            >>> EnumTopicType.from_suffix("commands")
            <EnumTopicType.COMMANDS: 'commands'>
            >>> EnumTopicType.from_suffix("unknown")
            None
        """
        suffix_lower = suffix.lower()
        for topic_type in cls:
            if topic_type.value == suffix_lower:
                return topic_type
        return None

    @classmethod
    def is_valid_suffix(cls, suffix: str) -> bool:
        """
        Check if a suffix is a valid topic type.

        Args:
            suffix: The suffix to validate

        Returns:
            True if the suffix is valid, False otherwise

        Example:
            >>> EnumTopicType.is_valid_suffix("events")
            True
            >>> EnumTopicType.is_valid_suffix("invalid")
            False
        """
        return cls.from_suffix(suffix) is not None

    def is_event_type(self) -> bool:
        """Check if this is an events topic type."""
        return self == EnumTopicType.EVENTS

    def is_command_type(self) -> bool:
        """Check if this is a commands topic type."""
        return self == EnumTopicType.COMMANDS

    def is_intent_type(self) -> bool:
        """Check if this is an intents topic type."""
        return self == EnumTopicType.INTENTS

    def is_snapshot_type(self) -> bool:
        """Check if this is a snapshots topic type."""
        return self == EnumTopicType.SNAPSHOTS


__all__ = ["EnumTopicType"]
