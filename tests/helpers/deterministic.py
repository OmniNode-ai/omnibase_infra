# Copyright (c) 2025 OmniNode.ai
# SPDX-License-Identifier: Apache-2.0
"""Deterministic test utilities for predictable testing.

This module provides utilities that generate deterministic values for testing,
enabling reproducible test behavior and eliminating flakiness from random
UUID generation or time-dependent logic.

Example usage:
    >>> from tests.helpers.deterministic import (
    ...     DeterministicIdGenerator, DeterministicClock
    ... )
    >>>
    >>> # Predictable UUID generation
    >>> id_gen = DeterministicIdGenerator(seed=100)
    >>> uuid1 = id_gen.next_uuid()
    >>> uuid2 = id_gen.next_uuid()
    >>> assert uuid1 != uuid2  # Different UUIDs
    >>> assert uuid1 == UUID(int=101)  # Predictable value
    >>>
    >>> # Controllable time
    >>> clock = DeterministicClock()
    >>> t1 = clock.now()
    >>> clock.advance(60)
    >>> t2 = clock.now()
    >>> assert (t2 - t1).total_seconds() == 60
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID

__all__ = [
    "DeterministicClock",
    "DeterministicIdGenerator",
]


class DeterministicIdGenerator:
    """Generates predictable UUIDs for testing.

    This class provides deterministic UUID generation based on an incrementing
    counter, enabling reproducible test assertions on generated identifiers.

    The generated UUIDs are valid UUID objects created from integer values,
    making them suitable for database operations and API responses.

    Attributes:
        _counter: Internal counter used to generate sequential UUID values.

    Example:
        >>> gen = DeterministicIdGenerator(seed=0)
        >>> gen.next_uuid()
        UUID('00000000-0000-0000-0000-000000000001')
        >>> gen.next_uuid()
        UUID('00000000-0000-0000-0000-000000000002')
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize the deterministic ID generator.

        Args:
            seed: Starting value for the internal counter. The first call to
                next_uuid() will return a UUID based on seed + 1. Defaults to 42.
        """
        self._counter: int = seed

    def next_uuid(self) -> UUID:
        """Generate the next deterministic UUID.

        Increments the internal counter and returns a UUID constructed from
        that integer value. Each call returns a unique, predictable UUID.

        Returns:
            A UUID object with a predictable integer value.

        Example:
            >>> gen = DeterministicIdGenerator(seed=100)
            >>> uuid = gen.next_uuid()
            >>> uuid.int
            101
        """
        self._counter += 1
        return UUID(int=self._counter)

    def reset(self, seed: int = 42) -> None:
        """Reset the counter to a specified seed value.

        Useful for resetting state between test cases to ensure
        reproducible behavior.

        Args:
            seed: The value to reset the counter to. Defaults to 42.
        """
        self._counter = seed

    @property
    def current_counter(self) -> int:
        """Return the current counter value.

        This is useful for debugging or verifying generator state in tests.

        Returns:
            The current integer counter value.
        """
        return self._counter


class DeterministicClock:
    """Provides controllable timestamps for testing.

    This class simulates a clock that can be manually controlled, enabling
    tests to verify time-dependent behavior without relying on real system time.

    The clock starts at a configurable initial time and can be advanced
    by specified durations, making it easy to test timeout logic,
    expiration behavior, and time-based ordering.

    Attributes:
        _now: The current simulated time.

    Example:
        >>> clock = DeterministicClock()
        >>> start = clock.now()
        >>> clock.advance(3600)  # Advance 1 hour
        >>> end = clock.now()
        >>> (end - start).total_seconds()
        3600.0
    """

    def __init__(self, start: datetime | None = None) -> None:
        """Initialize the deterministic clock.

        Args:
            start: The initial time for the clock. If None, defaults to
                2024-01-01 00:00:00 UTC. The datetime should be timezone-aware
                for consistency; naive datetimes are accepted but may cause
                comparison issues with timezone-aware datetimes.
        """
        if start is None:
            start = datetime(2024, 1, 1, tzinfo=UTC)
        self._now: datetime = start

    def now(self) -> datetime:
        """Return the current simulated time.

        Returns:
            The current datetime value of the simulated clock.

        Example:
            >>> clock = DeterministicClock(start=datetime(2024, 6, 15, 12, 0, 0))
            >>> clock.now()
            datetime.datetime(2024, 6, 15, 12, 0)
        """
        return self._now

    def advance(self, seconds: int) -> None:
        """Advance the clock by the specified number of seconds.

        Args:
            seconds: The number of seconds to advance. Can be negative
                to move the clock backward (useful for testing edge cases).

        Example:
            >>> clock = DeterministicClock()
            >>> t1 = clock.now()
            >>> clock.advance(120)
            >>> t2 = clock.now()
            >>> (t2 - t1).total_seconds()
            120.0
        """
        self._now += timedelta(seconds=seconds)

    def advance_minutes(self, minutes: int) -> None:
        """Advance the clock by the specified number of minutes.

        Convenience method for advancing by minutes instead of seconds.

        Args:
            minutes: The number of minutes to advance.
        """
        self._now += timedelta(minutes=minutes)

    def advance_hours(self, hours: int) -> None:
        """Advance the clock by the specified number of hours.

        Convenience method for advancing by hours instead of seconds.

        Args:
            hours: The number of hours to advance.
        """
        self._now += timedelta(hours=hours)

    def set_time(self, new_time: datetime) -> None:
        """Set the clock to a specific time.

        Useful for jumping to a specific point in time during tests.

        Args:
            new_time: The datetime to set the clock to.
        """
        self._now = new_time

    def reset(self, start: datetime | None = None) -> None:
        """Reset the clock to its initial or a specified time.

        Args:
            start: The time to reset to. If None, resets to the default
                start time (2024-01-01 00:00:00 UTC).
        """
        if start is None:
            start = datetime(2024, 1, 1, tzinfo=UTC)
        self._now = start
