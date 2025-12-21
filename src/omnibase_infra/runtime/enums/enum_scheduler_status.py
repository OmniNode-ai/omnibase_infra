# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Scheduler Status Enumeration.

Defines lifecycle status values for the ONEX runtime scheduler.
Used to track the operational state of the scheduler through its lifecycle
from initialization through shutdown.

Thread Safety:
    All enums in this module are immutable and thread-safe.
    Enum values can be safely shared across threads without synchronization.
"""

from enum import StrEnum, unique


@unique
class EnumSchedulerStatus(StrEnum):
    """
    Status values for the runtime scheduler lifecycle.

    Represents the operational state of the scheduler at any given moment.
    The scheduler transitions through these states during its lifecycle:
    STOPPED -> STARTING -> RUNNING -> STOPPING -> STOPPED

    Values:
        STOPPED: Scheduler is not running and has no active tick loop
        STARTING: Scheduler is initializing and preparing to emit ticks
        RUNNING: Scheduler is actively emitting ticks at configured intervals
        STOPPING: Scheduler is gracefully shutting down
        ERROR: Scheduler encountered a fatal error and cannot continue

    Example:
        >>> status = EnumSchedulerStatus.RUNNING
        >>> status.is_active()
        True
        >>> EnumSchedulerStatus.STOPPED.can_start()
        True
        >>> str(EnumSchedulerStatus.STARTING)
        'starting'
    """

    STOPPED = "stopped"
    """Scheduler is not running and has no active tick loop."""

    STARTING = "starting"
    """Scheduler is initializing and preparing to emit ticks."""

    RUNNING = "running"
    """Scheduler is actively emitting ticks at configured intervals."""

    STOPPING = "stopping"
    """Scheduler is gracefully shutting down."""

    ERROR = "error"
    """Scheduler encountered a fatal error and cannot continue."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    def is_active(self) -> bool:
        """
        Check if the scheduler is in an active operational state.

        Active states are those where the scheduler is either running
        or in the process of starting/stopping.

        Returns:
            True if the scheduler is in an active state, False otherwise

        Example:
            >>> EnumSchedulerStatus.RUNNING.is_active()
            True
            >>> EnumSchedulerStatus.STOPPED.is_active()
            False
        """
        return self in {
            EnumSchedulerStatus.STARTING,
            EnumSchedulerStatus.RUNNING,
            EnumSchedulerStatus.STOPPING,
        }

    def can_start(self) -> bool:
        """
        Check if the scheduler can transition to STARTING state.

        Only STOPPED and ERROR states allow starting the scheduler.

        Returns:
            True if the scheduler can be started, False otherwise

        Example:
            >>> EnumSchedulerStatus.STOPPED.can_start()
            True
            >>> EnumSchedulerStatus.RUNNING.can_start()
            False
        """
        return self in {
            EnumSchedulerStatus.STOPPED,
            EnumSchedulerStatus.ERROR,
        }

    def can_stop(self) -> bool:
        """
        Check if the scheduler can transition to STOPPING state.

        Only RUNNING and STARTING states allow stopping the scheduler.

        Returns:
            True if the scheduler can be stopped, False otherwise

        Example:
            >>> EnumSchedulerStatus.RUNNING.can_stop()
            True
            >>> EnumSchedulerStatus.STOPPED.can_stop()
            False
        """
        return self in {
            EnumSchedulerStatus.STARTING,
            EnumSchedulerStatus.RUNNING,
        }

    def is_terminal(self) -> bool:
        """
        Check if this status represents a terminal (non-running) state.

        Terminal states indicate the scheduler is not actively processing.

        Returns:
            True if the status is terminal, False otherwise

        Example:
            >>> EnumSchedulerStatus.STOPPED.is_terminal()
            True
            >>> EnumSchedulerStatus.RUNNING.is_terminal()
            False
        """
        return self in {
            EnumSchedulerStatus.STOPPED,
            EnumSchedulerStatus.ERROR,
        }

    @classmethod
    def get_description(cls, status: "EnumSchedulerStatus") -> str:
        """
        Get a human-readable description of the scheduler status.

        Args:
            status: The scheduler status to describe

        Returns:
            A human-readable description of the status

        Example:
            >>> EnumSchedulerStatus.get_description(EnumSchedulerStatus.RUNNING)
            'Scheduler is actively emitting ticks at configured intervals'
        """
        descriptions = {
            cls.STOPPED: "Scheduler is not running and has no active tick loop",
            cls.STARTING: "Scheduler is initializing and preparing to emit ticks",
            cls.RUNNING: "Scheduler is actively emitting ticks at configured intervals",
            cls.STOPPING: "Scheduler is gracefully shutting down",
            cls.ERROR: "Scheduler encountered a fatal error and cannot continue",
        }
        return descriptions.get(status, "Unknown scheduler status")


__all__: list[str] = ["EnumSchedulerStatus"]
