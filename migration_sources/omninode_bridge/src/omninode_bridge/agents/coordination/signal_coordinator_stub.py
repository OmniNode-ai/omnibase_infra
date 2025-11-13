"""
Stub interface for SignalCoordinator integration.

This stub provides the interface contract that DependencyResolver expects from
SignalCoordinator. Once SignalCoordinator is implemented (Component 1), this stub
will be replaced with the actual implementation.

The stub allows DependencyResolver to be developed and tested independently.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class ISignalCoordinator(ABC):
    """
    Interface contract for SignalCoordinator integration.

    This interface defines the methods that DependencyResolver requires from
    SignalCoordinator for dependency resolution signaling and event checking.
    """

    @abstractmethod
    async def signal_event(
        self,
        coordination_id: str,
        event_type: str,
        event_data: dict[str, Any],
        priority: int = 5,
    ) -> None:
        """
        Signal a coordination event.

        Args:
            coordination_id: Coordination session ID
            event_type: Type of event to signal (e.g., "dependency_resolved")
            event_data: Event payload data
            priority: Signal priority (1-10, higher = more urgent)
        """
        pass

    @abstractmethod
    async def check_agent_completion(
        self,
        coordination_id: str,
        agent_id: str,
        completion_event: str = "completion",
    ) -> bool:
        """
        Check if specific agent has completed.

        Args:
            coordination_id: Coordination session ID
            agent_id: Agent identifier to check
            completion_event: Completion event name to look for

        Returns:
            True if agent has signaled completion, False otherwise
        """
        pass

    @abstractmethod
    async def get_coordination_signals(
        self,
        coordination_id: str,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get coordination signals for session.

        Args:
            coordination_id: Coordination session ID
            event_type: Filter by event type (None = all events)

        Returns:
            List of signal dictionaries matching criteria
        """
        pass


class SignalCoordinatorStub(ISignalCoordinator):
    """
    Stub implementation of SignalCoordinator for testing.

    This stub provides a minimal in-memory implementation for testing dependency
    resolution without requiring the full SignalCoordinator implementation.

    Once SignalCoordinator is implemented, replace this stub with:
        from .signal_coordinator import SignalCoordinator
    """

    def __init__(self) -> None:
        """Initialize stub with in-memory signal storage."""
        self._signals: dict[str, list[dict[str, Any]]] = {}
        self._agent_completions: dict[str, set[str]] = {}

    async def signal_event(
        self,
        coordination_id: str,
        event_type: str,
        event_data: dict[str, Any],
        priority: int = 5,
    ) -> None:
        """Signal an event (stub implementation)."""
        if coordination_id not in self._signals:
            self._signals[coordination_id] = []

        signal = {
            "event_type": event_type,
            "event_data": event_data,
            "priority": priority,
        }
        self._signals[coordination_id].append(signal)

    async def check_agent_completion(
        self,
        coordination_id: str,
        agent_id: str,
        completion_event: str = "completion",
    ) -> bool:
        """Check if agent completed (stub implementation)."""
        session_key = f"{coordination_id}:{completion_event}"
        return (
            session_key in self._agent_completions
            and agent_id in self._agent_completions[session_key]
        )

    async def get_coordination_signals(
        self,
        coordination_id: str,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get coordination signals (stub implementation)."""
        signals = self._signals.get(coordination_id, [])

        if event_type:
            signals = [s for s in signals if s["event_type"] == event_type]

        return signals

    # Helper methods for testing

    async def mark_agent_completed(
        self,
        coordination_id: str,
        agent_id: str,
        completion_event: str = "completion",
    ) -> None:
        """Mark an agent as completed (for testing)."""
        session_key = f"{coordination_id}:{completion_event}"
        if session_key not in self._agent_completions:
            self._agent_completions[session_key] = set()

        self._agent_completions[session_key].add(agent_id)

    def clear(self) -> None:
        """Clear all signals and completions (for testing)."""
        self._signals.clear()
        self._agent_completions.clear()
