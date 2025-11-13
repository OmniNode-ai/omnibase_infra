"""
Protocol definitions for testable CLI components.

These protocols define the interfaces for dependency injection,
enabling comprehensive testing with mocks.
"""

from collections.abc import Callable
from typing import Any, Protocol
from uuid import UUID

from omninode_bridge.events.codegen import ModelEventNodeGenerationRequested


class KafkaClientProtocol(Protocol):
    """
    Protocol for Kafka client operations.

    Defines the interface for publishing requests and consuming progress events,
    enabling easy mocking in tests.
    """

    async def connect(self) -> None:
        """Connect to Kafka broker."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from Kafka broker."""
        ...

    async def publish_request(self, event: ModelEventNodeGenerationRequested) -> None:
        """
        Publish node generation request event.

        Args:
            event: Generation request event
        """
        ...

    async def consume_progress_events(
        self,
        correlation_id: UUID,
        callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """
        Consume progress events for a specific correlation ID.

        Args:
            correlation_id: Correlation ID to filter events
            callback: Callback function(event_type, event_data)
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        ...


class ProgressDisplayProtocol(Protocol):
    """
    Protocol for progress display operations.

    Defines the interface for displaying progress updates,
    enabling easy mocking in tests.
    """

    def on_event(self, event_type: str, event_data: dict[str, Any]) -> None:
        """
        Handle incoming progress event.

        Args:
            event_type: Type of event received
            event_data: Event payload
        """
        ...

    async def wait_for_completion(self, timeout_seconds: int = 300) -> dict[str, Any]:
        """
        Wait for workflow completion with timeout.

        Args:
            timeout_seconds: Timeout in seconds

        Returns:
            dict with generation results

        Raises:
            TimeoutError: If timeout is exceeded
            RuntimeError: If generation failed
        """
        ...

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since start."""
        ...

    def get_progress_percentage(self) -> float:
        """Get completion percentage (0-100)."""
        ...
