"""
Real-time Progress Display for CLI.

Displays live progress updates as pipeline stages complete.
Tracks timing, quality metrics, and final results.
"""

import asyncio
import time
from typing import Any
from uuid import UUID


class ProgressDisplay:
    """
    Real-time progress tracker for code generation pipeline.

    Features:
    - Stage-by-stage progress display
    - Duration tracking
    - Quality metrics
    - Completion detection
    - Timeout handling

    This implementation is designed for testability with dependency injection.
    """

    def __init__(self, correlation_id: UUID, total_stages: int = 8):
        """
        Initialize progress display.

        Args:
            correlation_id: Correlation ID for this generation request
            total_stages: Total number of pipeline stages (default: 8)
        """
        self.correlation_id = correlation_id
        self.total_stages = total_stages
        self.started_at: float | None = None
        self.completed_at: float | None = None

        # Stage tracking
        self.completed_stages = 0
        self.current_stage: str | None = None
        self.stage_durations: dict[str, float] = {}

        # Results
        self.result: dict[str, Any] | None = None
        self.error: str | None = None

        # Asyncio event for completion
        self._completion_event = asyncio.Event()

    def on_event(self, event_type: str, event_data: dict[str, Any]) -> None:
        """
        Handle incoming Kafka event.

        Args:
            event_type: Type of event received
            event_data: Event payload
        """
        if event_type == "NODE_GENERATION_STARTED":
            self._on_started(event_data)
        elif event_type == "NODE_GENERATION_STAGE_COMPLETED":
            self._on_stage_completed(event_data)
        elif event_type == "NODE_GENERATION_COMPLETED":
            self._on_completed(event_data)
        elif event_type == "NODE_GENERATION_FAILED":
            self._on_failed(event_data)
        elif event_type == "ORCHESTRATOR_CHECKPOINT_REACHED":
            self._on_checkpoint_reached(event_data)

    def _on_started(self, event_data: dict[str, Any]) -> None:
        """Handle workflow started event."""
        self.started_at = time.time()
        print("⏳ Workflow started...")

    def _on_stage_completed(self, event_data: dict[str, Any]) -> None:
        """Handle stage completed event."""
        stage_name = event_data.get("stage_name")
        stage_number = event_data.get("stage_number")
        duration_seconds = event_data.get("duration_seconds", 0)
        success = event_data.get("success", True)

        self.completed_stages = stage_number
        self.stage_durations[stage_name] = duration_seconds

        # Format stage name for display
        stage_display = stage_name.replace("_", " ").title()

        # Progress indicator
        progress_pct = (stage_number / self.total_stages) * 100

        # Status icon
        icon = "✓" if success else "⚠"

        print(
            f"[{stage_number}/{self.total_stages}] {icon} {stage_display} "
            f"({duration_seconds:.1f}s) - {progress_pct:.0f}%"
        )

    def _on_completed(self, event_data: dict[str, Any]) -> None:
        """Handle workflow completed event."""
        self.completed_at = time.time()

        self.result = {
            "workflow_id": event_data.get("workflow_id"),
            "total_duration_seconds": event_data.get("total_duration_seconds", 0),
            "generated_files": event_data.get("generated_files", []),
            "node_type": event_data.get("node_type"),
            "service_name": event_data.get("service_name"),
            "quality_score": event_data.get("quality_score", 0.0),
            "test_coverage": event_data.get("test_coverage"),
            "primary_model": event_data.get("primary_model"),
            "total_tokens": event_data.get("total_tokens", 0),
            "total_cost_usd": event_data.get("total_cost_usd", 0.0),
        }

        self._completion_event.set()

    def _on_failed(self, event_data: dict[str, Any]) -> None:
        """Handle workflow failed event."""
        self.completed_at = time.time()
        self.error = event_data.get("error_message", "Unknown error")

        print(f"\n❌ Generation failed: {self.error}")
        print(f"   Failed stage: {event_data.get('failed_stage', 'unknown')}")

        self._completion_event.set()

    def _on_checkpoint_reached(self, event_data: dict[str, Any]) -> None:
        """Handle interactive checkpoint event."""
        checkpoint_type = event_data.get("checkpoint_type")
        print(f"\n🔔 Checkpoint: {checkpoint_type}")
        print("   Waiting for user input...")
        # In production, this would prompt user for input

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
        try:
            await asyncio.wait_for(
                self._completion_event.wait(),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            elapsed = time.time() - (self.started_at or time.time())
            raise TimeoutError(
                f"Generation timed out after {elapsed:.1f}s "
                f"(limit: {timeout_seconds}s)"
            )

        if self.error:
            raise RuntimeError(self.error)

        if not self.result:
            raise RuntimeError("Generation completed without result")

        return self.result

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since start."""
        if not self.started_at:
            return 0.0

        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    def get_progress_percentage(self) -> float:
        """Get completion percentage (0-100)."""
        return (self.completed_stages / self.total_stages) * 100.0
