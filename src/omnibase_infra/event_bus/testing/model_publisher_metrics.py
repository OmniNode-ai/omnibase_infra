# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Publisher metrics model for test adapters.

This module provides the metrics model used by AdapterProtocolEventPublisherInmemory
to track publishing statistics for observability and testing assertions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelPublisherMetrics(BaseModel):
    """Metrics model for AdapterProtocolEventPublisherInmemory.

    Tracks publishing statistics for observability and testing assertions.

    Attributes:
        events_published: Total count of successfully published events.
        events_failed: Total count of failed publish attempts.
        events_sent_to_dlq: Count of events sent to dead letter queue (always 0 for inmemory).
        total_publish_time_ms: Cumulative publish time in milliseconds.
        avg_publish_time_ms: Average publish latency (computed from total/count).
        circuit_breaker_opens: Count of circuit breaker open events (always 0 for inmemory).
        retries_attempted: Total retry attempts (always 0 for inmemory).
        circuit_breaker_status: Current circuit breaker status (always "closed" for inmemory).
        current_failures: Current consecutive failure count.
    """

    events_published: int = Field(default=0, ge=0)
    events_failed: int = Field(default=0, ge=0)
    events_sent_to_dlq: int = Field(default=0, ge=0)
    total_publish_time_ms: float = Field(default=0.0, ge=0.0)
    avg_publish_time_ms: float = Field(default=0.0, ge=0.0)
    circuit_breaker_opens: int = Field(default=0, ge=0)
    retries_attempted: int = Field(default=0, ge=0)
    circuit_breaker_status: str = Field(default="closed")
    current_failures: int = Field(default=0, ge=0)

    model_config = ConfigDict(frozen=False, extra="forbid")

    def to_dict(self) -> dict[str, JsonType]:
        """Convert metrics to dictionary for JSON serialization.

        Returns:
            Dictionary representation compatible with JsonType.
        """
        return {
            "events_published": self.events_published,
            "events_failed": self.events_failed,
            "events_sent_to_dlq": self.events_sent_to_dlq,
            "total_publish_time_ms": self.total_publish_time_ms,
            "avg_publish_time_ms": self.avg_publish_time_ms,
            "circuit_breaker_opens": self.circuit_breaker_opens,
            "retries_attempted": self.retries_attempted,
            "circuit_breaker_status": self.circuit_breaker_status,
            "current_failures": self.current_failures,
        }


__all__: list[str] = ["ModelPublisherMetrics"]
