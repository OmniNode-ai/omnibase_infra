"""Publish Event Result Model."""

from pydantic import BaseModel, Field


class ModelPublishEventResult(BaseModel):
    """Result for publish event operations."""

    event_published: bool = Field(
        description="Whether the event was successfully published",
    )

    publisher_function: str | None = Field(
        default=None,
        max_length=200,
        description="Name of the publisher function used",
    )

    publish_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time taken to publish event in milliseconds",
    )

    circuit_action_taken: str = Field(
        pattern="^(published|queued|dropped|rejected)$",
        description="Action taken by circuit breaker",
    )

    queue_length_after: int | None = Field(
        default=None,
        ge=0,
        description="Length of event queue after operation",
    )

    dead_letter_queued: bool = Field(
        default=False,
        description="Whether event was moved to dead letter queue",
    )