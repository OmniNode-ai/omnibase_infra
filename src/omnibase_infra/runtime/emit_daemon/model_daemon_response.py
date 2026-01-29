# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Daemon response models for emit daemon protocol.

This module defines the strongly-typed response models for the emit daemon
Unix socket protocol. Using Pydantic models instead of dict[str, object]
provides compile-time type safety and eliminates runtime isinstance checks.

Response Types:
    - ModelDaemonPingResponse: Health check response with queue status
    - ModelDaemonQueuedResponse: Event successfully queued
    - ModelDaemonErrorResponse: Error response with reason

Protocol:
    Responses are JSON-encoded and sent as newline-delimited messages over
    the Unix socket. The "status" field discriminates between response types:
    - "ok": Ping response with queue/spool sizes
    - "queued": Event emission success with event_id
    - "error": Error with reason string

Related Tickets:
    - OMN-1610: Hook Event Daemon MVP

.. versionadded:: 0.2.6
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelDaemonPingResponse(BaseModel):
    """Response model for successful ping command.

    Contains daemon health status and current queue metrics.

    Attributes:
        status: Always "ok" for successful ping
        queue_size: Number of events in memory queue
        spool_size: Number of events spooled to disk

    Example:
        ```python
        response = ModelDaemonPingResponse(queue_size=5, spool_size=10)
        # Serializes to: {"status": "ok", "queue_size": 5, "spool_size": 10}
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    status: Literal["ok"] = Field(
        default="ok",
        description="Status indicator for successful ping",
    )

    queue_size: int = Field(
        ...,
        ge=0,
        description="Number of events currently in memory queue",
    )

    spool_size: int = Field(
        ...,
        ge=0,
        description="Number of events currently spooled to disk",
    )


class ModelDaemonQueuedResponse(BaseModel):
    """Response model for successfully queued event.

    Returned when an event has been accepted and queued for
    asynchronous publishing to Kafka.

    Attributes:
        status: Always "queued" for successful event submission
        event_id: UUID assigned to the queued event for tracking

    Example:
        ```python
        response = ModelDaemonQueuedResponse(
            event_id="550e8400-e29b-41d4-a716-446655440000"
        )
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    status: Literal["queued"] = Field(
        default="queued",
        description="Status indicator for queued event",
    )

    event_id: str = Field(
        ...,
        description="UUID assigned to the queued event",
        min_length=1,
    )


class ModelDaemonErrorResponse(BaseModel):
    """Response model for daemon errors.

    Returned when an error occurs processing a request.

    Attributes:
        status: Always "error" for error responses
        reason: Human-readable error description

    Example:
        ```python
        response = ModelDaemonErrorResponse(
            reason="Unknown event type: invalid.event"
        )
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    status: Literal["error"] = Field(
        default="error",
        description="Status indicator for error response",
    )

    reason: str = Field(
        ...,
        description="Human-readable error description",
        min_length=1,
    )


# Type alias for discriminated union of all response types
# Discriminated by "status" field:
# - "ok" -> ModelDaemonPingResponse
# - "queued" -> ModelDaemonQueuedResponse
# - "error" -> ModelDaemonErrorResponse
ModelDaemonResponse = Annotated[
    ModelDaemonPingResponse | ModelDaemonQueuedResponse | ModelDaemonErrorResponse,
    Field(discriminator="status", description="Union of all daemon response types"),
]


def parse_daemon_response(
    data: dict[str, object],
) -> ModelDaemonPingResponse | ModelDaemonQueuedResponse | ModelDaemonErrorResponse:
    """Parse raw dict into typed response model.

    Discriminates between response types based on "status" field:
    - "ok" -> ModelDaemonPingResponse
    - "queued" -> ModelDaemonQueuedResponse
    - "error" -> ModelDaemonErrorResponse

    Args:
        data: Raw response dict from JSON parsing

    Returns:
        Typed response model

    Raises:
        ValueError: If response format is invalid or status unknown
    """
    status = data.get("status")

    if status == "ok":
        return ModelDaemonPingResponse.model_validate(data)
    elif status == "queued":
        return ModelDaemonQueuedResponse.model_validate(data)
    elif status == "error":
        return ModelDaemonErrorResponse.model_validate(data)
    else:
        raise ValueError(f"Unknown response status: {status}")


__all__: list[str] = [
    "ModelDaemonErrorResponse",
    "ModelDaemonPingResponse",
    "ModelDaemonQueuedResponse",
    "ModelDaemonResponse",
    "parse_daemon_response",
]
