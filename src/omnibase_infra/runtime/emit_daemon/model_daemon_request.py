# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Daemon request models for emit daemon protocol.

This module defines the strongly-typed request models for the emit daemon
Unix socket protocol. Using Pydantic models instead of dict[str, object]
provides compile-time type safety and eliminates runtime isinstance checks.

Request Types:
    - ModelDaemonPingRequest: Health check / ping command
    - ModelDaemonEmitRequest: Event emission request

Protocol:
    Requests are JSON-encoded and sent as newline-delimited messages over
    the Unix socket. The daemon discriminates between request types based
    on the presence of the "command" field (commands) vs "event_type" field
    (event emissions).

Related Tickets:
    - OMN-1610: Hook Event Daemon MVP

.. versionadded:: 0.2.6
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelDaemonPingRequest(BaseModel):
    """Request model for daemon ping/health check command.

    This command is used to verify the daemon is running and get
    current queue status.

    Example:
        ```python
        request = ModelDaemonPingRequest()
        # Serializes to: {"command": "ping"}
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    command: Literal["ping"] = Field(
        default="ping",
        description="Command identifier for ping request",
    )


class ModelDaemonEmitRequest(BaseModel):
    """Request model for event emission.

    Contains the event type and payload to be published to Kafka
    via the daemon's persistent connection.

    Attributes:
        event_type: Semantic event type (e.g., "prompt.submitted").
            Must be registered with the daemon's EventRegistry.
        payload: Event payload. Must contain required fields for the
            event type as defined in EventRegistry.

    Example:
        ```python
        request = ModelDaemonEmitRequest(
            event_type="prompt.submitted",
            payload={"prompt_id": "abc123", "session_id": "sess-456"},
        )
        ```
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    event_type: str = Field(
        ...,
        description="Semantic event type (e.g., 'prompt.submitted')",
        min_length=1,
    )

    payload: JsonType = Field(
        default_factory=dict,
        description="Event payload data",
    )


# Type alias for discriminated union of all request types
# Use field presence for discrimination:
# - "command" field present -> ModelDaemonPingRequest
# - "event_type" field present -> ModelDaemonEmitRequest
ModelDaemonRequest = Annotated[
    ModelDaemonPingRequest | ModelDaemonEmitRequest,
    Field(description="Union of all daemon request types"),
]


def parse_daemon_request(
    data: dict[str, object],
) -> ModelDaemonPingRequest | ModelDaemonEmitRequest:
    """Parse raw dict into typed request model.

    Discriminates between request types based on field presence:
    - "command" field present -> ModelDaemonPingRequest
    - "event_type" field present -> ModelDaemonEmitRequest

    Args:
        data: Raw request dict from JSON parsing

    Returns:
        Typed request model

    Raises:
        ValueError: If request format is invalid
    """
    if "command" in data:
        return ModelDaemonPingRequest.model_validate(data)
    elif "event_type" in data:
        return ModelDaemonEmitRequest.model_validate(data)
    else:
        raise ValueError(
            "Invalid request: must contain either 'command' or 'event_type' field"
        )


__all__: list[str] = [
    "ModelDaemonEmitRequest",
    "ModelDaemonPingRequest",
    "ModelDaemonRequest",
    "parse_daemon_request",
]
