# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Health Check Payload Model.

This module provides the payload model for consul.health_check result.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from omnibase_infra.handlers.models.consul.model_payload_consul import (
    ModelPayloadConsul,
    RegistryPayloadConsul,
)


@RegistryPayloadConsul.register("health_check")
class ModelConsulHealthCheckPayload(ModelPayloadConsul):
    """Payload for consul.health_check result.

    Attributes:
        operation_type: Discriminator literal "health_check".
        healthy: True if Consul is healthy and has a leader.
        initialized: True if the handler has been initialized.
        handler_type: The handler type identifier ("consul").
        timeout_seconds: Configured timeout for operations.
        circuit_breaker_state: Current circuit breaker state (open/closed).
        circuit_breaker_failure_count: Number of consecutive failures.
        thread_pool_active_workers: Number of active worker threads.
        thread_pool_max_workers: Maximum number of worker threads.
        thread_pool_max_queue_size: Configured maximum queue size.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    operation_type: Literal["health_check"] = Field(
        default="health_check", description="Discriminator for payload type"
    )
    healthy: bool = Field(description="True if Consul is healthy and has a leader")
    initialized: bool = Field(description="True if the handler has been initialized")
    handler_type: str = Field(description="The handler type identifier")
    timeout_seconds: float = Field(description="Configured timeout for operations")
    circuit_breaker_state: str | None = Field(
        description="Current circuit breaker state (open/closed)"
    )
    circuit_breaker_failure_count: int = Field(
        description="Number of consecutive failures"
    )
    thread_pool_active_workers: int = Field(
        description="Number of active worker threads"
    )
    thread_pool_max_workers: int = Field(description="Maximum number of worker threads")
    thread_pool_max_queue_size: int = Field(description="Configured maximum queue size")


__all__: list[str] = ["ModelConsulHealthCheckPayload"]
