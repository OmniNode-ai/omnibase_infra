# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Health Check Payload Model.

This module provides the Pydantic model for vault.health_check operation results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from omnibase_infra.handlers.models.vault.enum_vault_operation_type import (
    EnumVaultOperationType,
)
from omnibase_infra.handlers.models.vault.model_payload_vault import (
    ModelPayloadVault,
    RegistryPayloadVault,
)


@RegistryPayloadVault.register("health_check")
class ModelVaultHealthCheckPayload(ModelPayloadVault):
    """Payload for vault.health_check operation result.

    Contains Vault handler health status and operational metrics.

    Attributes:
        operation_type: Discriminator set to "health_check"
        healthy: Whether the Vault connection is healthy
        initialized: Whether the handler has been initialized
        handler_type: Handler type identifier (e.g., "vault")
        timeout_seconds: Configured timeout in seconds
        token_ttl_remaining_seconds: Seconds until token expiration
        circuit_breaker_state: Circuit breaker state ("open" or "closed")
        circuit_breaker_failure_count: Current circuit breaker failure count
        thread_pool_active_workers: Active threads in the executor pool
        thread_pool_max_workers: Maximum threads in the executor pool

    Example:
        >>> payload = ModelVaultHealthCheckPayload(
        ...     healthy=True,
        ...     initialized=True,
        ...     handler_type="vault",
        ...     timeout_seconds=30.0,
        ... )
        >>> print(payload.healthy)
        True
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    operation_type: Literal[EnumVaultOperationType.HEALTH_CHECK] = Field(
        default=EnumVaultOperationType.HEALTH_CHECK,
        description="Operation type discriminator",
    )
    healthy: bool = Field(
        description="Whether the Vault connection is healthy",
    )
    initialized: bool = Field(
        description="Whether the handler has been initialized",
    )
    handler_type: str = Field(
        description="Handler type identifier",
    )
    timeout_seconds: float = Field(
        description="Configured timeout in seconds",
    )
    token_ttl_remaining_seconds: int | None = Field(
        default=None,
        description="Seconds until token expiration",
    )
    circuit_breaker_state: str | None = Field(
        default=None,
        description="Circuit breaker state",
    )
    circuit_breaker_failure_count: int = Field(
        default=0,
        ge=0,
        description="Current circuit breaker failure count",
    )
    thread_pool_active_workers: int = Field(
        default=0,
        ge=0,
        description="Active threads in the executor pool",
    )
    thread_pool_max_workers: int = Field(
        default=0,
        ge=0,
        description="Maximum threads in the executor pool",
    )


__all__: list[str] = ["ModelVaultHealthCheckPayload"]
