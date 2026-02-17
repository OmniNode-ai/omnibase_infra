# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infisical handler configuration model.

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ModelInfisicalHandlerConfig(BaseModel):
    """Configuration for HandlerInfisical.

    Attributes:
        host: Infisical server URL.
        client_id: Machine identity client ID for Universal Auth.
        client_secret: Machine identity client secret for Universal Auth.
        project_id: Default Infisical project ID.
        environment_slug: Default environment slug.
        secret_path: Default secret path prefix.
        cache_ttl_seconds: TTL for the handler-level secret cache (0 disables).
        circuit_breaker_threshold: Consecutive failures before circuit opens.
        circuit_breaker_reset_timeout: Seconds before half-open retry.
        circuit_breaker_enabled: Whether circuit breaker is active.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    host: str = Field(
        default="https://app.infisical.com",
        description="Infisical server URL.",
    )
    client_id: SecretStr = Field(
        ...,
        description="Machine identity client ID for Universal Auth.",
    )
    client_secret: SecretStr = Field(
        ...,
        description="Machine identity client secret for Universal Auth.",
    )
    project_id: UUID = Field(
        ...,
        description="Default Infisical project ID.",
    )
    environment_slug: str = Field(
        default="prod",
        min_length=1,
        description="Default environment slug.",
    )
    secret_path: str = Field(
        default="/",
        description="Default secret path prefix.",
    )
    cache_ttl_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description="TTL for the handler-level secret cache in seconds. "
        "Set to 0 to disable caching.",
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Consecutive failures before the circuit breaker opens.",
    )
    circuit_breaker_reset_timeout: float = Field(
        default=60.0,
        gt=0.0,
        description="Seconds to wait before attempting half-open recovery.",
    )
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Whether the circuit breaker is active.",
    )


__all__: list[str] = ["ModelInfisicalHandlerConfig"]
