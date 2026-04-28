# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Local runtime ingress configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelLocalRuntimeIngressConfig(BaseModel):
    """Configuration for the runtime-owned local command ingress."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    enabled: bool = Field(
        default=False,
        description="Whether the runtime should bind a local Unix-socket ingress.",
    )
    socket_path: str = Field(
        default="/tmp/onex-runtime.sock",  # noqa: S108
        min_length=1,
        description="Unix-socket path for local runtime command ingress.",
    )
    socket_permissions: int = Field(
        default=0o660,
        ge=0,
        le=0o777,
        description="Unix permission mode applied to the socket file.",
    )
    socket_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        le=300,
        description="Per-read timeout for local ingress client connections.",
    )
    max_payload_bytes: int = Field(
        default=1_048_576,
        gt=0,
        le=16_777_216,
        description="Maximum accepted request payload size in bytes.",
    )
    package_names: tuple[str, ...] = Field(
        default=("omnibase_infra",),
        description="Package roots scanned for runtime-executable node contracts.",
    )

    @field_validator("socket_path")
    @classmethod
    def _validate_socket_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("socket_path must be a non-empty string")
        return normalized

    @field_validator("package_names", mode="before")
    @classmethod
    def _coerce_package_names(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("package_names")
    @classmethod
    def _validate_package_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(part.strip() for part in value if part.strip())
        if not normalized:
            raise ValueError("package_names must contain at least one package name")
        return normalized


__all__ = ["ModelLocalRuntimeIngressConfig"]
