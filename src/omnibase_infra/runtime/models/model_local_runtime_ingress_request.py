# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request model for local runtime ingress."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.types import JsonType


class ModelLocalRuntimeIngressRequest(BaseModel):
    """Validated request envelope for local runtime ingress."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    node_alias: str = Field(
        ..., min_length=1, description="Requested runtime node alias."
    )
    payload: dict[str, JsonType] = Field(
        default_factory=dict,
        description="JSON payload forwarded to the backing node handler.",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Optional caller-supplied correlation identifier.",
    )
    timeout_ms: int = Field(
        default=300_000,
        gt=0,
        le=900_000,
        description="Maximum local dispatch time before returning a timeout response.",
    )

    @field_validator("node_alias")
    @classmethod
    def _validate_node_alias(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("node_alias must be a non-empty string")
        return normalized


__all__ = ["ModelLocalRuntimeIngressRequest"]
