# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request model for local runtime ingress."""

from __future__ import annotations

from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_core.types import JsonType


class ModelLocalRuntimeIngressRequest(BaseModel):
    """Validated request envelope for local runtime ingress."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    command_name: str | None = Field(
        default=None,
        min_length=1,
        description="Canonical broker command name for the runtime-backed request.",
    )
    node_alias: str | None = Field(
        default=None,
        min_length=1,
        description="Deprecated compatibility alias for the requested command.",
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

    @field_validator("command_name", "node_alias")
    @classmethod
    def _validate_name_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("command_name/node_alias must be a non-empty string")
        return normalized

    @model_validator(mode="after")
    def _validate_command_or_alias(self) -> Self:
        if self.command_name is None and self.node_alias is None:
            raise ValueError("Either command_name or node_alias must be provided")
        return self

    @property
    def requested_command_name(self) -> str:
        if self.command_name is not None:
            return self.command_name
        if self.node_alias is not None:
            return self.node_alias
        raise RuntimeError("validated request is missing command_name and node_alias")


__all__ = ["ModelLocalRuntimeIngressRequest"]
