# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Strict contract overlay for local Bifrost delegation bindings (OMN-15807)."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    ACTIVE_BACKEND_KEYS,
    ModelBifrostLaneBackendBinding,
)

_SCHEMA_VERSION = "bifrost_lane_overlay.v2"


class ModelBifrostLaneOverlay(BaseModel):
    """The sole typed authority for active local Bifrost delegation bindings."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    schema_version: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    backends: tuple[ModelBifrostLaneBackendBinding, ...] = Field(min_length=1)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != _SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {_SCHEMA_VERSION!r}, got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _validate_active_bindings(self) -> Self:
        backend_keys = [binding.backend_key for binding in self.backends]
        if len(backend_keys) != len(set(backend_keys)):
            raise ValueError("backends must not contain duplicate backend_id values")
        if set(backend_keys) != ACTIVE_BACKEND_KEYS:
            raise ValueError(
                "backends must declare exactly the active local backend IDs "
                f"{sorted(ACTIVE_BACKEND_KEYS)}, got {sorted(backend_keys)}"
            )
        return self


__all__ = [
    "ModelBifrostLaneOverlay",
]
