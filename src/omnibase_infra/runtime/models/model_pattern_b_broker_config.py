# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pattern B broker runtime configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

import omnibase_infra.event_bus.topic_constants as _tc


class ModelPatternBBrokerConfig(BaseModel):
    """Configuration for the runtime-owned Pattern B broker service."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    enabled: bool = Field(
        default=False,
        description="Whether the runtime should start the Pattern B broker service.",
    )
    command_topic: str = Field(
        default=_tc.TOPIC_PATTERN_B_DISPATCH_COMMAND,
        min_length=1,
        description="Command topic consumed by the Pattern B broker.",
    )
    package_names: tuple[str, ...] = Field(
        default=("omnibase_infra",),
        description="Package roots scanned for broker-addressable node contracts.",
    )

    @field_validator("command_topic")
    @classmethod
    def _validate_command_topic(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("command_topic must be a non-empty string")
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


__all__ = ["ModelPatternBBrokerConfig"]
