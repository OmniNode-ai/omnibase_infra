# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-owned policy for runtime event bus recovery."""

from __future__ import annotations

from importlib.resources import files

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelEventBusRecoveryConfig(BaseModel):
    """Bound each connect/attach attempt and cap the exponential retry delay."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    initial_backoff_seconds: float = Field(gt=0)
    max_backoff_seconds: float = Field(gt=0)
    backoff_multiplier: float = Field(ge=1)
    attempt_timeout_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> ModelEventBusRecoveryConfig:
        """Reject a cap smaller than the initial delay."""
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds")
        return self

    @classmethod
    def from_contract(cls) -> ModelEventBusRecoveryConfig:
        """Load the shipped policy; runtime overlays can replace this model."""
        resource = files("omnibase_infra").joinpath(
            "contracts/runtime_event_bus_recovery.yaml"
        )
        return cls.model_validate(yaml.safe_load(resource.read_text(encoding="utf-8")))


__all__ = ["ModelEventBusRecoveryConfig"]
