# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime profile policy model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.runtime.models.model_runtime_process_policy import (
    ModelRuntimeProcessPolicy,
    RuntimeProcessName,
)


class ModelRuntimeProfilePolicy(BaseModel):
    """Policy for a single runtime lane."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    compose_project: str = Field(min_length=1)
    main_port: int = Field(ge=1, le=65535)
    effects_port: int = Field(ge=1, le=65535)
    processes: dict[RuntimeProcessName, ModelRuntimeProcessPolicy] = Field(
        alias="services"
    )

    @model_validator(mode="after")
    def _requires_three_runtime_processes(self) -> ModelRuntimeProfilePolicy:
        required = {"main", "effects", "worker"}
        observed = set(self.processes)
        if observed != required:
            msg = f"runtime profile processes must be {sorted(required)}, got {sorted(observed)}"
            raise ValueError(msg)
        return self
