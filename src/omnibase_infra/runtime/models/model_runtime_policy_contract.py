# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime policy contract model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.runtime.models.model_runtime_profile_policy import (
    ModelRuntimeProfilePolicy,
)

RuntimeProfileName = Literal["dev", "stability-test", "prod"]


class ModelRuntimePolicyContract(BaseModel):
    """Contract-owned runtime policy rendered into deployment env."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    name: Literal["runtime_policy"]
    version: int = Field(ge=1)
    active_runtime_packages: tuple[str, ...] = Field(min_length=1)
    omnimemory_memgraph_port: int = Field(ge=1, le=65535)
    auxiliary_services_omnimemory_enabled: bool
    profiles: dict[RuntimeProfileName, ModelRuntimeProfilePolicy]

    @field_validator("active_runtime_packages")
    @classmethod
    def _packages_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = "active runtime packages must be unique"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _requires_three_runtime_profiles(self) -> ModelRuntimePolicyContract:
        required = {"dev", "stability-test", "prod"}
        observed = set(self.profiles)
        if observed != required:
            msg = f"runtime policy profiles must be {sorted(required)}, got {sorted(observed)}"
            raise ValueError(msg)

        addresses: set[str] = set()
        for profile in self.profiles.values():
            for process in profile.processes.values():
                if process.runtime_address in addresses:
                    msg = f"duplicate runtime address {process.runtime_address}"
                    raise ValueError(msg)
                addresses.add(process.runtime_address)
        return self
