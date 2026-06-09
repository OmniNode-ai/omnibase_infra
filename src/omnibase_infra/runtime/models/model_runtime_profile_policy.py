# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime profile policy model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.runtime.models.model_runtime_process_policy import (
    ModelRuntimeProcessPolicy,
    RuntimeProcessName,
)
from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping


class ModelRuntimeProfilePolicy(BaseModel):
    """Policy for a single runtime lane."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    compose_project: str = Field(min_length=1)
    main_port: int = Field(ge=1, le=65535)
    effects_port: int = Field(ge=1, le=65535)
    topic_provisioner_max_partitions: int = Field(ge=0)
    secret_resolver_config_path: str = ""
    secret_resolver_mappings: tuple[ModelSecretMapping, ...] = ()
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
        logical_names = [
            mapping.logical_name for mapping in self.secret_resolver_mappings
        ]
        if len(logical_names) != len(set(logical_names)):
            msg = "runtime profile secret resolver logical names must be unique"
            raise ValueError(msg)
        if (
            self.secret_resolver_mappings
            and not self.secret_resolver_config_path.strip()
        ):
            msg = "runtime profile secret resolver mappings require a config path"
            raise ValueError(msg)
        return self
