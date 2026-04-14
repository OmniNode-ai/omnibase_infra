# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Machine registry models for ONEX environment management.

Defines ModelMachineEntry, ModelMachineRegistry, and EnumMachineRole.
All paths derive from omni_home and resolved_home_dir — no hardcoding.

Related:
    - OMN-7527: Task 1 — machine registry schema and seed file
    - OMN-7526: Environment suite epic
"""

from __future__ import annotations

import socket
from enum import Enum
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class EnumMachineRole(str, Enum):
    INFRA_SERVER = "infra_server"
    DEV_WORKSTATION = "dev_workstation"
    CI_RUNNER = "ci_runner"


class ModelMachineEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    machine_id: str = Field(..., description="Unique machine identifier.")
    hostname: str = Field(..., description="Fully qualified or short hostname.")
    role: EnumMachineRole = Field(..., description="Machine role in the fleet.")
    omni_home: str = Field(..., description="Absolute path to omni_home checkout.")
    resolved_home_dir: str = Field(..., description="Absolute home directory path.")

    @field_validator("omni_home", "resolved_home_dir", mode="after")
    @classmethod
    def _require_absolute(cls, v: str) -> str:
        if not Path(v).is_absolute():
            raise ValueError(f"omni_home must be absolute, got: {v}")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def onex_state_dir(self) -> str:
        return f"{self.omni_home}/.onex_state"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def statusline_path(self) -> str:
        return f"{self.omni_home}/omniclaude/plugins/onex/hooks/scripts/statusline.sh"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def plugin_path(self) -> str:
        return f"{self.omni_home}/omniclaude/plugins/onex"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def claude_settings_path(self) -> str:
        return f"{self.resolved_home_dir}/.claude/settings.json"


class ModelMachineRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    machines: list[ModelMachineEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_duplicate_ids(self) -> ModelMachineRegistry:
        ids = [m.machine_id for m in self.machines]
        for mid in ids:
            if ids.count(mid) > 1:
                raise ValueError(f"Duplicate machine_id: {mid}")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ModelMachineRegistry:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def get_machine(self, machine_id: str) -> ModelMachineEntry:
        for m in self.machines:
            if m.machine_id == machine_id:
                return m
        raise KeyError(f"machine_id not found: {machine_id!r}")

    def get_machines_by_role(self, role: EnumMachineRole) -> list[ModelMachineEntry]:
        return [m for m in self.machines if m.role == role]

    def resolve_local_machine(self) -> ModelMachineEntry | None:
        local_hostname = socket.gethostname()
        for m in self.machines:
            if m.hostname == local_hostname or m.hostname.startswith(local_hostname):
                return m
        return None
