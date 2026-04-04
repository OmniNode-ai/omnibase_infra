# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Machine registry models for multi-machine fleet management."""

from __future__ import annotations

import socket
from enum import StrEnum
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


class EnumMachineRole(StrEnum):
    DEV = "dev"
    INFRA = "infra"
    AGENT_RUNNER = "agent_runner"


class ModelMachineEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    machine_id: str = Field(..., min_length=1, description="Unique machine identifier")
    hostname: str = Field(
        ..., min_length=1, description="Network hostname (used for locality detection)"
    )
    ip: str = Field(..., description="IP address or hostname for SSH")
    role: EnumMachineRole = Field(..., description="Machine role in the fleet")
    omni_home: str = Field(
        ..., description="Absolute path to omni_home on this machine"
    )
    ssh_user: str = Field(default="jonah", description="SSH username")
    ssh_port: int = Field(default=22, description="SSH port")
    home_dir: str | None = Field(
        default=None,
        description="User home dir override (default: /Users/{ssh_user} on mac, /home/{ssh_user} on linux)",
    )
    infisical_participant: bool = Field(
        default=True,
        description="Whether this machine participates in Infisical provisioning",
    )

    @field_validator("omni_home")
    @classmethod
    def validate_absolute_path(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("omni_home must be absolute (start with /)")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_home_dir(self) -> str:
        if self.home_dir:
            return self.home_dir
        if self.omni_home.startswith("/data/") or self.omni_home.startswith("/home/"):
            return f"/home/{self.ssh_user}"
        return f"/Users/{self.ssh_user}"

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
    def validate_unique_machine_ids(self) -> ModelMachineRegistry:
        ids = [m.machine_id for m in self.machines]
        dupes = [mid for mid in ids if ids.count(mid) > 1]
        if dupes:
            raise ValueError(f"Duplicate machine_id: {set(dupes)}")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ModelMachineRegistry:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_machine(self, machine_id: str) -> ModelMachineEntry:
        for m in self.machines:
            if m.machine_id == machine_id:
                return m
        raise KeyError(f"Machine {machine_id!r} not found in registry")

    def get_machines_by_role(self, role: EnumMachineRole) -> list[ModelMachineEntry]:
        return [m for m in self.machines if m.role == role]

    def resolve_local_machine(self) -> ModelMachineEntry | None:
        local_hostname = socket.gethostname().split(".")[0].lower()
        for m in self.machines:
            if m.hostname.split(".")[0].lower() == local_hostname:
                return m
        return None
