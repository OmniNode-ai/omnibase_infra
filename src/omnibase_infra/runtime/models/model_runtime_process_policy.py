# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime process activation policy model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RuntimeProcessName = Literal["main", "effects", "worker"]


class ModelRuntimeProcessPolicy(BaseModel):
    """Runtime process policy owned by the runtime profile contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    runtime_instance: str = Field(alias="runtime_id", min_length=1)
    runtime_address: str = Field(pattern=r"^runtime://[^/]+/[^/]+/[^/]+$")
    capabilities: tuple[str, ...] = Field(min_length=1)
    bifrost_verify_endpoints: bool
    omnimemory_enabled: bool
    omnimemory_memgraph_host: str = ""
    publish_introspection: bool = False

    @field_validator("capabilities")
    @classmethod
    def _capabilities_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = "runtime process capabilities must be unique"
            raise ValueError(msg)
        return value
