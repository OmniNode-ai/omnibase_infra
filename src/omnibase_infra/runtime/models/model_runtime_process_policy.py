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
    # OMN-12990: contract-declared replica count for this runtime process.
    # The base compose worker deploy default is ${WORKER_REPLICAS:-0}, which
    # silently scales the worker to zero on a plain compose recreate. Each lane
    # pins this value via {PROFILE}_WORKER_REPLICAS (rendered into
    # runtime-policy.env) and the lane overrides reference it fail-fast, so a
    # recreate that omits the policy env fails loudly instead of dropping the
    # worker with no signal.
    replicas: int = Field(default=1, ge=1)

    @field_validator("capabilities")
    @classmethod
    def _capabilities_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = "runtime process capabilities must be unique"
            raise ValueError(msg)
        return value
