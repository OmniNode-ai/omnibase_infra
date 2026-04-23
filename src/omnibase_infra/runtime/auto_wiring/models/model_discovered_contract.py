# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Discovered contract model from onex.nodes entry point scanning (OMN-7653)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing import (
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.models.model_runtime_profile_policy import (
    ModelRuntimeProfilePolicy,
)


class ModelDiscoveredContract(BaseModel):
    """A single contract discovered from an onex.nodes entry point.

    Captures the subset of contract YAML fields needed for auto-wiring
    without importing any handler or node classes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(..., description="Node name from contract")
    node_type: str = Field(..., description="Node type (e.g. EFFECT_GENERIC)")
    description: str = Field(default="", description="Node description")
    contract_version: ModelContractVersion = Field(
        ..., description="Contract semantic version"
    )
    node_version: str = Field(default="1.0.0", description="Node version string")
    contract_path: Path = Field(..., description="Filesystem path to contract.yaml")
    entry_point_name: str = Field(..., description="Name of the onex.nodes entry point")
    package_name: str = Field(
        ..., description="Distribution package that registered the entry point"
    )
    package_version: str = Field(
        default="0.0.0", description="Distribution package version"
    )
    event_bus: ModelEventBusWiring | None = Field(
        default=None, description="Event bus wiring if declared"
    )
    handler_routing: ModelHandlerRouting | None = Field(
        default=None, description="Handler routing if declared"
    )
    runtime_profiles: dict[str, ModelRuntimeProfilePolicy] = Field(
        default_factory=dict,
        description=(
            "Optional per-profile contract hints. Runtime policy remains "
            "authoritative; contracts may narrow but not broaden eligibility."
        ),
    )
