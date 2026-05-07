# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Discovered contract model from onex.nodes entry point scanning (OMN-7653)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing import (
    ModelHandlerRouting,
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
    runtime_profiles: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Optional runtime profiles allowed to own this contract. "
            "Empty means backward-compatible ownership by every runtime profile."
        ),
    )
    terminal_event: str | None = Field(
        default=None,
        description="Optional terminal event topic declared by orchestrator contracts.",
    )
    event_bus: ModelEventBusWiring | None = Field(
        default=None, description="Event bus wiring if declared"
    )
    handler_routing: ModelHandlerRouting | None = Field(
        default=None, description="Handler routing if declared"
    )

    @field_validator("runtime_profiles", mode="before")
    @classmethod
    def validate_runtime_profiles(cls, value: object) -> tuple[str, ...]:
        """Normalize optional runtime profile ownership declarations."""
        if value is None:
            return ()
        if isinstance(value, str):
            raw_values = (value,)
        elif isinstance(value, (list, tuple, set)):
            raw_values = tuple(value)
        else:
            raise TypeError("runtime_profiles must be a string or sequence of strings")

        profiles: list[str] = []
        for raw in raw_values:
            if not isinstance(raw, str):
                raise TypeError("runtime_profiles entries must be strings")
            profile = raw.strip().lower()
            if not profile:
                raise ValueError("runtime_profiles entries cannot be blank")
            profiles.append(profile)
        return tuple(dict.fromkeys(profiles))
