# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed, required settings for process-local runtime state paths."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class ModelLocalStatePaths(BaseModel):
    """Load local state locations from the shipped declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    runtime_socket_path: str = Field(min_length=1)
    gateway_egress_health_path: str = Field(min_length=1)
    gateway_lane_mirror_health_path: str = Field(min_length=1)
    gateway_canary_probe_state_path: str = Field(min_length=1)
    scope_manifest_path: str = Field(min_length=1)

    @classmethod
    def from_file(cls, path: Path) -> ModelLocalStatePaths:
        """Read the required settings mapping and validate every location."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("settings"), dict):
            raise ProtocolConfigurationError(
                "Local state paths configuration requires a settings mapping",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.FILESYSTEM,
                    operation="load_local_state_paths",
                ),
                config_path=str(path),
            )
        return cls.model_validate(data["settings"])


@cache
def load_local_state_paths() -> ModelLocalStatePaths:
    """Load the packaged declaration once per process."""
    return ModelLocalStatePaths.from_file(
        Path(__file__).with_name("local_state_paths.yaml")
    )


__all__ = ["ModelLocalStatePaths", "load_local_state_paths"]
