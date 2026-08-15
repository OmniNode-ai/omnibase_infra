# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed application-database topology profile catalog."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.topology.models.model_application_database_topology_profile import (
    ModelApplicationDatabaseTopologyProfile,
)

__all__ = ["ModelApplicationDatabaseTopologyProfileCatalog"]


class ModelApplicationDatabaseTopologyProfileCatalog(BaseModel):
    """Secret-free profile bindings owned by ``omnibase_infra``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"]
    profiles: dict[str, ModelApplicationDatabaseTopologyProfile] = Field(min_length=1)

    @field_validator("profiles")
    @classmethod
    def profile_names_are_canonical(
        cls,
        value: dict[str, ModelApplicationDatabaseTopologyProfile],
    ) -> dict[str, ModelApplicationDatabaseTopologyProfile]:
        """Keep profile identifiers exact and shell/YAML safe."""
        for profile in value:
            if not profile or any(
                not (character.islower() or character.isdigit() or character == "-")
                for character in profile
            ):
                raise ValueError(f"invalid database topology profile {profile!r}")
        return value
