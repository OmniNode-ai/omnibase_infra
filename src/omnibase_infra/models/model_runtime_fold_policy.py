# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-owned runtime fold values and compose activation (OMN-21288)."""

from __future__ import annotations

import json
import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_core.constants.constants_runtime_profiles import (
    REGISTERED_RUNTIME_PROFILES,
)


class ModelRuntimeFoldPolicy(BaseModel):
    """Roles owned by a runtime and the existing compose services they replace."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    # The roles this process carries besides its own, so a lane runs
    # one runtime container instead of a container per role. Rendered as
    # `{PROFILE}_RUNTIME_{PROCESS}_FOLDED_PROFILES` only when declared; the
    # runtime reads it as ONEX_RUNTIME_FOLDED_PROFILES and the auto-wiring
    # ownership filter attaches the folded roles' contracts in this process.
    folded_profiles: tuple[str, ...] = ()
    # OMN-21288: profile -> existing compose service. The policy renderer feeds
    # both compose activation and deploy-agent retirement from this mapping.
    foldable_services: dict[str, str] = Field(default_factory=dict)
    # OMN-21288: compose services whose node main runs by default, with no
    # runtime profile of its own to fold (the claude hook-events projection).
    # Their compose service is gated off by the fold overlay, and the deploy
    # agent and deploy handler retire a container of the previous revision from
    # this one list when main takes over.
    retired_services: tuple[str, ...] = ()

    @field_validator("retired_services")
    @classmethod
    def _retired_services_are_compose_services(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not re.fullmatch(r"[a-z][a-z0-9-]*", service) for service in value):
            raise ValueError("retired_services must name compose services")
        if len(set(value)) != len(value):
            raise ValueError("retired_services must be unique")
        return value

    @model_validator(mode="after")
    def _retired_services_are_not_foldable(self) -> Self:
        if overlap := sorted(
            set(self.retired_services) & set(self.foldable_services.values())
        ):
            raise ValueError(
                f"retired_services {overlap} are already foldable services"
            )
        return self

    @field_validator("folded_profiles")
    @classmethod
    def _folded_profiles_are_registered(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = "runtime process folded_profiles must be unique"
            raise ValueError(msg)
        unknown = sorted(set(value) - REGISTERED_RUNTIME_PROFILES)
        if unknown:
            msg = (
                f"runtime process folded_profiles {unknown} are not registered "
                f"runtime profiles: {sorted(REGISTERED_RUNTIME_PROFILES)}"
            )
            raise ValueError(msg)
        return value

    @field_validator("foldable_services")
    @classmethod
    def _foldable_services_are_registered(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) - REGISTERED_RUNTIME_PROFILES:
            raise ValueError("foldable_services must name registered runtime profiles")
        if any(
            not re.fullmatch(r"[a-z][a-z0-9-]*", service) for service in value.values()
        ):
            raise ValueError("foldable_services must name compose services")
        if len(set(value.values())) != len(value):
            raise ValueError("foldable_services must name unique compose services")
        return value

    def render_fold_env(self, prefix: str) -> dict[str, str]:
        """Render both ownership and gates from one contract mapping."""
        env = {}
        if self.folded_profiles:
            env[f"{prefix}_FOLDED_PROFILES"] = ",".join(self.folded_profiles)
        if self.foldable_services:
            env[f"{prefix}_FOLDABLE_SERVICES"] = json.dumps(
                self.foldable_services, sort_keys=True, separators=(",", ":")
            )
        if self.retired_services:
            env[f"{prefix}_RETIRED_SERVICES"] = ",".join(self.retired_services)
        for role, service in self.foldable_services.items():
            key = f"{prefix}_{service.upper().replace('-', '_')}"
            for active in ("runtime", "full"):
                env[f"{key}_{active.upper()}_PROFILE"] = (
                    f"folded-{role}-{active}"
                    if role in self.folded_profiles
                    else active
                )
        return env
