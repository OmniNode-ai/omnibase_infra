# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Inert RSD live-delegation preflight result."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.enums.enum_rsd_live_delegation_preflight_failure import (
    EnumRsdLiveDelegationPreflightFailure,
)
from omnibase_infra.runtime.models.model_rsd_live_delegation_result_anchor import (
    ModelRsdLiveDelegationResultAnchor,
)
from omnibase_infra.runtime.models.rsd_live_delegation_schema import (
    CanonicalCapabilityRef,
)


class ModelRsdLiveDelegationPreflightResult(BaseModel):
    """A permanently not-ready decision with explicit fail-closed facts."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    ready: Literal[False] = False
    failures: tuple[EnumRsdLiveDelegationPreflightFailure, ...] = Field(
        default_factory=tuple
    )
    missing_capability_refs: tuple[CanonicalCapabilityRef, ...] = Field(
        default_factory=tuple
    )
    result_anchor: ModelRsdLiveDelegationResultAnchor | None = None


__all__ = ["ModelRsdLiveDelegationPreflightResult"]
