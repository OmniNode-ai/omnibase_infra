# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed policy exclusion from contract discovery (OMN-18713)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_secret_gated_membership import (
    ModelSecretGatedMembership,
)


class ModelDiscoverySkip(BaseModel):
    """A parsed contract excluded by runtime policy, rather than a failure."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract_path: Path = Field(..., description="Excluded contract.yaml path")
    reason: Literal[
        "inactive_runtime_package", "dormant_cloud_gateway", "required_secret_absent"
    ] = Field(..., description="Machine-readable policy exclusion reason")
    message: str = Field(..., description="Explanation of the policy exclusion")
    membership: tuple[ModelSecretGatedMembership, ...] = Field(
        default_factory=tuple,
        description="Membership records for a required_secret_absent skip (OMN-21423); empty for other reasons",
    )
