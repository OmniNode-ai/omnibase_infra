# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Placement policy a runtime applies to one loaded node (OMN-21421).

Mirrors the ``placement`` block of the node contract (omnibase_core
``ModelContractPlacement``, OMN-21420). That model is merged on omnibase_core
dev but not in a released version this repository can pin yet; reading it from
the contract replaces this model once the release train publishes it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelPlacementPolicy(BaseModel):
    """Whether every runtime or only the elected primary handles a node."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    mode: Literal["any", "primary"]
    failover_target_s: float = Field(default=10.0, gt=0, allow_inf_nan=False)


__all__ = ["ModelPlacementPolicy"]
