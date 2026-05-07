# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Compatibility subset of the core per-artifact enforcement model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

EnforcementTier = Literal["observe", "warn", "block", "fail-fast"]


class ModelArtifactEnforcement(BaseModel):
    """Compatibility subset of the core per-artifact enforcement model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default: EnforcementTier = "block"
    non_matching_scope: EnforcementTier = "observe"
    missing_dependency: EnforcementTier = "observe"


__all__ = ["ModelArtifactEnforcement"]
