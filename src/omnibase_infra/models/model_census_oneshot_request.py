# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One bounded sampling invocation of the existing collector."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_census_oneshot_policy import ModelCensusOneshotPolicy


class ModelCensusOneshotRequest(BaseModel):
    """One bounded sampling invocation of the existing collector."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    operation: Literal["census_oneshot.sample"] = "census_oneshot.sample"
    policy: ModelCensusOneshotPolicy
    script_directory: str
    arguments: tuple[str, ...] = ()
