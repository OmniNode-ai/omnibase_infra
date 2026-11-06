# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host-local estimator build request; credentials never enter the request."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelCostEstimatorBuildRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["build_estimator"] = "build_estimator"
    lane: str
    enabled: bool
    image: str | None = None
