# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Names and revision only: no environment or command output in the receipt."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelCostEstimatorBuildResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["built", "current", "not-applicable"]
    revision: str | None = None
    image: str | None = None
