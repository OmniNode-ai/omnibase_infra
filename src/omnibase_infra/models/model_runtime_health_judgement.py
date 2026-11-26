# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a /health reading says about the runtime (OMN-18640, OMN-20864)."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_answering_degraded import ModelAnsweringDegraded


class ModelRuntimeHealthJudgement(BaseModel):
    """Why the reading is not a pass (empty when it is) and any degraded monitor."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    reason: str
    answering_degraded: ModelAnsweringDegraded | None
