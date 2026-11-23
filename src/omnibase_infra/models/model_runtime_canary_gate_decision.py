# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whether the dev roll may build a candidate, from its canary receipt (OMN-26463)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__: list[str] = ["ModelRuntimeCanaryGateDecision"]


class ModelRuntimeCanaryGateDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    reason: str
    receipt_path: str
