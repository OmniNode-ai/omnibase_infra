# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev recreate window decision (OMN-25912)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_dev_recreate_reason import EnumDevRecreateReason
from omnibase_infra.enums.enum_dev_recreate_verdict import EnumDevRecreateVerdict


class ModelDevRecreateDecision(BaseModel):
    """Recreate now, or hold until ``next_recreate_at``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumDevRecreateVerdict
    reason: EnumDevRecreateReason
    next_recreate_at: datetime
