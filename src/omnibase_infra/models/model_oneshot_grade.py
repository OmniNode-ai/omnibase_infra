# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A finding, pending execution or completed/absent observation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_pending_oneshot import ModelPendingOneshot


class ModelOneshotGrade(BaseModel):
    """A finding, pending execution or completed/absent observation."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    kind: str = ""
    detail: str = ""
    pending: ModelPendingOneshot | None = None
